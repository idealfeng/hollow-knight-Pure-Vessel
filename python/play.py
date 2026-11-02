"""Play a frozen Radiant policy on one game, stopping after a flawless win."""

import argparse
from datetime import datetime
import json
from itertools import count
import math
import multiprocessing as mp
import os
from pathlib import Path

from policy_runtime import ROOT, load_checkpoint, stage_fresh_fight
import torch
from agent.action import BUTTON_ORDER, NUM_BUTTONS, SPELL_ACTION_NAMES, EXECUTION_PROFILES
from macro_actions import (MACRO_ACTIONS, MACRO_DISTRIBUTION, MACRO_EXECUTION_PROFILE,
                           macro_metadata, validate_macro_checkpoint)
from evaluation import evaluate_pool
from main_multi import CHECKPOINT_FORMAT, ExternalEnvPool, prepare_fight
from obs_norm import RunningNorm
from policy_selection import is_flawless
from ppo_model import ActorCritic
from observation_schema import LEGACY_SCHEMA


DEFAULT_CHECKPOINT = ROOT / "models/traitor_lord_radiant.pth"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--episodes", type=int, default=100,
                        help="maximum fights; 0 retries indefinitely until a flawless win")
    parser.add_argument("--episode-time-limit", type=float, default=None,
                        help="fight cutoff; defaults to the checkpoint's training limit")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--reward-profile", default=None)
    parser.add_argument("--execution-profile", choices=EXECUTION_PROFILES, default=None)
    args = parser.parse_args()
    if args.episodes < 0 or not 1 <= args.port <= 65535:
        parser.error("episodes must be non-negative and port must be valid")
    saved, digest = load_checkpoint(args.checkpoint)
    training_config = saved.get("training_config", {})
    episode_time_limit = (args.episode_time_limit if args.episode_time_limit is not None
                          else training_config.get("episode_time_limit", 120.0))
    observation_time_limit = training_config.get("observation_time_limit") or 60.0
    if any(not math.isfinite(value) or value <= 0
           for value in (episode_time_limit, observation_time_limit)):
        parser.error("fight and observation time limits must be finite and positive")
    spells = saved.get("action_distribution") == "exclusive_directions_spells_v1"
    macros = saved.get("action_distribution") == MACRO_DISTRIBUTION
    action_names = ("macro",) if macros else SPELL_ACTION_NAMES if spells else BUTTON_ORDER
    action_sizes = [len(MACRO_ACTIONS)] if macros else [1] * NUM_BUTTONS + [4] if spells else [1] * NUM_BUTTONS
    execution_profile = (args.execution_profile or saved.get("execution_profile")
                         or training_config.get("execution_profile")
                         or (MACRO_EXECUTION_PROFILE if macros else "legacy_v1"))
    if macros:
        validate_macro_checkpoint(saved)
    if (saved.get("format_version") != (10 if macros else 9 if spells else CHECKPOINT_FORMAT)
            or saved.get("action_distribution") not in ("exclusive_directions_v1", "exclusive_directions_spells_v1", MACRO_DISTRIBUTION)
            or saved.get("action_names") != list(action_names)
            or saved.get("action_sizes") != action_sizes):
        raise ValueError("Checkpoint does not match the current button policy")
    torch.set_num_threads(2)
    model = ActorCritic(saved["observation_size"], action_sizes,
                        hidden=saved["model_hidden"], bernoulli=not macros,
                        exclusive_directions=not macros)
    model.load_state_dict(saved["model"])
    model.eval()
    normalizer = RunningNorm(saved["observation_size"])
    normalizer.load_state_dict(saved["obs_norm"])
    normalization_before = json.dumps(normalizer.state_dict(),
                                      default=lambda x: x.tolist(), sort_keys=True)
    provenance = {
        "checkpoint": str(args.checkpoint.resolve()), "sha256": digest,
        "update": saved["update"], "port": args.port, "boss_level": 2,
        "game_speed": 1.0, "episode_time_limit": float(episode_time_limit),
        "observation_time_limit": float(observation_time_limit), "action_selection": "greedy",
        "episodes_limit": args.episodes, "retry_until_flawless": args.episodes == 0,
        "control_hz": int(saved.get("control_hz", 10)),
        "action_distribution": saved["action_distribution"],
        "observation_schema": saved.get("observation_schema", LEGACY_SCHEMA),
        "execution_profile": execution_profile,
        **(macro_metadata() if macros else {}),
    }
    reward_profile = (args.reward_profile or training_config.get("reward_profile")
                      or ("radiant_spells_v1" if spells else "radiant_spacing_v1"))
    provenance["reward_profile"] = reward_profile
    if macros and (reward_profile == "radiant_potential_v1"
                   or provenance["observation_schema"] != LEGACY_SCHEMA
                   or saved.get("frame_stack") != 16 or saved.get("observation_size") != 1888):
        raise ValueError("Macro playback requires legacy16 observations and no potential shaping")
    if execution_profile not in EXECUTION_PROFILES:
        raise ValueError("Checkpoint has an unknown execution profile")
    print(json.dumps(provenance), flush=True)
    if args.check_only:
        print("Checkpoint and normalizer loaded. No game connection made.", flush=True)
        return

    directory = ROOT / "python/runs/recordings" / datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True)
    (directory / "session.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    os.environ["HOLLOWRL_FAST_RESET"] = "auto"
    endpoint = ("127.0.0.1", args.port)
    pool = None
    wins = 0
    completed = 0
    try:
        stage_fresh_fight(endpoint, 2)
        prepare_fight(*endpoint)
        pool_options = {"observation_time_limit": observation_time_limit}
        if spells or macros:
            pool_options["action_distribution"] = saved["action_distribution"]
        if provenance["observation_schema"] != LEGACY_SCHEMA:
            pool_options["observation_schema"] = provenance["observation_schema"]
        if execution_profile != "legacy_v1":
            pool_options["execution_profile"] = execution_profile
        pool = ExternalEnvPool([endpoint], provenance["control_hz"], saved["frame_stack"],
                               episode_time_limit, 1.0, reward_profile, 2, **pool_options)
        with torch.inference_mode():
            for episode in count(1):
                if args.episodes and episode > args.episodes:
                    break
                def record(result):
                    result["recording_episode"] = episode
                    with (directory / "fights.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(result) + "\n")

                result = evaluate_pool(pool, model, normalizer.normalize, 1, record)[0]
                info = result["info"]
                if (info.get("boss_level") != 2 or info.get("reset_mode") != "scene"
                        or info.get("truncated_reason") in ("disconnect", "fight_lost")):
                    raise RuntimeError(f"Invalid Radiant fight: {info}")
                completed += 1
                flawless = is_flawless(info)
                wins += int(flawless)
                budget = args.episodes or "until flawless"
                print(f"Fight {episode}/{budget}: won={info['won']} "
                      f"flawless={flawless} boss_hp={info['min_boss_health']} "
                      f"steps={result['steps']}", flush=True)
                if flawless:
                    print("FLAWLESS RADIANT WIN! Stopping retries. You can stop recording.", flush=True)
                    break
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C.", flush=True)
    finally:
        if pool is not None:
            pool.close()
        unchanged = digest == load_checkpoint(args.checkpoint)[1]
        normalization_unchanged = normalization_before == json.dumps(
            normalizer.state_dict(), default=lambda x: x.tolist(), sort_keys=True)
        (directory / "summary.json").write_text(json.dumps({
            **provenance, "completed": completed, "flawless_wins": wins,
            "checkpoint_unchanged": unchanged,
            "normalizer_unchanged": normalization_unchanged,
        }, indent=2), encoding="utf-8")
        print(f"Results: {directory}", flush=True)
        if not unchanged or not normalization_unchanged:
            raise RuntimeError("Frozen policy verification failed")


if __name__ == "__main__":
    mp.freeze_support()
    main()
