"""Play a frozen Radiant policy on one game, stopping after a flawless win."""

import argparse
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path

from policy_runtime import ROOT, load_checkpoint, stage_fresh_fight
import torch
from agent.action import BUTTON_ORDER, NUM_BUTTONS
from evaluation import evaluate_pool
from main_multi import CHECKPOINT_FORMAT, ExternalEnvPool, prepare_fight
from obs_norm import RunningNorm
from policy_selection import is_flawless
from ppo_model import ActorCritic


DEFAULT_CHECKPOINT = ROOT / "models/traitor_lord_radiant.pth"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or not 1 <= args.port <= 65535:
        parser.error("episodes must be positive and port must be valid")
    saved, digest = load_checkpoint(args.checkpoint)
    if (saved.get("format_version") != CHECKPOINT_FORMAT
            or saved.get("action_distribution") != "exclusive_directions_v1"
            or saved.get("action_names") != list(BUTTON_ORDER)):
        raise ValueError("Checkpoint does not match the current button policy")
    torch.set_num_threads(2)
    model = ActorCritic(saved["observation_size"], [1] * NUM_BUTTONS,
                        hidden=saved["model_hidden"], bernoulli=True,
                        exclusive_directions=True)
    model.load_state_dict(saved["model"])
    model.eval()
    normalizer = RunningNorm(saved["observation_size"])
    normalizer.load_state_dict(saved["obs_norm"])
    normalization_before = json.dumps(normalizer.state_dict(),
                                      default=lambda x: x.tolist(), sort_keys=True)
    provenance = {
        "checkpoint": str(args.checkpoint.resolve()), "sha256": digest,
        "update": saved["update"], "port": args.port, "boss_level": 2,
        "game_speed": 1.0, "episode_time_limit": 120.0,
        "observation_time_limit": 60.0, "action_selection": "greedy",
        "control_hz": int(saved.get("control_hz", 10)),
    }
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
        pool = ExternalEnvPool([endpoint], provenance["control_hz"], saved["frame_stack"],
                               120.0, 1.0, "radiant_spacing_v1", 2,
                               observation_time_limit=60.0)
        with torch.inference_mode():
            for episode in range(1, args.episodes + 1):
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
                print(f"Fight {episode}/{args.episodes}: won={info['won']} "
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
