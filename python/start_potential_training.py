"""Validate all games, assess the migrated source, then continue one learner."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import socket
import traceback

import torch

from auto_continue_training import Supervisor, live_trainers, verified_checkpoint, write_json
from dodge_reward_trial import DodgeRewardTrial, TrialStopped
from game_state import _parse_frame
from policy_runtime import ROOT


def check_endpoint(endpoint):
    """Require real new-Mod data; zero-filled legacy replies are rejected."""
    host, port = endpoint.rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=5) as sock:
        sock.settimeout(5)
        with sock.makefile("rwb", buffering=0) as stream:
            stream.write(b"pos\n")
            reply = stream.readline(65536).decode().strip()
            frame = _parse_frame(reply)
    if not frame.dash_state_valid or frame.knight_health <= 0 or frame.knight_box is None:
        raise RuntimeError(f"{endpoint}: load save 1 with the dash_v1 Mod before training")
    if any(h.hazard_type is None for h in frame.hazards):
        raise RuntimeError(f"{endpoint}: missing hazard type telemetry")
    return {"endpoint": endpoint, "dash_state_valid": frame.dash_state_valid,
            "health": frame.knight_health, "mana": frame.mana,
            "can_dash_now": frame.can_dash_now, "is_shadow_dashing": frame.is_shadow_dashing,
            "shadow_cooldown_remaining": frame.shadow_cooldown_remaining,
            "shadow_dash_remaining": frame.shadow_dash_remaining,
            "invulnerability_time_valid": frame.invulnerability_time_valid}


def run(directory):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    source = Path(config["initial_checkpoint"])
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != config["initial_sha256"]:
        raise ValueError("Prepared potential checkpoint changed")
    verified_checkpoint(payload, "radiant_potential_v1")
    supervisor = Supervisor(directory, continue_checkpoint=source,
                            reward_profile="radiant_potential_v1", episode_time_limit=150)
    lock = directory / "auto_continue.lock"
    if supervisor.stopped():
        supervisor.status("disabled")
        return
    with lock.open("x", encoding="utf-8") as stream:
        import os
        json.dump({"pid": os.getpid(), "purpose": "potential_baseline_and_training"}, stream)
    try:
        if live_trainers():
            raise RuntimeError("An existing learner still owns the games")
        supervisor.status("verifying_four_updated_games")
        with ThreadPoolExecutor(max_workers=4) as executor:
            checks = list(executor.map(check_endpoint, config["endpoints"]))
        write_json(directory / "telemetry_validation.json", {"valid": True, "windows": checks})
        write_json(ROOT / "python/runs/traitor_4env_iframes_fixed/active_experiment.json",
                   {"directory": str(directory), "kind": "potential_baseline_then_training"})
        baseline_path = directory / "baseline_source_summary.json"
        fights_path = directory / "baseline_source_fights.jsonl"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            if (not baseline.get("valid") or baseline.get("checkpoint_sha256") != digest
                    or baseline.get("episodes") != 40
                    or baseline.get("episode_time_limit") != 150
                    or baseline.get("observation_time_limit") != 60):
                raise RuntimeError("Preserved baseline does not match the current source")
        else:
            if fights_path.exists():
                raise RuntimeError("Incomplete prior baseline exists; preserve and inspect it before restarting")
            supervisor.status("evaluating_migrated_source", episodes=40)
            evaluator = DodgeRewardTrial(directory, source, digest,
                                         config["entropy_coefficient"], config["endpoints"],
                                         supervisor.stopped, episode_time_limit=150)
            baseline = evaluator.evaluate(source, "baseline_source")
        supervisor.continuation_assessment = baseline
        supervisor.continuation_fights = fights_path
        write_json(directory / "progress.json", {"phase": "baseline_complete_preparing_training",
                                                "active": None, "baseline": baseline})
        # Direct continuation preserves the user's requested objective and
        # never reruns the earlier reward acceptance/fallback experiment.
        supervisor.run()
    except TrialStopped:
        supervisor.status("disabled")
    except Exception as error:
        supervisor.status("potential_start_failed", error=repr(error))
        (directory / "startup_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.experiment_dir.resolve())


if __name__ == "__main__":
    mp.freeze_support()
    main()
