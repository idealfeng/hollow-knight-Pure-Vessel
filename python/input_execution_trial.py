"""Frozen-policy comparisons of isolated input-dispatch changes; never train."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import statistics
import subprocess
import traceback
from zoneinfo import ZoneInfo

import torch

from auto_continue_training import verified_checkpoint, write_json
from dodge_reward_trial import DodgeRewardTrial, TrialStopped
from policy_runtime import ROOT
from policy_selection import is_flawless


PROFILES = ("legacy_v1", "repeat_dash_v1", "soul_guard_v1")


def existing_controllers():
    command = ("Get-CimInstance Win32_Process | Where-Object { "
               "$_.Name -match '^python(w)?\\.exe$' -and "
               f"$_.ProcessId -notin @({os.getpid()},{os.getppid()}) -and "
               "$_.CommandLine -match 'main_multi\\.py|play\\.py|input_execution_trial\\.py|"
               "auto_continue_training\\.py|start_potential_training\\.py' } | "
               "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else []


def freeze_endpoint(endpoint):
    host, port = endpoint.rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=5) as sock:
        sock.settimeout(5)
        with sock.makefile("rwb", buffering=0) as stream:
            result = {}
            for command in ("CLEAR", "SPEED 0"):
                stream.write((command + "\n").encode("ascii"))
                result[command] = stream.readline(65536).decode().strip()
                if result[command] != "ok":
                    raise RuntimeError(f"{endpoint}: {command}: {result[command]}")
            return {"endpoint": endpoint, **result}


def run(source, directory, endpoints):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    payload = source.read_bytes()
    saved = torch.load(source, map_location="cpu", weights_only=False)
    reward = saved["training_config"]["reward_profile"]
    verified_checkpoint(payload, reward)
    if (saved["frame_stack"] != 4 or saved["observation_size"] != 472
            or saved["update"] != 550):
        raise ValueError("This controlled comparison requires the original four-frame update550")
    if len(set(endpoints)) != 4:
        raise ValueError("Exactly four distinct game endpoints are required")
    if existing_controllers():
        raise RuntimeError("A learner or player still owns the games")
    directory.mkdir(parents=True, exist_ok=False)
    backup = directory / "source_update550.pth"
    backup.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    plan = dict(source_checkpoint=str(source), frozen_checkpoint=str(backup),
        checkpoint_sha256=digest, update=550, frame_stack=4,
        endpoints=endpoints, profiles=list(PROFILES), fights_per_profile=40,
        fights_per_endpoint=10, episode_time_limit=150, observation_time_limit=60,
        reward_profile=reward, control_hz=10, game_speed=1, action_selection="greedy",
        caveat="Sequential 40-fight screens, not a guarantee of statistical significance")
    write_json(directory / "experiment.json", plan)
    write_json(ROOT / "python/runs/recordings/active_execution_trial.json", {"directory": str(directory)})
    write_json(ROOT / "python/runs/traitor_4env_iframes_fixed/active_experiment.json",
               {"directory": str(directory), "kind": "frozen_input_execution_comparison"})
    stop = lambda: (directory / "STOP").exists()
    evaluator = DodgeRewardTrial(directory, backup, digest, .005, endpoints, stop,
                                 episode_time_limit=150)
    summaries = []
    try:
        for profile in PROFILES:
            if existing_controllers():
                raise RuntimeError("Another controller appeared before a comparison stage")
            summary = evaluator.evaluate(backup, profile, execution_profile=profile, reward_profile=reward)
            fights = [json.loads(line) for line in (directory / f"{profile}_fights.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
            by_endpoint = []
            for index, endpoint in enumerate(endpoints):
                group = [row for row in fights if row["env"] == index]
                by_endpoint.append(dict(endpoint=endpoint, fights=len(group),
                    nohit_wins=sum(is_flawless(row["info"]) for row in group),
                    mean_min_boss_hp=statistics.mean(row["info"]["min_boss_health"] for row in group),
                    best_min_boss_hp=min(row["info"]["min_boss_health"] for row in group),
                    timeouts=sum(row["info"].get("truncated_reason") == "timeout" for row in group)))
            summary["by_endpoint"] = by_endpoint
            write_json(directory / f"{profile}_summary.json", summary)
            summaries.append(summary)
            write_json(directory / "comparison_partial.json", {"completed": summaries})
        if source.read_bytes() != payload or backup.read_bytes() != payload:
            raise RuntimeError("Frozen source checkpoint changed")
        write_json(directory / "comparison.json", dict(valid=True, checkpoint_unchanged=True,
            normalization_unchanged=True, trained=False, summaries=summaries,
            next_step="Review endpoint-stratified results; do not automatically promote a profile"))
        evaluator.status("complete", active=None)
    except TrialStopped:
        evaluator.status("stopped", active=None)
    except Exception as error:
        evaluator.status("failed", error=repr(error))
        (directory / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if not existing_controllers():
            with ThreadPoolExecutor(max_workers=4) as executor:
                frozen = list(executor.map(freeze_endpoint, endpoints))
            write_json(directory / "games_paused_after_trial.json", {"games": frozen})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--endpoints", nargs=4,
                        default=["127.0.0.1:9001", "127.0.0.1:9002", "192.168.0.184:9003", "192.168.0.184:9004"])
    args = parser.parse_args()
    directory = args.run_dir or ROOT / "python/runs" / ("input_execution_trial_" +
        datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S"))
    run(args.checkpoint.resolve(), directory.resolve(), args.endpoints)


if __name__ == "__main__":
    mp.freeze_support()
    main()
