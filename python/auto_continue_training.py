"""Locally supervise a finite experiment and continue its verified policy.

This is a rule-based local watcher, not a scheduled language-model session.
Place STOP_AUTO_CONTINUE in the experiment directory to relinquish control.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch
from agent.action import SPELL_ACTION_NAMES
from agent.reward import get_reward_config
from policy_selection import is_flawless, nohit_selection_score
from policy_runtime import stage_fresh_fight
from observation_schema import LEGACY_SCHEMA, COMBAT_SCHEMA, observation_width

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def live_trainers():
    """Inspect process metadata only; never open a game socket for monitoring."""
    command = (
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.Name -match '^python(w)?\\.exe$' -and $_.CommandLine -match 'main_multi\\.py' "
        "} | Select-Object ProcessId,ParentProcessId | ConvertTo-Json -Compress"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                            capture_output=True, text=True, timeout=30, check=True)
    if not result.stdout.strip():
        return []
    records = json.loads(result.stdout)
    return records if isinstance(records, list) else [records]


def choose_checkpoint(directory, selection=None):
    """Adopt low entropy only after the trial's independent confirmation."""
    config = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    source = Path(config["source_best"]["path"]).resolve()
    if not source.is_relative_to(directory.resolve()):
        raise ValueError("Source checkpoint is outside this experiment")
    source_hash = config["source_best"]["sha256"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Backed-up source checkpoint changed")
    if not selection or not selection.get("low_entropy_qualified"):
        return source, source_hash, 0.005, "Retain source 425; no independently confirmed low-entropy improvement"
    baseline, control, low = (selection[k] for k in ("baseline", "control", "low_entropy"))
    confirmation = selection.get("confirmation")
    if not confirmation:
        raise ValueError("Candidate has no independent confirmation")
    lc, sc = confirmation["low"], confirmation["source"]
    if not all(r.get("valid") and r.get("episodes") == 40 for r in (baseline, control, low, lc, sc)):
        raise ValueError("Incomplete assessment cannot select a new candidate")
    qualified = (low["nohit_wins"] >= control["nohit_wins"] + 4
                 and low["nohit_wins"] >= baseline["nohit_wins"]
                 and low["mean_min_boss_hp"] <= control["mean_min_boss_hp"] + 65
                 and lc["nohit_wins"] >= sc["nohit_wins"] + 2
                 and lc["mean_min_boss_hp"] <= sc["mean_min_boss_hp"] + 65)
    if not qualified:
        raise ValueError("Candidate qualification does not match recorded results")
    candidate = (directory / "low_entropy_001/ppo_checkpoint.pth").resolve()
    if Path(selection["selected_checkpoint"]).resolve() != candidate:
        raise ValueError("Selection points at an unexpected checkpoint")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if digest != low["checkpoint_sha256"] or digest != lc["checkpoint_sha256"]:
        raise ValueError("Candidate differs from its assessed weights")
    return candidate, digest, 0.001, "Low entropy passed screening and independent confirmation"


def verified_checkpoint(payload, reward_profile="radiant_spells_v1"):
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    config = saved.get("training_config", {})
    schema = saved.get("observation_schema", LEGACY_SCHEMA)
    frame_stack = saved.get("frame_stack")
    if not isinstance(frame_stack, int) or frame_stack not in (4, 16):
        raise ValueError("Selected checkpoint must use the verified 4 or 16 frame stack")
    expected_size = observation_width(schema) * frame_stack
    if (saved.get("format_version") != 9
            or saved.get("action_distribution") != "exclusive_directions_spells_v1"
            or saved.get("action_sizes") != [1] * 7 + [4]
            or saved.get("action_names") != list(SPELL_ACTION_NAMES)
            or saved.get("observation_size") != expected_size
            or config.get("observation_schema", schema) != schema
            or config.get("frame_stack", frame_stack) != frame_stack
            or config.get("observation_size", expected_size) != expected_size
            or saved.get("control_hz") != 10
            or not saved.get("freeze_obs_norm")
            or "optimizer" not in saved
            or reward_profile not in ("radiant_spells_v1", "radiant_spells_dodge_v1", "radiant_potential_v1")
            or config.get("reward_profile") != reward_profile
            or config.get("boss_level") != 2
            or config.get("reward_config") != asdict(get_reward_config(reward_profile))
            or config.get("episode_time_limit") not in (120, 150) or config.get("observation_time_limit") != 60
            or config.get("game_speed") != 1 or config.get("train_mode") != "all"):
        raise ValueError("Selected full checkpoint does not match the verified 21-damage spell run")
    if reward_profile == "radiant_potential_v1" and schema != COMBAT_SCHEMA:
        raise ValueError("Potential checkpoint requires combat_v2 observations")
    return saved


def validate_dodge_selection(result, directory, original_digest, entropy, episode_time_limit=None):
    """Recheck promotion evidence before changing the continuation objective."""
    if result["entropy_coefficient"] != entropy:
        raise ValueError("Reward trial changed the entropy coefficient")
    if result["reward_profile"] == "radiant_spells_v1":
        if result["selected_sha256"] != original_digest:
            raise ValueError("Unqualified reward trial must preserve its original source")
        return
    from dodge_reward_trial import qualifies
    if result["reward_profile"] != "radiant_spells_dodge_v1" or not result.get("dodge_reward_qualified"):
        raise ValueError("New reward has no confirmed qualification")
    confirmation = result.get("confirmation") or {}
    candidate, control = result.get("candidate", {}), result.get("control", {})
    cc, rc = confirmation.get("candidate", {}), confirmation.get("control", {})
    if not qualifies(candidate, control, 4) or not qualifies(cc, rc, 2):
        raise ValueError("Reward trial does not meet screening and confirmation thresholds")
    if episode_time_limit is not None and any(
            record.get("episode_time_limit") != episode_time_limit
            or record.get("observation_time_limit") != 60 for record in (candidate, control, cc, rc)):
        raise ValueError("Reward trial assessments use a different fight time limit")
    expected = (directory / "candidate/ppo_checkpoint.pth").resolve()
    if Path(result["selected_checkpoint"]).resolve() != expected:
        raise ValueError("Reward trial points at an unexpected candidate")
    if (candidate.get("checkpoint_sha256") != result["selected_sha256"]
            or cc.get("checkpoint_sha256") != result["selected_sha256"]
            or control.get("checkpoint_sha256") != rc.get("checkpoint_sha256")
            or result.get("assessment") != cc
            or Path(result.get("fights_path") or "").resolve()
            != (directory / "candidate_confirm_fights.jsonl").resolve()):
        raise ValueError("Reward candidate differs from its confirmed assessment")


class Supervisor:
    def __init__(self, directory, poll_seconds=30, total_updates=10000, max_restarts=2,
                 dodge_trial=False, episode_time_limit=150.0, continue_checkpoint=None,
                 reward_profile=None):
        if episode_time_limit not in (120.0, 150.0):
            raise ValueError("Continuation supports the verified 120 or requested 150 second horizon")
        if reward_profile is not None and continue_checkpoint is None:
            raise ValueError("An explicit reward override requires --continue-checkpoint")
        if continue_checkpoint is not None and dodge_trial:
            raise ValueError("Direct continuation does not rerun the completed reward comparison")
        self.directory = directory.resolve()
        self.poll_seconds = poll_seconds
        self.total_updates = total_updates
        self.max_restarts = max_restarts
        self.status_path = self.directory / "auto_continue_status.json"
        self.stop_file = self.directory / "STOP_AUTO_CONTINUE"
        self.dodge_trial = dodge_trial
        self.episode_time_limit = episode_time_limit
        self.continuation_profile = "radiant_spells_v1"
        self.continuation_assessment = None
        self.continuation_fights = None
        self.continue_checkpoint = Path(continue_checkpoint).resolve() if continue_checkpoint is not None else None
        self.requested_reward_profile = reward_profile
        self.allow_reward_change = False

    def status(self, phase, **details):
        write_json(self.status_path, {"phase": phase, "updated_at_unix": time.time(),
                                    "supervisor_pid": os.getpid(), **details})

    def stopped(self):
        return self.stop_file.exists()

    def archive_best(self, training):
        path = training / "best_eval_checkpoint.pth"
        if not path.exists():
            return
        payload = path.read_bytes()
        saved = verified_checkpoint(payload, self.continuation_profile)
        archive = training / "best_snapshots"
        archive.mkdir(exist_ok=True)
        digest = hashlib.sha256(payload).hexdigest()
        destination = archive / f"update_{saved['update']}_{digest[:12]}.pth"
        if not destination.exists():
            destination.write_bytes(payload)

    def prepare(self, source, digest, entropy, reason, selection):
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("Selected checkpoint changed before continuation")
        source_profile = (torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
                          .get("training_config", {}).get("reward_profile"))
        saved = verified_checkpoint(payload, source_profile)
        if source_profile != self.continuation_profile and not self.allow_reward_change:
            raise ValueError("Changing the reward requires explicit direct-continuation authorization")
        if self.total_updates <= saved["update"]:
            raise ValueError("Continuation budget must exceed the selected checkpoint update")
        envelope = self.directory / ("continuation_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        training = envelope / "training"
        backup = envelope / "backup"
        training.mkdir(parents=True, exist_ok=False)
        backup.mkdir()
        initial = backup / f"source_{saved['update']}.pth"
        initial.write_bytes(payload)
        if source_profile == self.continuation_profile:
            (training / "best_eval_checkpoint.pth").write_bytes(payload)
        assessed = None
        label = "baseline_425"
        fights_path = None
        if self.continuation_assessment is not None:
            assessed = dict(self.continuation_assessment)
            label = assessed["label"]
            fights_path = self.continuation_fights
        elif selection:
            if entropy == 0.001:
                assessed = dict(selection["confirmation"]["low"])
                label = assessed["label"]
            else:
                assessed = dict(selection["baseline"])
        if assessed:
            if not assessed.get("valid") or assessed.get("episodes") != 40 or assessed.get("checkpoint_sha256") != digest:
                raise ValueError("Initial best score does not match this verified checkpoint")
            fights_path = fights_path or self.directory / f"{label}_fights.jsonl"
            fights = [json.loads(line) for line in fights_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(fights) != 40:
                raise ValueError("Initial best score has incomplete fight records")
            if any(r["info"].get("boss_level") != 2 or r["info"].get("reset_mode") != "scene"
                   or not r["info"].get("damage_tracking_valid") for r in fights):
                raise ValueError("Invalid fights cannot seed the best score")
            wins = sum(is_flawless(r["info"]) for r in fights)
            hp = statistics.mean(r["info"]["min_boss_health"] for r in fights)
            if wins != assessed["nohit_wins"] or abs(hp - assessed["mean_min_boss_hp"]) > 1e-6:
                raise ValueError("Assessment summary disagrees with its fight records")
            assessed.update(event="eval_summary", mean_damage=statistics.mean(r["info"]["damage"] for r in fights),
                            nohit_rate=wins / 40, win_rate=statistics.mean(bool(r["info"]["won"]) for r in fights),
                            selection_objective="nohit")
            assessed["selection_score"] = list(nohit_selection_score(assessed))
        elif self.continue_checkpoint is None:
            assessed = json.loads((self.directory / "backup/best_eval.json").read_text())
        assessment_horizon = (assessed.get("episode_time_limit", saved["training_config"]["episode_time_limit"])
                              if assessed else None)
        write_json(backup / "source_assessment.json", {**(assessed or {}),
                                                     "episode_time_limit": assessment_horizon,
                                                     "assessed": assessed is not None})
        # A longer fight budget changes the assessment conditions. Preserve
        # historical evidence without comparing its score to fresh 150s runs.
        if assessed and assessment_horizon == self.episode_time_limit and source_profile == self.continuation_profile:
            write_json(training / "best_eval.json", assessed)
        endpoints = json.loads((self.directory / "experiment.json").read_text())["endpoints"]
        arguments = [str(ROOT / "python/.venv/Scripts/python.exe"), "-u", str(ROOT / "python/main_multi.py"), "--endpoints"]
        arguments += endpoints
        arguments += ["--run-dir", str(training), "--init-checkpoint", str(initial),
                      "--action-distribution", "exclusive_directions_spells_v1", "--reward-profile", self.continuation_profile,
                      "--boss-level", "2", "--eval-objective", "nohit", "--episode-time-limit", str(self.episode_time_limit),
                      "--observation-time-limit", "60", "--control-hz", "10", "--frame-stack", str(saved["frame_stack"]),
                      "--game-speed", "1", "--lr", "0.0001", "--ent-coef", str(entropy),
                      "--freeze-obs-norm", "--steps-per-env", "512", "--total-updates", str(self.total_updates),
                      "--eval-every", "25", "--eval-episodes", "10", "--no-monitor"]
        decision = {"source_checkpoint": str(source), "source_sha256": digest, "source_update": saved["update"],
                    "reason": reason, "entropy_coefficient": entropy,
                    "reward_changed": self.continuation_profile != "radiant_spells_v1",
                    "reward_profile": self.continuation_profile,
                    "reward_config": asdict(get_reward_config(self.continuation_profile)),
                    "episode_time_limit": self.episode_time_limit, "observation_time_limit": 60,
                    "source_reward_profile": source_profile,
                    "user_requested_reward_override": self.requested_reward_profile is not None,
                    "best_score_seeded": bool(assessed and assessment_horizon == self.episode_time_limit
                                              and source_profile == self.continuation_profile),
                    "directory": str(envelope), "training_directory": str(training), "arguments": arguments}
        write_json(self.directory / "continuation_decision.json", decision)
        write_json(envelope / "launch.json", decision)
        return envelope, training, arguments

    def wait_until_idle(self, phase):
        while not self.stopped():
            trainers = live_trainers()
            if not trainers:
                # Let disconnected reset owners finish cleanup after the finite
                # runner closes its workers; do not probe the game sockets.
                time.sleep(3)
                if self.stopped():
                    return False
                if not live_trainers():
                    return True
                continue
            self.status(phase, existing_trainers=trainers)
            time.sleep(self.poll_seconds)
        return False

    def stage(self):
        """Take over only after the previous trainer has relinquished all games."""
        config = json.loads((self.directory / "experiment.json").read_text())
        endpoints = [(item.rsplit(":", 1)[0], int(item.rsplit(":", 1)[1])) for item in config["endpoints"]]
        with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
            list(executor.map(lambda endpoint: stage_fresh_fight(endpoint, 2), endpoints))

    def run(self, max_wait_hours=12):
        deadline = time.monotonic() + max_wait_hours * 3600
        selection = None
        while self.continue_checkpoint is None and not self.stopped() and time.monotonic() < deadline:
            try:
                progress = json.loads((self.directory / "progress.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                time.sleep(self.poll_seconds)
                continue
            phase = progress.get("phase")
            if phase == "complete_comparison_stopped":
                selection = json.loads((self.directory / "selection.json").read_text(encoding="utf-8"))
                break
            if phase == "failed":
                # A failed experiment cannot promote a candidate. The user has
                # authorized continued training from a previously verified source.
                self.status("comparison_failed_using_preserved_source", error=progress.get("error"))
                break
            self.status("waiting_for_comparison", comparison_phase=phase, active_branch=progress.get("active"),
                        pending_dodge_reward_trial=self.dodge_trial,
                        next_episode_time_limit=self.episode_time_limit)
            time.sleep(self.poll_seconds)
        else:
            if self.continue_checkpoint is None or self.stopped():
                self.status("disabled" if self.stopped() else "comparison_wait_limit_reached")
                return
        if not self.wait_until_idle("waiting_for_previous_trainer_exit"):
            self.status("disabled")
            return
        if self.continue_checkpoint is not None:
            source = self.continue_checkpoint
            payload = source.read_bytes()
            source_config = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False).get("training_config", {})
            source_profile = source_config.get("reward_profile")
            verified_checkpoint(payload, source_profile)
            digest = hashlib.sha256(payload).hexdigest()
            entropy = source_config["ent_coef"]
            if entropy not in (0.005, 0.001) or source_config.get("lr") != 0.0001:
                raise ValueError("Direct continuation must preserve the verified learning rate and entropy")
            self.continuation_profile = self.requested_reward_profile or source_profile
            get_reward_config(self.continuation_profile)
            self.allow_reward_change = self.requested_reward_profile is not None
            reason = "User explicitly requested direct continuation; preserve full weights, optimizer and normalization"
            if self.requested_reward_profile is not None:
                reason += "; retain requested reward instead of repeating automatic comparison/fallback"
        else:
            source, digest, entropy, reason = choose_checkpoint(self.directory, selection)
        if self.dodge_trial:
            from dodge_reward_trial import DodgeRewardTrial, TrialStopped
            trial_directory = self.directory / ("dodge_reward_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
            endpoints = json.loads((self.directory / "experiment.json").read_text())["endpoints"]
            self.status("running_dodge_reward_comparison", trial_directory=str(trial_directory),
                        source_checkpoint=str(source), source_sha256=digest, entropy_coefficient=entropy)
            write_json(ROOT / "python/runs/traitor_4env_iframes_fixed/active_experiment.json",
                       {"directory": str(trial_directory), "kind": "finite_phase_dodge_reward_trial"})
            preserved = source, digest, entropy, reason
            try:
                result = DodgeRewardTrial(trial_directory, source, digest, entropy, endpoints, self.stopped,
                                          episode_time_limit=self.episode_time_limit).run()
                validate_dodge_selection(result, trial_directory, digest, entropy, self.episode_time_limit)
                selected = Path(result["selected_checkpoint"])
                payload = selected.read_bytes()
                if hashlib.sha256(payload).hexdigest() != result["selected_sha256"]:
                    raise ValueError("Reward-trial selection changed before continuation")
                profile = result["reward_profile"]
                verified_checkpoint(payload, profile)
                if profile == "radiant_spells_dodge_v1":
                    if not result.get("assessment") or not result.get("fights_path"):
                        raise ValueError("New reward has no confirmed assessment/fight records")
                    self.continuation_assessment = result["assessment"]
                    self.continuation_fights = Path(result["fights_path"])
                new_reason = reason + "; " + result["reason"]
                source, digest, reason = selected, result["selected_sha256"], new_reason
                self.continuation_profile = profile
            except TrialStopped:
                self.status("disabled")
                return
            except Exception as error:
                (self.directory / "dodge_reward_error.log").write_text(traceback.format_exc(), encoding="utf-8")
                self.status("reward_trial_failed_using_preserved_source", error=repr(error),
                            trial_directory=str(trial_directory))
                self.continuation_profile = "radiant_spells_v1"
                self.continuation_assessment = self.continuation_fights = None
                source, digest, entropy, reason = preserved
                reason += "; Reward trial failed; retain verified source and original reward"
            if self.stopped():
                self.status("disabled")
                return
        envelope, training, arguments = self.prepare(source, digest, entropy, reason, selection)
        active = ROOT / "python/runs/traitor_4env_iframes_fixed/active_experiment.json"
        write_json(active, {"directory": str(envelope), "kind": "supervised_spell_continuation"})
        environment = dict(os.environ)
        environment.pop("HOLLOWRL_TRACE_DIR", None)
        for attempt in range(self.max_restarts + 1):
            while True:
                if not self.wait_until_idle("waiting_to_avoid_duplicate_trainer"):
                    self.status("disabled")
                    return
                if not live_trainers():
                    break
            if self.stopped():
                self.status("disabled")
                return
            try:
                self.status("preparing_four_games", training_directory=str(training), restart_attempt=attempt)
                self.stage()
            except Exception as error:
                self.status("game_preparation_failed", error=repr(error), restart_attempt=attempt)
                if attempt < self.max_restarts:
                    time.sleep(30)
                    continue
                break
            if self.stopped():
                self.status("disabled")
                return
            trainers = live_trainers()
            if trainers:
                self.status("external_trainer_detected", existing_trainers=trainers)
                return
            with (training / "stdout.log").open("a", encoding="utf-8") as stdout, \
                    (training / "stderr.log").open("a", encoding="utf-8") as stderr:
                child = subprocess.Popen(arguments, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
                while child.poll() is None:
                    self.archive_best(training)
                    write_json(envelope / "progress.json", {"phase": "continuous_spell_training_running", "active": "training",
                                                            "launcher_pid": child.pid, "restart_attempt": attempt})
                    self.status("training_running", training_directory=str(training), launcher_pid=child.pid,
                                restart_attempt=attempt, entropy_coefficient=entropy,
                                reward_changed=self.continuation_profile != "radiant_spells_v1",
                                reward_profile=self.continuation_profile)
                    if self.stopped():
                        # Relinquish supervision, leaving process shutdown to the
                        # user/controller which requested this sentinel.
                        self.status("supervision_disabled", training_directory=str(training), launcher_pid=child.pid)
                        return
                    time.sleep(self.poll_seconds)
                code = child.wait()
            self.archive_best(training)
            if code == 0:
                self.status("training_budget_complete", training_directory=str(training))
                write_json(envelope / "progress.json", {"phase": "training_budget_complete", "active": None})
                return
            self.status("training_exited", exit_code=code, training_directory=str(training), restart_attempt=attempt)
            if attempt < self.max_restarts:
                time.sleep(30)
        self.status("training_failed_after_bounded_retries", training_directory=str(training))
        write_json(envelope / "progress.json", {"phase": "training_failed_after_bounded_retries", "active": None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--total-updates", type=int, default=10000)
    parser.add_argument("--max-restarts", type=int, default=2)
    parser.add_argument("--max-wait-hours", type=float, default=12)
    parser.add_argument("--episode-time-limit", type=float, choices=(120.0, 150.0), default=150.0,
                        help="fight cutoff for the next reward trial and continued training; observation scale stays 60s")
    parser.add_argument("--dodge-trial", action="store_true",
                        help="after entropy comparison, test +0.1 per completed clean boss move before continuation")
    parser.add_argument("--continue-checkpoint", type=Path,
                        help="explicitly continue a backed-up full checkpoint instead of selecting from a completed comparison")
    parser.add_argument("--reward-profile", choices=("radiant_spells_v1", "radiant_spells_dodge_v1", "radiant_potential_v1"),
                        help="user-requested reward for a new direct continuation; never changes an existing run in place")
    args = parser.parse_args()
    if args.poll_seconds < 5 or args.total_updates < 1 or not 0 <= args.max_restarts <= 3 or args.max_wait_hours <= 0:
        parser.error("Invalid polling interval, update budget or restart limit")
    supervisor = Supervisor(args.experiment_dir, args.poll_seconds, args.total_updates, args.max_restarts,
                            dodge_trial=args.dodge_trial, episode_time_limit=args.episode_time_limit,
                            continue_checkpoint=args.continue_checkpoint, reward_profile=args.reward_profile)
    lock = supervisor.directory / "auto_continue.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created_at_unix": time.time()}, stream)
    except FileExistsError:
        raise SystemExit("An auto-continuation lock already exists; inspect its owner before starting another watcher")
    try:
        supervisor.run(args.max_wait_hours)
    except Exception as error:
        supervisor.status("supervisor_failed", error=repr(error))
        (supervisor.directory / "auto_continue_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
