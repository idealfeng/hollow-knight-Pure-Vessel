"""Finite same-source comparison for the optional action-phase dodge reward.

The caller owns supervision. This module never starts long training and never
promotes a model from reward totals. Game access is limited to exclusive trial
stages; process checks inspect metadata without opening game sockets.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from agent.reward import get_reward_config
from evaluation import evaluate_pool
from main_multi import ExternalEnvPool, prepare_fights
from obs_norm import RunningNorm
from policy_runtime import load_checkpoint, stage_fresh_fight
from policy_selection import is_flawless
from ppo_model import ActorCritic
from observation_schema import LEGACY_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
CONTROL_PROFILE = "radiant_spells_v1"
DODGE_PROFILE = "radiant_spells_dodge_v1"
UPDATES = 20
STEPS_PER_ENV = 512
EPISODES_PER_ENV = 10


class TrialStopped(RuntimeError):
    """Control was relinquished; a child trainer may still be running."""


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def qualifies(candidate, control, extra_wins):
    """A practical screening rule, not a statistical significance claim."""
    return (candidate.get("valid") and control.get("valid")
            and candidate.get("episodes") == 40 and control.get("episodes") == 40
            and candidate["nohit_wins"] >= control["nohit_wins"] + extra_wins
            and candidate["mean_min_boss_hp"] <= control["mean_min_boss_hp"] + 65)


def normalization_fingerprint(value):
    return json.dumps(value, default=lambda item: item.tolist(), sort_keys=True)


class DodgeRewardTrial:
    def __init__(self, directory, source, digest, entropy, endpoints, stopped=lambda: False,
                 episode_time_limit=150.0):
        if len(endpoints) != 4 or len(set(endpoints)) != 4:
            raise ValueError("The paired trial requires four distinct endpoints")
        if entropy not in (0.005, 0.001):
            raise ValueError("Unexpected entropy coefficient")
        try:
            duration = float(episode_time_limit)
        except (TypeError, ValueError):
            raise ValueError("Episode time limit must be finite and positive") from None
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Episode time limit must be finite and positive")
        self.directory = Path(directory).resolve()
        self.original_source = Path(source).resolve()
        self.source = self.original_source
        self.digest = digest
        self.entropy = entropy
        self.episode_time_limit = duration
        self.endpoint_names = list(endpoints)
        self.endpoints = [(item.rsplit(":", 1)[0], int(item.rsplit(":", 1)[1]))
                          for item in endpoints]
        self.stopped = stopped
        self.child = None
        self.progress = {"phase": "pending", "active": None, "evaluations": []}

    def status(self, phase, **details):
        self.progress.update(phase=phase, updated_at_unix=time.time(), **details)
        if self.directory.exists():
            write_json(self.directory / "progress.json", self.progress)

    def check_stop(self):
        if self.stopped():
            self.status("stopped", child_pid=self.child.pid if self.child else None)
            raise TrialStopped("Dodge reward trial supervision was disabled")

    def check_idle(self):
        # Import lazily so the parent supervisor can call this module without a
        # module import cycle. This function only uses Win32 process metadata.
        from auto_continue_training import live_trainers
        self.check_stop()
        trainers = live_trainers()
        if trainers:
            raise RuntimeError(f"Another trainer owns the games: {trainers}")

    def stage(self):
        self.check_idle()
        time.sleep(3)
        self.check_idle()
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda endpoint: stage_fresh_fight(endpoint, 2), self.endpoints))
        prepare_fights(self.endpoints)
        self.check_idle()

    def prepare(self):
        self.check_stop()
        payload = self.original_source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.digest:
            raise ValueError("Source checkpoint changed before the dodge trial")
        from auto_continue_training import verified_checkpoint
        saved = verified_checkpoint(payload)
        self.directory.mkdir(parents=True, exist_ok=False)
        backup = self.directory / "backup"
        backup.mkdir()
        self.source = backup / f"source_{saved['update']}.pth"
        self.source.write_bytes(payload)
        write_json(self.directory / "experiment.json", {
            "source_checkpoint": str(self.original_source), "backup_checkpoint": str(self.source),
            "source_sha256": self.digest, "source_update": saved["update"],
            "entropy_coefficient": self.entropy, "endpoints": self.endpoint_names,
            "updates_per_branch": UPDATES, "transitions_per_branch": UPDATES * STEPS_PER_ENV * 4,
            "episodes_per_assessment": 40, "control_profile": CONTROL_PROFILE,
            "candidate_profile": DODGE_PROFILE, "evaluation_profile": CONTROL_PROFILE,
            "episode_time_limit": self.episode_time_limit, "observation_time_limit": 60.0,
            "candidate_reward": asdict(get_reward_config(DODGE_PROFILE)),
            "screen_extra_wins": 4, "confirmation_extra_wins": 2,
            "maximum_hp_regression": 65,
            "limitations": "Sequential single-seed trial; game RNG is unpaired; thresholds are screening rules.",
        })
        self.status("prepared")

    def train(self, label, profile):
        self.check_stop()
        saved, digest = load_checkpoint(self.source)
        if digest != self.digest:
            raise ValueError("Backed-up source checkpoint changed")
        run = self.directory / label
        run.mkdir(exist_ok=False)
        target = saved["update"] + UPDATES
        arguments = [sys.executable, "-u", str(ROOT / "python/main_multi.py"), "--endpoints"]
        arguments += self.endpoint_names
        arguments += ["--run-dir", str(run), "--init-checkpoint", str(self.source),
                      "--action-distribution", "exclusive_directions_spells_v1",
                      "--reward-profile", profile, "--boss-level", "2",
                      "--eval-objective", "nohit", "--episode-time-limit", str(self.episode_time_limit),
                      "--observation-time-limit", "60", "--control-hz", "10",
                      "--frame-stack", "4", "--game-speed", "1", "--seed", "42",
                      "--torch-threads", "2", "--lr", "0.0001", "--ent-coef", str(self.entropy),
                      "--freeze-obs-norm", "--steps-per-env", str(STEPS_PER_ENV),
                      "--train-mode", "all", "--total-updates", str(target + 1),
                      "--eval-every", "0", "--no-monitor"]
        write_json(run / "launch_arguments.json", arguments)
        self.status("preparing_training", active=label, source_update=saved["update"],
                    target_update=target, updates_budget=UPDATES, reward_profile=profile)
        self.stage()
        self.check_idle()
        environment = dict(os.environ)
        environment.pop("HOLLOWRL_TRACE_DIR", None)
        environment["HOLLOWRL_FAST_RESET"] = "auto"
        with (run / "stdout.log").open("w", encoding="utf-8") as stdout, \
                (run / "stderr.log").open("w", encoding="utf-8") as stderr:
            self.child = subprocess.Popen(arguments, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
            write_json(run / "process.json", {"pid": self.child.pid})
            self.status("training_branch", active=label, child_pid=self.child.pid)
            while self.child.poll() is None:
                self.check_stop()
                time.sleep(2)
            code = self.child.wait()
            self.child = None
        self.check_stop()
        if code:
            raise RuntimeError(f"{label} training failed with code {code}; see stderr.log")
        checkpoint = run / "ppo_checkpoint.pth"
        final, _ = load_checkpoint(checkpoint)
        config = final.get("training_config", {})
        if (final["update"] != target
                or final["global_step"] - saved["global_step"] != UPDATES * STEPS_PER_ENV * 4
                or config.get("ent_coef") != self.entropy or config.get("lr") != 0.0001
                or config.get("reward_profile") != profile
                or config.get("reward_config") != asdict(get_reward_config(profile))
                or config.get("episode_time_limit") != self.episode_time_limit
                or config.get("observation_time_limit") != 60
                or not final.get("freeze_obs_norm")
                or normalization_fingerprint(final.get("obs_norm")) != normalization_fingerprint(saved["obs_norm"])):
            raise RuntimeError(f"Wrong branch budget/settings: {label}")
        return checkpoint

    def evaluate(self, checkpoint, label):
        self.check_stop()
        saved, digest = load_checkpoint(checkpoint)
        model = ActorCritic(saved["observation_size"], saved["action_sizes"],
                            hidden=saved["model_hidden"], bernoulli=True, exclusive_directions=True)
        model.load_state_dict(saved["model"])
        model.eval()
        normalizer = RunningNorm(saved["observation_size"])
        normalizer.load_state_dict(saved["obs_norm"])
        norm_before = normalization_fingerprint(normalizer.state_dict())
        self.status("preparing_evaluation", active=label, completed_fights=0, target_fights=40)
        self.stage()
        fights_path = self.directory / f"{label}_fights.jsonl"
        trace = self.directory / "evaluation_traces" / label
        if fights_path.exists() or trace.exists():
            raise RuntimeError(f"Refusing to reuse evaluation records: {label}")
        old_trace = os.environ.get("HOLLOWRL_TRACE_DIR")
        os.environ["HOLLOWRL_TRACE_DIR"] = str(trace)
        results = []
        pool = None
        started = time.monotonic()
        try:
            self.check_idle()
            options = ({"observation_schema": saved["observation_schema"]}
                       if saved.get("observation_schema", LEGACY_SCHEMA) != LEGACY_SCHEMA else {})
            pool = ExternalEnvPool(self.endpoints, 10, saved["frame_stack"], self.episode_time_limit,
                                   1.0, CONTROL_PROFILE, 2, 60.0, "exclusive_directions_spells_v1", **options)

            def record(result):
                self.check_stop()
                results.append(result)
                with fights_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result) + "\n")
                self.status("evaluating", active=label, completed_fights=len(results),
                            wins=sum(is_flawless(item["info"]) for item in results),
                            mean_min_boss_hp=statistics.mean(item["info"]["min_boss_health"] for item in results))

            with torch.inference_mode():
                evaluate_pool(pool, model, normalizer.normalize, EPISODES_PER_ENV, record)
        finally:
            if pool is not None:
                pool.close()
            if old_trace is None:
                os.environ.pop("HOLLOWRL_TRACE_DIR", None)
            else:
                os.environ["HOLLOWRL_TRACE_DIR"] = old_trace
        self.check_stop()
        if (len(results) != 40 or any(sum(r["env"] == index for r in results) != 10 for index in range(4))
                or any(r["info"].get("boss_level") != 2 or r["info"].get("reset_mode") != "scene"
                       or r["info"].get("truncated_reason") in ("disconnect", "fight_lost")
                       or not r["info"].get("damage_tracking_valid", False) for r in results)):
            raise RuntimeError(f"Incomplete/invalid assessment: {label}")
        if (digest != load_checkpoint(checkpoint)[1]
                or norm_before != normalization_fingerprint(normalizer.state_dict())):
            raise RuntimeError("Frozen checkpoint/normalizer changed")
        summary = {
            "label": label, "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
            "update": saved["update"], "episodes": 40, "valid": True,
            "nohit_wins": sum(is_flawless(item["info"]) for item in results),
            "win_rate": statistics.mean(bool(item["info"]["won"]) for item in results),
            "mean_min_boss_hp": statistics.mean(item["info"]["min_boss_health"] for item in results),
            "mean_damage": statistics.mean(item["info"]["damage"] for item in results),
            "mean_hits": statistics.mean(item["info"]["hits"] for item in results),
            "timeouts": sum(item["info"].get("truncated_reason") == "timeout" for item in results),
            "evaluation_reward_profile": CONTROL_PROFILE, "elapsed_seconds": time.monotonic() - started,
            "episode_time_limit": self.episode_time_limit, "observation_time_limit": 60.0,
        }
        write_json(self.directory / f"{label}_summary.json", summary)
        self.progress["evaluations"].append(summary)
        self.status("evaluation_complete", active=label)
        return summary

    def run(self):
        self.prepare()
        torch.set_num_threads(2)
        torch.manual_seed(42)
        try:
            control_path = self.train("control", CONTROL_PROFILE)
            control = self.evaluate(control_path, "control_screen")
            candidate_path = self.train("candidate", DODGE_PROFILE)
            candidate = self.evaluate(candidate_path, "candidate_screen")
            promoted = bool(qualifies(candidate, control, 4))
            confirmation = None
            assessment = None
            if promoted:
                candidate_confirm = self.evaluate(candidate_path, "candidate_confirm")
                control_confirm = self.evaluate(control_path, "control_confirm")
                confirmation = {"candidate": candidate_confirm, "control": control_confirm}
                promoted = bool(qualifies(candidate_confirm, control_confirm, 2))
                if promoted:
                    assessment = candidate_confirm
            selected = candidate_path if promoted else self.source
            result = {
                "selected_checkpoint": str(selected),
                "selected_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
                "reward_profile": DODGE_PROFILE if promoted else CONTROL_PROFILE,
                "entropy_coefficient": self.entropy, "dodge_reward_qualified": promoted,
                "reason": ("Action-phase dodge reward passed screening and independent confirmation"
                           if promoted else "Retain original source; dodge reward has no confirmed improvement"),
                "assessment": assessment,
                "fights_path": str(self.directory / "candidate_confirm_fights.jsonl") if promoted else None,
                "control": control, "candidate": candidate, "confirmation": confirmation,
                "long_training_started": False,
            }
            self.check_stop()
            write_json(self.directory / "selection.json", result)
            self.status("complete_reward_comparison_stopped", active=None,
                        selected_checkpoint=str(selected), dodge_reward_qualified=promoted)
            return result
        except TrialStopped:
            raise
        except Exception as error:
            self.status("failed", error=repr(error), child_pid=self.child.pid if self.child else None)
            raise
