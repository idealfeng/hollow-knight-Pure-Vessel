"""Offline checks for unattended continuation; no game/process access."""
import hashlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import auto_continue_training as auto
from agent.action import SPELL_ACTION_NAMES
from agent.reward import get_reward_config


class AutoContinueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        root_patch = patch.object(auto, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.directory = self.root / "experiment"
        (self.directory / "backup").mkdir(parents=True)
        (self.root / "python/runs/traitor_4env_iframes_fixed").mkdir(parents=True)
        # Exercise the controller without requiring an installed game or the
        # user's private training files in a fresh checkout.
        self.saved = {
            "format_version": 9, "action_distribution": "exclusive_directions_spells_v1",
            "action_sizes": [1] * 7 + [4], "update": 425, "optimizer": {},
            "action_names": list(SPELL_ACTION_NAMES), "observation_size": 472,
            "frame_stack": 4, "control_hz": 10, "freeze_obs_norm": True,
            "training_config": {"reward_profile": "radiant_spells_v1", "boss_level": 2,
                                "reward_config": asdict(get_reward_config("radiant_spells_v1")),
                                "episode_time_limit": 120, "observation_time_limit": 60,
                                "game_speed": 1, "train_mode": "all"},
        }
        self.source = self.directory / "backup/source_425.pth"
        torch.save(self.saved, self.source)
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        auto.write_json(self.directory / "experiment.json", {
            "source_best": {"path": str(self.source), "sha256": self.digest},
            "endpoints": ["127.0.0.1:9001", "127.0.0.1:9002", "192.168.0.184:9003", "192.168.0.184:9004"],
        })
        self.baseline = {"label": "baseline_425", "valid": True, "episodes": 40,
                         "nohit_wins": 0, "win_rate": 0, "mean_min_boss_hp": 541,
                         "checkpoint_sha256": self.digest, "episode_time_limit": 150,
                         "observation_time_limit": 60}
        auto.write_json(self.directory / "backup/best_eval.json", {"selection_score": [0, .05, -434, -8.55, 0]})
        (self.directory / "baseline_425_fights.jsonl").write_text(
            "\n".join(json.dumps({"info": {"damage": 9, "won": False, "episode_damage": 9,
                                           "min_boss_health": 541, "boss_level": 2,
                                           "reset_mode": "scene", "damage_tracking_valid": True}})
                      for _ in range(40)), encoding="utf-8")

    def test_inconclusive_comparison_keeps_full_source_and_original_entropy(self):
        path, digest, entropy, _ = auto.choose_checkpoint(self.directory, {"low_entropy_qualified": False})
        self.assertEqual((path, digest, entropy), (self.source, self.digest, .005))
        self.source.write_bytes(self.source.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "source checkpoint changed"):
            auto.choose_checkpoint(self.directory)

    def qualified_selection(self):
        candidate = self.directory / "low_entropy_001/ppo_checkpoint.pth"
        candidate.parent.mkdir()
        torch.save(dict(self.saved, update=445), candidate)
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        low = dict(self.baseline, nohit_wins=4, mean_min_boss_hp=350, checkpoint_sha256=digest)
        confirmation = dict(low, nohit_wins=3)
        return {"low_entropy_qualified": True, "selected_checkpoint": str(candidate),
                "baseline": self.baseline, "control": self.baseline, "low_entropy": low,
                "confirmation": {"low": confirmation, "source": self.baseline}}

    def test_low_entropy_requires_both_thresholds_and_independent_confirmation(self):
        selection = self.qualified_selection()
        path, _, entropy, _ = auto.choose_checkpoint(self.directory, selection)
        self.assertEqual(path, self.directory / "low_entropy_001/ppo_checkpoint.pth")
        self.assertEqual(entropy, .001)
        selection["confirmation"]["low"]["nohit_wins"] = 1
        with self.assertRaisesRegex(ValueError, "qualification"):
            auto.choose_checkpoint(self.directory, selection)

    def test_preparation_preserves_source_seed_best_and_reward(self):
        supervisor = auto.Supervisor(self.directory)
        selection = {"baseline": self.baseline}
        with patch.object(auto, "ROOT", self.root):
            envelope, training, args = supervisor.prepare(self.source, self.digest, .005, "test", selection)
        self.assertEqual((training / "best_eval_checkpoint.pth").read_bytes(), self.source.read_bytes())
        self.assertEqual(args[args.index("--reward-profile") + 1], "radiant_spells_v1")
        self.assertEqual(args[args.index("--lr") + 1], "0.0001")
        self.assertEqual(args[args.index("--eval-every") + 1], "25")
        self.assertEqual(args[args.index("--total-updates") + 1], "10000")
        self.assertEqual(float(args[args.index("--episode-time-limit") + 1]), 150)
        self.assertEqual(args[args.index("--observation-time-limit") + 1], "60")
        self.assertEqual(json.loads((training / "best_eval.json").read_text())["selection_score"], [0, 0, -541, -9, 0])
        self.assertTrue((envelope / "backup/source_425.pth").exists())

    def checkpoint_payload(self, checkpoint):
        stream = io.BytesIO()
        torch.save(checkpoint, stream)
        return stream.getvalue()

    def combat_checkpoint(self, frames=16, profile="radiant_spells_dodge_v1"):
        config = dict(self.saved["training_config"], frame_stack=frames,
                      observation_schema="combat_v2", observation_size=128 * frames,
                      reward_profile=profile, reward_config=asdict(get_reward_config(profile)))
        return dict(self.saved, frame_stack=frames, observation_schema="combat_v2",
                    observation_size=128 * frames, training_config=config)

    def test_verified_checkpoint_accepts_exact_schema_width_for_four_and_sixteen_frames(self):
        for frames, profile in ((4, "radiant_spells_dodge_v1"),
                                (16, "radiant_spells_dodge_v1"),
                                (4, "radiant_potential_v1")):
            with self.subTest(frames=frames, profile=profile):
                checkpoint = self.combat_checkpoint(frames, profile)
                actual = auto.verified_checkpoint(self.checkpoint_payload(checkpoint), profile)
                self.assertEqual(actual["observation_size"], 128 * frames)
        self.assertEqual(auto.verified_checkpoint(self.source.read_bytes())["observation_size"], 472)

    def test_wrong_stack_width_or_conflicting_schema_is_rejected(self):
        for overrides, config_overrides in (({"observation_size": 512}, {}),
                                             ({"frame_stack": 8}, {}),
                                             ({}, {"frame_stack": 4}),
                                             ({}, {"observation_schema": "legacy_v1"}),
                                             ({}, {"observation_size": 512})):
            with self.subTest(overrides=overrides, config_overrides=config_overrides):
                checkpoint = self.combat_checkpoint()
                checkpoint.update(overrides)
                checkpoint["training_config"].update(config_overrides)
                with self.assertRaises(ValueError):
                    auto.verified_checkpoint(self.checkpoint_payload(checkpoint), "radiant_spells_dodge_v1")

    def test_preparation_inherits_sixteen_frames_without_changing_rewards_or_learning_rate(self):
        source = self.directory / "combat16.pth"
        source.write_bytes(self.checkpoint_payload(self.combat_checkpoint()))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        supervisor = auto.Supervisor(self.directory, continue_checkpoint=source)
        supervisor.continuation_profile = "radiant_spells_dodge_v1"
        _, training, arguments = supervisor.prepare(source, digest, .005, "16 frame continuation", None)
        self.assertEqual(arguments[arguments.index("--frame-stack") + 1], "16")
        self.assertEqual(arguments[arguments.index("--reward-profile") + 1], "radiant_spells_dodge_v1")
        self.assertEqual(arguments[arguments.index("--lr") + 1], "0.0001")
        self.assertEqual((training.parent / "backup/source_425.pth").read_bytes(), source.read_bytes())

    def test_legacy_sixteen_frames_accepts_1888_columns_without_combat_fields(self):
        config = dict(self.saved["training_config"], frame_stack=16, observation_schema="legacy_v1",
                      observation_size=1888, reward_profile="radiant_spells_dodge_v1",
                      reward_config=asdict(get_reward_config("radiant_spells_dodge_v1")))
        checkpoint = dict(self.saved, frame_stack=16, observation_schema="legacy_v1",
                          observation_size=1888, training_config=config)
        source = self.directory / "legacy16.pth"
        source.write_bytes(self.checkpoint_payload(checkpoint))
        verified = auto.verified_checkpoint(source.read_bytes(), "radiant_spells_dodge_v1")
        self.assertEqual(verified["observation_schema"], "legacy_v1")
        self.assertEqual(verified["observation_size"], 118 * 16)
        supervisor = auto.Supervisor(self.directory, continue_checkpoint=source)
        supervisor.continuation_profile = "radiant_spells_dodge_v1"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        _, training, arguments = supervisor.prepare(source, digest, .005, "old features sixteen frames", None)
        self.assertEqual(arguments[arguments.index("--frame-stack") + 1], "16")
        self.assertEqual(arguments[arguments.index("--reward-profile") + 1], "radiant_spells_dodge_v1")
        if "--observation-schema" in arguments:
            self.assertEqual(arguments[arguments.index("--observation-schema") + 1], "legacy_v1")
        self.assertEqual((training.parent / "backup/source_425.pth").read_bytes(), source.read_bytes())

    def test_120_second_source_score_is_preserved_without_seeding_150_second_comparison(self):
        supervisor = auto.Supervisor(self.directory, episode_time_limit=150)
        _, training, args = supervisor.prepare(self.source, self.digest, .005, "longer horizon",
                                               {"baseline": dict(self.baseline, episode_time_limit=120)})
        self.assertFalse((training / "best_eval.json").exists())
        self.assertTrue((training / "best_eval_checkpoint.pth").exists())
        preserved = json.loads((training.parent / "backup/source_assessment.json").read_text())
        self.assertEqual(preserved["episode_time_limit"], 120)
        self.assertEqual(float(args[args.index("--episode-time-limit") + 1]), 150)

    def test_stop_sentinel_never_starts_or_queries_a_trainer(self):
        (self.directory / "STOP_AUTO_CONTINUE").touch()
        with patch.object(auto.subprocess, "Popen") as launch, patch.object(auto, "live_trainers") as query:
            auto.Supervisor(self.directory).run()
        launch.assert_not_called()
        query.assert_not_called()

    def test_explicit_reward_continuation_preserves_full_source_without_repeating_selection(self):
        config = dict(self.saved["training_config"], ent_coef=.005, lr=.0001, episode_time_limit=150)
        torch.save(dict(self.saved, training_config=config), self.source)
        original = self.source.read_bytes()
        child = MagicMock(pid=1234)
        child.poll.return_value = 0
        child.wait.return_value = 0
        with patch.object(auto, "live_trainers", return_value=[]), patch.object(auto.time, "sleep"), \
                patch.object(auto.Supervisor, "stage"), patch.object(auto, "choose_checkpoint") as choose, \
                patch.object(auto.subprocess, "Popen", return_value=child) as launch:
            auto.Supervisor(self.directory, continue_checkpoint=self.source,
                            reward_profile="radiant_spells_dodge_v1").run()
        choose.assert_not_called()
        args = launch.call_args.args[0]
        self.assertEqual(args[args.index("--reward-profile") + 1], "radiant_spells_dodge_v1")
        self.assertEqual(Path(args[args.index("--init-checkpoint") + 1]).read_bytes(), original)
        decision = json.loads((self.directory / "continuation_decision.json").read_text())
        self.assertTrue(decision["user_requested_reward_override"])
        training = Path(decision["training_directory"])
        self.assertFalse((training / "best_eval.json").exists())
        self.assertFalse((training / "best_eval_checkpoint.pth").exists())

    def test_reward_change_cannot_be_silently_applied_during_preparation(self):
        supervisor = auto.Supervisor(self.directory)
        supervisor.continuation_profile = "radiant_spells_dodge_v1"
        with self.assertRaisesRegex(ValueError, "explicit direct-continuation authorization"):
            supervisor.prepare(self.source, self.digest, .005, "unexpected", None)

    def test_direct_reward_override_cannot_be_combined_with_automatic_reward_trial(self):
        with self.assertRaisesRegex(ValueError, "does not rerun"):
            auto.Supervisor(self.directory, continue_checkpoint=self.source, dodge_trial=True)
        with self.assertRaisesRegex(ValueError, "requires --continue-checkpoint"):
            auto.Supervisor(self.directory, reward_profile="radiant_spells_dodge_v1")

    def test_unconfirmed_dodge_trial_cannot_change_continuation_reward(self):
        from dodge_reward_trial import DodgeRewardTrial
        auto.write_json(self.directory / "progress.json", {"phase": "complete_comparison_stopped"})
        auto.write_json(self.directory / "selection.json", {"low_entropy_qualified": False, "baseline": self.baseline})
        result = {"selected_checkpoint": str(self.source), "selected_sha256": self.digest,
                  "reward_profile": "radiant_spells_v1", "entropy_coefficient": .005,
                  "reason": "No confirmed improvement", "assessment": None, "fights_path": None}
        child = MagicMock(pid=1234)
        child.poll.return_value = 0
        child.wait.return_value = 0
        with patch.object(auto, "ROOT", self.root), patch.object(auto, "live_trainers", return_value=[]), \
                patch.object(auto.Supervisor, "stage"), patch.object(auto.time, "sleep"), \
                patch.object(DodgeRewardTrial, "run", return_value=result) as trial, \
                patch.object(auto.subprocess, "Popen", return_value=child) as launch:
            auto.Supervisor(self.directory, dodge_trial=True).run()
        trial.assert_called_once()
        args = launch.call_args.args[0]
        self.assertEqual(args[args.index("--reward-profile") + 1], "radiant_spells_v1")

    def test_stop_inside_reward_trial_never_starts_a_continuation(self):
        from dodge_reward_trial import DodgeRewardTrial, TrialStopped
        auto.write_json(self.directory / "progress.json", {"phase": "complete_comparison_stopped"})
        auto.write_json(self.directory / "selection.json", {"low_entropy_qualified": False, "baseline": self.baseline})
        with patch.object(auto, "live_trainers", return_value=[]), patch.object(auto.time, "sleep"), \
                patch.object(DodgeRewardTrial, "run", side_effect=TrialStopped("test")), \
                patch.object(auto.subprocess, "Popen") as launch:
            auto.Supervisor(self.directory, dodge_trial=True).run()
        launch.assert_not_called()
        self.assertEqual(json.loads((self.directory / "auto_continue_status.json").read_text())["phase"], "disabled")

    def test_preparation_of_confirmed_dodge_candidate_uses_its_own_assessment(self):
        candidate = self.directory / "dodge_candidate.pth"
        config = dict(self.saved["training_config"], reward_profile="radiant_spells_dodge_v1",
                      reward_config=asdict(get_reward_config("radiant_spells_dodge_v1")))
        torch.save(dict(self.saved, update=445, training_config=config), candidate)
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        fights = self.directory / "candidate_confirm_fights.jsonl"
        fights.write_bytes((self.directory / "baseline_425_fights.jsonl").read_bytes())
        supervisor = auto.Supervisor(self.directory, dodge_trial=True)
        supervisor.continuation_profile = "radiant_spells_dodge_v1"
        supervisor.continuation_assessment = dict(self.baseline, label="candidate_confirm", checkpoint_sha256=digest)
        supervisor.continuation_fights = fights
        with patch.object(auto, "ROOT", self.root):
            _, training, args = supervisor.prepare(candidate, digest, .005, "confirmed", {"baseline": self.baseline})
        self.assertEqual(args[args.index("--reward-profile") + 1], "radiant_spells_dodge_v1")
        self.assertEqual(json.loads((training / "best_eval.json").read_text())["checkpoint_sha256"], digest)
        supervisor.archive_best(training)

    def test_reward_candidate_cannot_bypass_independent_qualification(self):
        result = {"selected_checkpoint": str(self.directory / "candidate/ppo_checkpoint.pth"),
                  "selected_sha256": "candidate", "reward_profile": "radiant_spells_dodge_v1",
                  "entropy_coefficient": .005, "assessment": self.baseline}
        with self.assertRaisesRegex(ValueError, "qualification"):
            auto.validate_dodge_selection(result, self.directory, self.digest, .005)
        result["dodge_reward_qualified"] = True
        with self.assertRaisesRegex(ValueError, "thresholds"):
            auto.validate_dodge_selection(result, self.directory, self.digest, .005)

    def test_reward_confirmation_must_match_candidate_bytes_and_expected_records(self):
        source = dict(self.baseline, nohit_wins=0)
        candidate = dict(self.baseline, nohit_wins=4, checkpoint_sha256="candidate")
        result = {"selected_checkpoint": str(self.directory / "candidate/ppo_checkpoint.pth"),
                  "selected_sha256": "candidate", "reward_profile": "radiant_spells_dodge_v1",
                  "entropy_coefficient": .005, "dodge_reward_qualified": True,
                  "control": source, "candidate": candidate,
                  "confirmation": {"candidate": candidate, "control": source},
                  "assessment": candidate, "fights_path": str(self.directory / "candidate_confirm_fights.jsonl")}
        auto.validate_dodge_selection(result, self.directory, self.digest, .005)
        result["selected_sha256"] = "different weights"
        with self.assertRaisesRegex(ValueError, "confirmed assessment"):
            auto.validate_dodge_selection(result, self.directory, self.digest, .005)

    def test_completed_experiment_waits_for_existing_trainer_then_starts_only_once(self):
        auto.write_json(self.directory / "progress.json", {"phase": "complete_comparison_stopped"})
        auto.write_json(self.directory / "selection.json", {"low_entropy_qualified": False, "baseline": self.baseline})
        child = MagicMock()
        child.pid = 1234
        child.poll.side_effect = [None, 0]
        child.wait.return_value = 0
        with patch.object(auto, "ROOT", self.root), \
                patch.object(auto, "live_trainers", side_effect=[[{"ProcessId": 99}], [], [], [], [], [], []]), \
                patch.object(auto.Supervisor, "stage") as stage, \
                patch.object(auto.time, "sleep"), \
                patch.object(auto.subprocess, "Popen", return_value=child) as launch:
            auto.Supervisor(self.directory).run()
        launch.assert_called_once()
        stage.assert_called_once()
        status = json.loads((self.directory / "auto_continue_status.json").read_text())
        self.assertEqual(status["phase"], "training_budget_complete")
        self.assertEqual(len(list(self.directory.glob("continuation_*/training/best_snapshots/*.pth"))), 1)

    def test_failed_game_preparation_is_bounded_and_never_launches_trainer(self):
        auto.write_json(self.directory / "progress.json", {"phase": "failed", "error": "test"})
        with patch.object(auto, "ROOT", self.root), \
                patch.object(auto, "live_trainers", return_value=[]), \
                patch.object(auto.Supervisor, "stage", side_effect=RuntimeError("game unavailable")) as stage, \
                patch.object(auto.time, "sleep"), \
                patch.object(auto.subprocess, "Popen") as launch:
            auto.Supervisor(self.directory, max_restarts=2).run()
        self.assertEqual(stage.call_count, 3)
        launch.assert_not_called()
        self.assertEqual(json.loads((self.directory / "auto_continue_status.json").read_text())["phase"],
                         "training_failed_after_bounded_retries")

    def test_nonzero_trainer_exit_retries_with_same_run_and_stops_at_limit(self):
        auto.write_json(self.directory / "progress.json", {"phase": "complete_comparison_stopped"})
        auto.write_json(self.directory / "selection.json", {"low_entropy_qualified": False, "baseline": self.baseline})
        child = MagicMock()
        child.pid = 1234
        child.poll.return_value = 1
        child.wait.return_value = 1
        with patch.object(auto, "ROOT", self.root), \
                patch.object(auto, "live_trainers", return_value=[]), \
                patch.object(auto.Supervisor, "stage"), \
                patch.object(auto.time, "sleep"), \
                patch.object(auto.subprocess, "Popen", return_value=child) as launch:
            auto.Supervisor(self.directory, max_restarts=1).run()
        self.assertEqual(launch.call_count, 2)
        self.assertEqual(launch.call_args_list[0].args, launch.call_args_list[1].args)
        self.assertEqual(json.loads((self.directory / "auto_continue_status.json").read_text())["phase"],
                         "training_failed_after_bounded_retries")


if __name__ == "__main__":
    unittest.main()
