"""Offline safeguards for the paired dodge-reward experiment."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

import dodge_reward_trial as trial_module
from agent.action import SPELL_ACTION_NAMES
from agent.reward import get_reward_config


class DodgeRewardTrialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.pth"
        self.endpoints = ["127.0.0.1:9001", "127.0.0.1:9002", "192.168.0.184:9003", "192.168.0.184:9004"]
        self.base = get_reward_config("radiant_spells_v1")
        self.dodge = get_reward_config("radiant_spells_dodge_v1")
        self.saved = {
            "format_version": 9, "action_distribution": "exclusive_directions_spells_v1",
            "action_sizes": [1] * 7 + [4], "update": 425, "global_step": 872448,
            "optimizer": {"sentinel": "preserved optimizer"}, "obs_norm": {"sentinel": "preserved normalizer"},
            "action_names": list(SPELL_ACTION_NAMES), "observation_size": 472,
            "frame_stack": 4, "control_hz": 10, "freeze_obs_norm": True,
            "training_config": {"reward_profile": "radiant_spells_v1", "boss_level": 2,
                                "reward_config": asdict(self.base), "lr": .0001,
                                "episode_time_limit": 120, "observation_time_limit": 60,
                                "game_speed": 1, "train_mode": "all", "ent_coef": .005},
        }
        torch.save(self.saved, self.source)
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.trial = trial_module.DodgeRewardTrial(
            self.root / "trial", self.source, self.digest, .005, self.endpoints)
        reward_patch = patch.object(trial_module, "get_reward_config",
                                    side_effect=lambda name: self.dodge if name == trial_module.DODGE_PROFILE else self.base)
        reward_patch.start()
        self.addCleanup(reward_patch.stop)

    def summary(self, label, wins, hp=400):
        return {"label": label, "valid": True, "episodes": 40, "nohit_wins": wins,
                "mean_min_boss_hp": hp, "checkpoint_sha256": "assessed"}

    def test_source_hash_is_checked_before_any_game_or_process_access(self):
        self.source.write_bytes(self.source.read_bytes() + b"changed")
        with patch.object(self.trial, "stage") as stage, \
                patch.object(trial_module.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "Source checkpoint changed"):
                self.trial.run()
        stage.assert_not_called()
        launch.assert_not_called()
        self.assertFalse(self.trial.directory.exists())

    def test_full_source_backup_retains_optimizer_and_normalizer_bytes(self):
        self.trial.prepare()
        self.assertEqual(self.trial.source.read_bytes(), self.source.read_bytes())
        saved, digest = trial_module.load_checkpoint(self.trial.source)
        self.assertEqual(digest, self.digest)
        self.assertEqual(saved["optimizer"], self.saved["optimizer"])
        self.assertEqual(saved["obs_norm"], self.saved["obs_norm"])

    def test_twenty_updates_mean_exactly_40960_samples_and_same_full_initializer(self):
        self.trial.prepare()
        child = MagicMock(pid=1234)
        child.poll.return_value = 0
        child.wait.return_value = 0

        def launch(*args, **kwargs):
            run = self.trial.directory / "candidate"
            config = dict(self.saved["training_config"], reward_profile=trial_module.DODGE_PROFILE,
                          reward_config=asdict(self.dodge))
            torch.save(dict(self.saved, update=445, global_step=913408,
                            training_config=config), run / "ppo_checkpoint.pth")
            return child

        with patch.object(self.trial, "stage"), patch.object(self.trial, "check_idle"), \
                patch.object(trial_module.subprocess, "Popen", side_effect=launch) as launch_mock:
            checkpoint = self.trial.train("candidate", trial_module.DODGE_PROFILE)
        args = launch_mock.call_args.args[0]
        self.assertEqual(args[args.index("--total-updates") + 1], "446")
        self.assertEqual(args[args.index("--init-checkpoint") + 1], str(self.trial.source))
        self.assertEqual(args[args.index("--eval-every") + 1], "0")
        self.assertEqual(args[args.index("--reward-profile") + 1], trial_module.DODGE_PROFILE)
        self.assertEqual(args[args.index("--ent-coef") + 1], "0.005")
        self.assertEqual(checkpoint.name, "ppo_checkpoint.pth")

    def test_incomplete_sampling_budget_is_rejected_even_at_expected_update(self):
        self.trial.prepare()
        child = MagicMock(pid=1234)
        child.poll.return_value = 0
        child.wait.return_value = 0

        def launch(*args, **kwargs):
            config = dict(self.saved["training_config"], reward_profile=trial_module.DODGE_PROFILE,
                          reward_config=asdict(self.dodge))
            torch.save(dict(self.saved, update=445, global_step=913407,
                            training_config=config), self.trial.directory / "candidate/ppo_checkpoint.pth")
            return child

        with patch.object(self.trial, "stage"), patch.object(self.trial, "check_idle"), \
                patch.object(trial_module.subprocess, "Popen", side_effect=launch):
            with self.assertRaisesRegex(RuntimeError, "budget/settings"):
                self.trial.train("candidate", trial_module.DODGE_PROFILE)

    def test_exit_error_never_proceeds_to_checkpoint_or_evaluation(self):
        self.trial.prepare()
        child = MagicMock(pid=1234)
        child.poll.return_value = 23
        child.wait.return_value = 23
        with patch.object(self.trial, "stage"), patch.object(self.trial, "check_idle"), \
                patch.object(trial_module.subprocess, "Popen", return_value=child):
            with self.assertRaisesRegex(RuntimeError, "code 23"):
                self.trial.train("candidate", trial_module.DODGE_PROFILE)
        self.assertIsNone(self.trial.child)

    def test_stop_before_prepare_never_touches_games(self):
        self.trial.stopped = lambda: True
        with patch.object(self.trial, "stage") as stage, \
                patch.object(trial_module.subprocess, "Popen") as launch:
            with self.assertRaises(trial_module.TrialStopped):
                self.trial.run()
        launch.assert_not_called()
        stage.assert_not_called()

    def test_stop_during_training_relinquishes_child_and_records_owner(self):
        self.trial.prepare()
        self.trial.stopped = lambda: self.trial.child is not None
        child = MagicMock(pid=1234)
        child.poll.return_value = None
        with patch.object(self.trial, "stage"), patch.object(self.trial, "check_idle"), \
                patch.object(trial_module.subprocess, "Popen", return_value=child):
            with self.assertRaises(trial_module.TrialStopped):
                self.trial.train("candidate", trial_module.DODGE_PROFILE)
        child.terminate.assert_not_called()
        child.kill.assert_not_called()
        status = json.loads((self.trial.directory / "progress.json").read_text())
        self.assertEqual((status["phase"], status["child_pid"]), ("stopped", 1234))

    def test_better_reward_or_hp_alone_cannot_promote_and_no_confirmation_runs(self):
        paths = [self.root / "control.pth", self.root / "candidate.pth"]
        for path in paths:
            path.write_bytes(b"branch")
        with patch.object(self.trial, "train", side_effect=paths), \
                patch.object(self.trial, "evaluate", side_effect=[self.summary("control", 1, 450),
                                                                 self.summary("candidate", 2, 200)]) as evaluate:
            result = self.trial.run()
        self.assertFalse(result["dodge_reward_qualified"])
        self.assertEqual(result["selected_sha256"], self.digest)
        self.assertIsNone(result["assessment"])
        self.assertEqual(evaluate.call_count, 2)

    def test_screen_success_followed_by_failed_independent_confirmation_keeps_source(self):
        control, candidate = self.root / "control.pth", self.root / "candidate.pth"
        control.write_bytes(b"control")
        candidate.write_bytes(b"candidate")
        summaries = [self.summary("control_screen", 0), self.summary("candidate_screen", 4),
                     self.summary("candidate_confirm", 1), self.summary("control_confirm", 0)]
        with patch.object(self.trial, "train", side_effect=[control, candidate]), \
                patch.object(self.trial, "evaluate", side_effect=summaries):
            result = self.trial.run()
        self.assertFalse(result["dodge_reward_qualified"])
        self.assertEqual(result["selected_sha256"], self.digest)

    def test_only_confirmed_candidate_carries_matching_fresh_assessment(self):
        control, candidate = self.root / "control.pth", self.root / "candidate.pth"
        control.write_bytes(b"control")
        candidate.write_bytes(b"candidate")
        confirmation = self.summary("candidate_confirm", 3)
        summaries = [self.summary("control_screen", 0), self.summary("candidate_screen", 4),
                     confirmation, self.summary("control_confirm", 1)]
        with patch.object(self.trial, "train", side_effect=[control, candidate]), \
                patch.object(self.trial, "evaluate", side_effect=summaries):
            result = self.trial.run()
        self.assertTrue(result["dodge_reward_qualified"])
        self.assertEqual(result["selected_checkpoint"], str(candidate))
        self.assertEqual(result["selected_sha256"], hashlib.sha256(b"candidate").hexdigest())
        self.assertEqual(result["assessment"], confirmation)
        self.assertEqual(result["reward_profile"], trial_module.DODGE_PROFILE)
        self.assertEqual(Path(result["fights_path"]).name, "candidate_confirm_fights.jsonl")

    def test_missing_fights_or_severe_hp_regression_cannot_qualify(self):
        control = self.summary("control", 0)
        candidate = self.summary("candidate", 5, hp=466)
        self.assertFalse(trial_module.qualifies(candidate, control, 4))
        candidate.update(mean_min_boss_hp=400, episodes=39)
        self.assertFalse(trial_module.qualifies(candidate, control, 4))


if __name__ == "__main__":
    unittest.main()
