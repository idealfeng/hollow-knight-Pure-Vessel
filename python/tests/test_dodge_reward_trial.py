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
        root_patch = patch.object(trial_module, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
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
        experiment = json.loads((self.trial.directory / "experiment.json").read_text())
        self.assertEqual((experiment["episode_time_limit"], experiment["observation_time_limit"]), (150, 60))
        self.assertEqual(saved["training_config"]["episode_time_limit"], 120)

    def test_both_profiles_keep_same_40960_sample_budget_and_150_second_limit(self):
        self.trial.prepare()
        child = MagicMock(pid=1234)
        child.poll.return_value = 0
        child.wait.return_value = 0

        def launch(*args, **kwargs):
            arguments = args[0]
            run = Path(arguments[arguments.index("--run-dir") + 1])
            profile = arguments[arguments.index("--reward-profile") + 1]
            config = dict(self.saved["training_config"], reward_profile=profile,
                          reward_config=asdict(get_reward_config(profile)), episode_time_limit=150)
            torch.save(dict(self.saved, update=445, global_step=913408,
                            training_config=config), run / "ppo_checkpoint.pth")
            return child

        for label, profile in (("control", trial_module.CONTROL_PROFILE),
                               ("candidate", trial_module.DODGE_PROFILE)):
            with self.subTest(profile=profile), patch.object(self.trial, "stage"), \
                    patch.object(self.trial, "check_idle"), \
                    patch.object(trial_module.subprocess, "Popen", side_effect=launch) as launch_mock:
                checkpoint = self.trial.train(label, profile)
            args = launch_mock.call_args.args[0]
            self.assertEqual(args[args.index("--total-updates") + 1], "446")
            self.assertEqual(args[args.index("--init-checkpoint") + 1], str(self.trial.source))
            self.assertEqual(args[args.index("--eval-every") + 1], "0")
            self.assertEqual(args[args.index("--reward-profile") + 1], profile)
            self.assertEqual(args[args.index("--ent-coef") + 1], "0.005")
            self.assertEqual(float(args[args.index("--episode-time-limit") + 1]), 150)
            self.assertEqual(float(args[args.index("--observation-time-limit") + 1]), 60)
            self.assertEqual(checkpoint.name, "ppo_checkpoint.pth")

    def test_incomplete_sampling_budget_is_rejected_even_at_expected_update(self):
        self.trial.prepare()
        child = MagicMock(pid=1234)
        child.poll.return_value = 0
        child.wait.return_value = 0

        def launch(*args, **kwargs):
            config = dict(self.saved["training_config"], reward_profile=trial_module.DODGE_PROFILE,
                          reward_config=asdict(self.dodge), episode_time_limit=150)
            torch.save(dict(self.saved, update=445, global_step=913407,
                            training_config=config), self.trial.directory / "candidate/ppo_checkpoint.pth")
            return child

        with patch.object(self.trial, "stage"), patch.object(self.trial, "check_idle"), \
                patch.object(trial_module.subprocess, "Popen", side_effect=launch):
            with self.assertRaisesRegex(RuntimeError, "budget/settings"):
                self.trial.train("candidate", trial_module.DODGE_PROFILE)

    def test_invalid_episode_time_limits_are_rejected_without_accessing_games(self):
        for value in (0, -1, float("nan"), float("inf"), None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite and positive"):
                trial_module.DodgeRewardTrial(self.root / "invalid", self.source,
                                             self.digest, .005, self.endpoints,
                                             episode_time_limit=value)

    def test_wrong_checkpoint_time_limit_is_rejected_at_full_sample_budget(self):
        self.trial.prepare()
        child = MagicMock(pid=1234)
        child.poll.return_value = 0
        child.wait.return_value = 0

        def launch(*args, **kwargs):
            config = dict(self.saved["training_config"], reward_profile=trial_module.DODGE_PROFILE,
                          reward_config=asdict(self.dodge), episode_time_limit=120)
            torch.save(dict(self.saved, update=445, global_step=913408,
                            training_config=config), self.trial.directory / "candidate/ppo_checkpoint.pth")
            return child

        with patch.object(self.trial, "stage"), patch.object(self.trial, "check_idle"), \
                patch.object(trial_module.subprocess, "Popen", side_effect=launch):
            with self.assertRaisesRegex(RuntimeError, "budget/settings"):
                self.trial.train("candidate", trial_module.DODGE_PROFILE)

    def test_fixed_evaluation_uses_150_seconds_without_changing_observation_scale(self):
        self.trial.prepare()
        saved = dict(self.saved, model_hidden=256, model={})
        normalizer = MagicMock()
        normalizer.state_dict.return_value = {"count": 1}

        def evaluate(pool, model, normalize, episodes, record):
            self.assertEqual(episodes, 10)
            for index in range(4):
                for episode in range(10):
                    record({"env": index, "episode_in_eval": episode,
                            "info": {"boss_level": 2, "reset_mode": "scene",
                                     "damage_tracking_valid": True, "won": False,
                                     "damage": 9, "min_boss_health": 400, "hits": 30,
                                     "truncated_reason": None}})

        with patch.object(self.trial, "stage"), patch.object(self.trial, "check_idle"), \
                patch.object(trial_module, "load_checkpoint", return_value=(saved, self.digest)), \
                patch.object(trial_module, "ActorCritic"), \
                patch.object(trial_module, "RunningNorm", return_value=normalizer), \
                patch.object(trial_module, "ExternalEnvPool") as pool, \
                patch.object(trial_module, "evaluate_pool", side_effect=evaluate):
            summary = self.trial.evaluate(self.trial.source, "fixed_150")
        args = pool.call_args.args
        self.assertEqual((args[3], args[7]), (150, 60))
        self.assertEqual((summary["episode_time_limit"], summary["observation_time_limit"]), (150, 60))
        self.assertEqual(summary["episodes"], 40)
        pool.return_value.close.assert_called_once()

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
