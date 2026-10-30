"""Real PPO updates must preserve the fixed actor, including Adam momentum."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import main_multi
from ppo_model import ActorCritic
from ppo_training import ppo_update
from rollout_buffer import RolloutBuffer


class DashOnlyTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(51)
        np.random.seed(51)
        self.model = ActorCritic(8, [1] * 7, hidden=16, bernoulli=True,
                                 exclusive_directions=True)

    def rollout(self, model):
        buffer = RolloutBuffer(64, 8, 7)
        rng = np.random.default_rng(56)
        for index in range(64):
            observation = rng.normal(size=8).astype(np.float32)
            actions, logprobs, _, value = model.act(observation)
            reward = 1.0 if actions[6] else -0.75
            buffer.add(observation, actions, logprobs, value, reward, index % 8 == 7)
        return buffer

    def update(self, model, optimizer):
        return ppo_update(model, optimizer, self.rollout(model), 0.0,
                          epochs=2, minibatch_size=16, target_kl=None,
                          ent_coef=0.005, value_clip_eps=1.0)

    def test_actual_ppo_with_restored_momentum_only_changes_dash_and_critic(self):
        source_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, eps=1e-5)
        self.update(self.model, source_optimizer)
        # Restore a real all-head optimizer state as the live fork does.
        restored = ActorCritic(8, [1] * 7, hidden=16, bernoulli=True,
                              exclusive_directions=True)
        restored.load_state_dict(copy.deepcopy(self.model.state_dict()))
        optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3, eps=1e-5)
        optimizer.load_state_dict(copy.deepcopy(source_optimizer.state_dict()))
        self.assertTrue(all(parameter in optimizer.state for parameter in restored.parameters()))
        names = restored.configure_trainable_parameters("dash_only")
        self.assertEqual(names, ["heads.6.weight", "heads.6.bias", "critic.weight", "critic.bias"])
        before = {name: parameter.detach().clone() for name, parameter in restored.named_parameters()}
        observations = torch.randn(12, 8)
        before_logits, _ = restored(observations)
        before_logits = [logits.detach().clone() for logits in before_logits]
        frozen_moments = {name: copy.deepcopy(optimizer.state[parameter])
                          for name, parameter in restored.named_parameters() if not parameter.requires_grad}
        self.update(restored, optimizer)
        after_logits, _ = restored(observations)
        changed = {name for name, parameter in restored.named_parameters()
                   if not torch.equal(parameter, before[name])}
        self.assertTrue(any(name.startswith("heads.6.") for name in changed))
        self.assertTrue(any(name.startswith("critic.") for name in changed))
        self.assertTrue(changed.issubset(set(names)))
        for index in range(6):
            self.assertTrue(torch.equal(before_logits[index], after_logits[index]), index)
        for name, parameter in restored.named_parameters():
            if not parameter.requires_grad:
                self.assertIsNone(parameter.grad, name)
                for key, expected in frozen_moments[name].items():
                    actual = optimizer.state[parameter][key]
                    if torch.is_tensor(expected):
                        self.assertTrue(torch.equal(actual, expected), (name, key))
                    else:
                        self.assertEqual(actual, expected)

    def test_freeze_mode_preserves_sampling_and_logprob_semantics(self):
        observation = np.arange(8, dtype=np.float32) / 8
        torch.manual_seed(31)
        before = self.model.act(observation)
        self.model.configure_trainable_parameters("dash_only")
        torch.manual_seed(31)
        after = self.model.act(observation)
        self.assertEqual(before, after)
        logits, entropy, _ = self.model.evaluate(
            torch.tensor(observation).unsqueeze(0), torch.tensor(after[0]).unsqueeze(0))
        torch.testing.assert_close(logits[0], torch.tensor(after[1]))
        self.assertGreater(float(entropy[0, :6].sum().detach()), 0)

    def test_default_all_mode_restores_normal_updates_and_rejects_unknown_mode(self):
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.parameters()))
        self.model.configure_trainable_parameters("dash_only")
        with self.assertRaises(ValueError):
            self.model.configure_trainable_parameters("unknown")
        self.model.configure_trainable_parameters("all")
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.parameters()))
        legacy = ActorCritic(8, [2, 3], hidden=8)
        with self.assertRaisesRegex(ValueError, "seven-button"):
            legacy.configure_trainable_parameters("dash_only")


class TrainingModeConfigurationTests(unittest.TestCase):
    def args(self, directory, mode=None):
        argv = ["main_multi.py", "--run-dir", directory, "--no-monitor"]
        if mode is not None:
            argv += ["--train-mode", mode]
        with patch.object(sys, "argv", argv):
            return main_multi.parse_args()

    def test_cli_default_inherits_and_explicit_dash_mode_parses(self):
        self.assertIsNone(self.args("unused").train_mode)
        self.assertEqual(self.args("unused", "dash_only").train_mode, "dash_only")

    def test_existing_run_inherits_mode_and_rejects_switch_before_game_access(self):
        class BeforeGameAccess(Exception):
            pass
        for saved_mode in (None, "all", "dash_only"):
            with self.subTest(saved_mode=saved_mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "ppo_checkpoint.pth").touch()
                config = {} if saved_mode is None else {"train_mode": saved_mode}
                (root / "run_config.json").write_text(json.dumps(config))
                args = self.args(directory)
                with patch.object(main_multi, "parse_args", return_value=args), \
                        patch.object(main_multi, "prepare_fights", side_effect=BeforeGameAccess):
                    with self.assertRaises(BeforeGameAccess):
                        main_multi.main()
                self.assertEqual(args.train_mode, saved_mode or "all")
                incompatible = "all" if saved_mode == "dash_only" else "dash_only"
                args = self.args(directory, incompatible)
                with patch.object(main_multi, "parse_args", return_value=args), \
                        patch.object(main_multi, "prepare_fights") as prepare:
                    with self.assertRaisesRegex(SystemExit, "Cannot change train_mode"):
                        main_multi.main()
                    prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
