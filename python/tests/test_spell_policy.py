"""Spell extension keeps the original actor and a correct joint PPO objective."""

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from migrate_spell_policy import (BASE_ACTION_NAMES, SPELL_ACTION_SIZES,
                                 migrate_checkpoint)
from obs_norm import RunningNorm
from ppo_model import ActorCritic
from ppo_training import ppo_update
from rollout_buffer import RolloutBuffer


class SpellPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(13)
        self.model = ActorCritic(8, SPELL_ACTION_SIZES, hidden=16,
                                 bernoulli=True, exclusive_directions=True)

    def test_spell_selection_is_categorical_and_logprobs_match_training(self):
        observations = np.random.default_rng(13).normal(size=(80, 8)).astype(np.float32)
        actions, sampled_logprobs = [], []
        for observation in observations:
            action, logprob, joint, _ = self.model.act(observation)
            self.assertEqual(len(action), 8)
            self.assertIn(action[7], range(4))
            self.assertLessEqual(action[0] + action[1], 1)
            self.assertLessEqual(action[2] + action[3], 1)
            self.assertAlmostEqual(joint, sum(logprob), places=7)
            actions.append(action)
            sampled_logprobs.append(logprob)
        logprob, entropy, _ = self.model.evaluate(
            torch.tensor(observations), torch.tensor(actions))
        torch.testing.assert_close(logprob, torch.tensor(sampled_logprobs), atol=1e-6, rtol=1e-6)
        self.assertEqual(tuple(entropy.shape), (80, 8))
        self.assertTrue(torch.isfinite(entropy).all())

    def test_greedy_spell_selects_exactly_one_choice(self):
        with torch.no_grad():
            self.model.heads[7].weight.zero_()
            self.model.heads[7].bias.copy_(torch.tensor([-2.0, -1.0, 1.0, 2.0]))
        action = self.model.greedy_act(np.zeros(8, dtype=np.float32))
        self.assertEqual(action[7], 3)
        with self.assertRaisesRegex(ValueError, "seven-button"):
            self.model.configure_trainable_parameters("dash_only")

    def test_invalid_spell_choices_do_not_get_silently_clipped(self):
        for choice in (-1, 4, 1.5):
            action = torch.tensor([[0, 0, 0, 0, 0, 0, 0, choice]])
            with self.subTest(choice=choice), self.assertRaises(ValueError):
                self.model.evaluate(torch.zeros(1, 8), action)

    def test_real_ppo_update_trains_new_spell_head(self):
        buffer = RolloutBuffer(128, 8, 8)
        rng = np.random.default_rng(44)
        for index in range(128):
            observation = rng.normal(size=8).astype(np.float32)
            action, logprobs, _, value = self.model.act(observation)
            reward = 2.0 if action[7] == 3 else -0.5
            buffer.add(observation, action, logprobs, value, reward, index % 8 == 7)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, eps=1e-5)
        before = self.model.heads[7].weight.detach().clone()
        result = ppo_update(self.model, optimizer, buffer, 0.0, epochs=2,
                            minibatch_size=32, target_kl=None, ent_coef=0.005)
        self.assertFalse(torch.equal(before, self.model.heads[7].weight))
        self.assertTrue(torch.isfinite(self.model.heads[7].weight).all())
        self.assertEqual(len(result["entropy_per_head"]), 8)


class SpellMigrationTests(unittest.TestCase):
    def source(self):
        torch.manual_seed(73)
        model = ActorCritic(8, [1] * 7, hidden=16,
                            bernoulli=True, exclusive_directions=True)
        normalizer = RunningNorm(8)
        normalizer.update(np.random.default_rng(4).normal(size=(12, 8)))
        return model, {
            "format_version": 8, "observation_size": 8, "frame_stack": 1,
            "model_hidden": 16, "action_distribution": "exclusive_directions_v1",
            "action_names": list(BASE_ACTION_NAMES), "action_sizes": [1] * 7,
            "control_hz": 10, "model": copy.deepcopy(model.state_dict()),
            "obs_norm": normalizer.state_dict(), "update": 1000,
            "global_step": 2050048, "episode": 8000,
            "optimizer": {"legacy_optimizer_marker": True},
        }

    def test_migration_preserves_actor_and_normalizer_resets_training_state(self):
        old_model, source = self.source()
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "old.pth"
            output_path = Path(directory) / "new.pth"
            torch.save(source, source_path)
            payload = source_path.read_bytes()
            initialized = migrate_checkpoint(source_path, output_path)
            self.assertEqual(source_path.read_bytes(), payload)
            self.assertTrue(output_path.exists())
            self.assertEqual(initialized["format_version"], 9)
            self.assertEqual(initialized["update"], -1)
            self.assertEqual(initialized["episode"], 0)
            self.assertEqual(initialized["global_step"], 0)
            self.assertTrue(initialized["initialization_only"])
            self.assertNotIn("optimizer", initialized)
            self.assertEqual(initialized["pretraining"]["source_sha256"], hashlib.sha256(payload).hexdigest())
            for key, expected in source["model"].items():
                if key.startswith("critic."):
                    continue
                self.assertTrue(torch.equal(initialized["model"][key], expected), key)
            self.assertFalse(torch.equal(initialized["model"]["critic.weight"], source["model"]["critic.weight"]))
            for key, expected in source["obs_norm"].items():
                self.assertTrue(np.array_equal(initialized["obs_norm"][key], expected), key)
            model = ActorCritic(8, initialized["action_sizes"], hidden=16,
                                bernoulli=True, exclusive_directions=True)
            model.load_state_dict(initialized["model"])
            for observation in np.random.default_rng(2).normal(size=(10, 8)).astype(np.float32):
                action = model.greedy_act(observation)
                self.assertEqual(action[:7], old_model.greedy_act(observation))
                self.assertEqual(action[7], 0)
            logits, _ = model(torch.randn(10, 8))
            torch.testing.assert_close(logits[7].softmax(-1),
                                       torch.tensor([[0.91, 0.03, 0.03, 0.03]]).expand(10, -1))
            with self.assertRaises(FileExistsError):
                migrate_checkpoint(source_path, output_path)
            with self.assertRaisesRegex(ValueError, "source must remain unchanged"):
                migrate_checkpoint(source_path, source_path)

    def test_rejects_mismatched_source_action_semantics(self):
        _, source = self.source()
        source["action_names"][6] = "cast"
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "old.pth"
            torch.save(source, source_path)
            with self.assertRaisesRegex(ValueError, "format-8"):
                migrate_checkpoint(source_path, Path(directory) / "new.pth")


if __name__ == "__main__":
    unittest.main()
