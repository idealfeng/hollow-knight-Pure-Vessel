"""Behavior-preserving observation expansion, including real Adam updates."""

import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from agent.action import SPELL_ACTION_NAMES
from agent.reward import get_reward_config
from migrate_combat_policy import migrate_checkpoint, old_column_indices
from observation_schema import COMBAT_FRAME_WIDTH, COMBAT_SCHEMA, LEGACY_FRAME_WIDTH
from obs_norm import RunningNorm
from ppo_model import ActorCritic


class CombatMigrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.pth"
        self.destination = self.root / "expanded.pth"
        torch.manual_seed(5)
        self.old_width, self.new_width = 4 * LEGACY_FRAME_WIDTH, 4 * COMBAT_FRAME_WIDTH
        self.indices = old_column_indices(4)
        self.model = ActorCritic(self.old_width, [1] * 7 + [4], hidden=16,
                                 bernoulli=True, exclusive_directions=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.0001, amsgrad=True)
        logits, values = self.model(torch.randn(8, self.old_width))
        (sum(t.square().mean() for t in logits) + values.square().mean()).backward()
        optimizer.step()
        normalizer = RunningNorm(self.old_width)
        normalizer.mean = np.linspace(-.5, .5, self.old_width)
        normalizer.var = np.linspace(.8, 1.5, self.old_width)
        normalizer.count = 5500.0
        self.saved = {
            "format_version": 9, "observation_size": self.old_width, "frame_stack": 4,
            "action_names": list(SPELL_ACTION_NAMES), "action_sizes": [1] * 7 + [4],
            "action_distribution": "exclusive_directions_spells_v1", "model_hidden": 16,
            "control_hz": 10, "model": self.model.state_dict(), "optimizer": optimizer.state_dict(),
            "obs_norm": normalizer.state_dict(), "freeze_obs_norm": True,
            "update": 550, "global_step": 1128448, "episode": 3200,
            "training_config": {"reward_profile": "radiant_spells_dodge_v1",
                                "reward_config": asdict(get_reward_config("radiant_spells_dodge_v1")),
                                "episode_time_limit": 150, "observation_time_limit": 60,
                                "boss_level": 2, "lr": .0001, "ent_coef": .005},
            "pretraining": {"kind": "preserved prior history"},
            "evaluation": {"update": 550, "nohit_wins": 2},
        }
        torch.save(self.saved, self.source)

    def migrate(self):
        return migrate_checkpoint(self.source, self.destination)

    def assert_tensor_tree_equal(self, a, b):
        if torch.is_tensor(a):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_tensor_tree_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tensor_tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_old_actor_critic_outputs_and_greedy_actions_ignore_arbitrary_new_features_initially(self):
        result = self.migrate()
        expanded = ActorCritic(self.new_width, [1] * 7 + [4], hidden=16,
                               bernoulli=True, exclusive_directions=True)
        expanded.load_state_dict(result["model"])
        old_norm, new_norm = RunningNorm(self.old_width), RunningNorm(self.new_width)
        old_norm.load_state_dict(self.saved["obs_norm"])
        new_norm.load_state_dict(result["obs_norm"])
        random = np.random.default_rng(42)
        old_raw = random.normal(size=(32, self.old_width))
        new_raw = random.normal(size=(32, self.new_width)) * 100
        new_raw[:, self.indices] = old_raw
        old_input, new_input = old_norm.normalize(old_raw), new_norm.normalize(new_raw)
        old_logits, old_value = self.model(torch.as_tensor(old_input))
        new_logits, new_value = expanded(torch.as_tensor(new_input))
        for a, b in zip(old_logits + [old_value], new_logits + [new_value]):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
        self.assertEqual([self.model.greedy_act(x) for x in old_input],
                         [expanded.greedy_act(x) for x in new_input])

    def test_all_original_parameters_and_normalizer_columns_are_retained(self):
        result = self.migrate()
        for name, value in self.saved["model"].items():
            actual = result["model"][name]
            if name == "trunk.0.weight":
                self.assertTrue(torch.equal(actual[:, self.indices], value))
                extra = np.setdiff1d(np.arange(self.new_width), self.indices)
                self.assertEqual(torch.count_nonzero(actual[:, extra]).item(), 0)
            else:
                self.assertTrue(torch.equal(actual, value))
        np.testing.assert_array_equal(result["obs_norm"]["mean"][self.indices], self.saved["obs_norm"]["mean"])
        np.testing.assert_array_equal(result["obs_norm"]["var"][self.indices], self.saved["obs_norm"]["var"])
        extra = np.setdiff1d(np.arange(self.new_width), self.indices)
        np.testing.assert_array_equal(result["obs_norm"]["mean"][extra], 0)
        np.testing.assert_array_equal(result["obs_norm"]["var"][extra], 1)
        for key in ("count", "clip"):
            self.assertEqual(result["obs_norm"][key], self.saved["obs_norm"][key])

    def test_adam_moments_are_mapped_exactly_and_a_real_update_can_learn_new_columns(self):
        result = self.migrate()
        first_id = self.saved["optimizer"]["param_groups"][0]["params"][0]
        for identifier, old_state in self.saved["optimizer"]["state"].items():
            new_state = result["optimizer"]["state"][identifier]
            for key, value in old_state.items():
                if identifier == first_id and key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    self.assertTrue(torch.equal(new_state[key][:, self.indices], value))
                    extra = np.setdiff1d(np.arange(self.new_width), self.indices)
                    self.assertEqual(torch.count_nonzero(new_state[key][:, extra]).item(), 0)
                else:
                    self.assert_tensor_tree_equal(value, new_state[key])
        self.assert_tensor_tree_equal(self.saved["optimizer"]["param_groups"], result["optimizer"]["param_groups"])
        expanded = ActorCritic(self.new_width, [1] * 7 + [4], hidden=16,
                               bernoulli=True, exclusive_directions=True)
        expanded.load_state_dict(result["model"])
        optimizer = torch.optim.Adam(expanded.parameters(), lr=.0001, amsgrad=True)
        optimizer.load_state_dict(result["optimizer"])
        logits, values = expanded(torch.randn(8, self.new_width))
        (sum(x.square().mean() for x in logits) + values.square().mean()).backward()
        optimizer.step()
        extra = np.setdiff1d(np.arange(self.new_width), self.indices)
        self.assertGreater(torch.count_nonzero(expanded.trunk[0].weight[:, extra]).item(), 0)
        self.assertEqual(optimizer.state[expanded.trunk[0].weight]["step"].item(), 2)

    def test_source_bytes_counters_and_provenance_are_preserved_and_destination_is_exclusive(self):
        payload = self.source.read_bytes()
        result = self.migrate()
        self.assertEqual(payload, self.source.read_bytes())
        for key in ("update", "global_step", "episode"):
            self.assertEqual(result[key], self.saved[key])
        self.assertEqual(result["observation_schema"], COMBAT_SCHEMA)
        self.assertEqual(result["observation_size"], self.new_width)
        self.assertEqual(result["training_config"]["reward_profile"], "radiant_potential_v1")
        self.assertEqual(result["training_config"]["start_update"], 551)
        self.assertTrue(result["freeze_obs_norm"])
        history = result["combat_migration"]
        self.assertEqual(history["source_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(history["previous_pretraining"], self.saved["pretraining"])
        self.assertFalse(history["critic_reset"])
        self.assertFalse(history["optimizer_reset"])
        preserved = self.destination.read_bytes()
        with self.assertRaises(FileExistsError):
            self.migrate()
        self.assertEqual(self.destination.read_bytes(), preserved)
        with self.assertRaisesRegex(ValueError, "source must remain unchanged"):
            migrate_checkpoint(self.source, self.source)

    def test_incompatible_or_incomplete_source_is_rejected_without_destination(self):
        invalid = []
        for key, value in (("format_version", 8), ("action_sizes", [1] * 7),
                           ("observation_schema", COMBAT_SCHEMA), ("observation_size", self.new_width)):
            item = copy.deepcopy(self.saved)
            item[key] = value
            invalid.append(item)
        item = copy.deepcopy(self.saved)
        item.pop("optimizer")
        invalid.append(item)
        item = copy.deepcopy(self.saved)
        item["optimizer"]["state"][0]["exp_avg"] = torch.zeros(16, 1)
        invalid.append(item)
        for source in invalid:
            with self.subTest(source_keys=list(source)):
                torch.save(source, self.source)
                with self.assertRaises(ValueError):
                    self.migrate()
                self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
