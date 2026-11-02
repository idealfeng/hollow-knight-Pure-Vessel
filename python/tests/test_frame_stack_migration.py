"""Full-policy 4-to-16-frame expansion preserves behavior and trained Adam."""

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
from migrate_frame_stack import (migrate_checkpoint, source_column_indices,
                                 TARGET_REWARD_PROFILE)
from observation_schema import (COMBAT_FRAME_WIDTH, COMBAT_SCHEMA,
                                LEGACY_FRAME_WIDTH, LEGACY_SCHEMA, observation_width)
from obs_norm import RunningNorm
from ppo_model import ActorCritic


class FrameStackMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.pth"
        self.model, self.saved = self.make_source(LEGACY_SCHEMA)
        torch.save(self.saved, self.source)

    def make_source(self, schema):
        torch.manual_seed(5)
        width = observation_width(schema) * 4
        model = ActorCritic(width, [1] * 7 + [4], hidden=256,
                            bernoulli=True, exclusive_directions=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, amsgrad=True)
        logits, values = model(torch.randn(8, width))
        (sum(x.square().mean() for x in logits) + values.square().mean()).backward()
        optimizer.step()
        normalizer = RunningNorm(width, clip=7.0)
        # Nontrivial world-space calibration exposes accidental mean=0/var=1.
        normalizer.mean = np.linspace(20, 200, width)
        normalizer.var = np.linspace(.8, 40, width)
        normalizer.count = 5500.0
        source = {
            "format_version": 9, "observation_size": width, "frame_stack": 4,
            "observation_schema": schema, "action_names": list(SPELL_ACTION_NAMES),
            "action_sizes": [1] * 7 + [4], "model_hidden": 256,
            "action_distribution": "exclusive_directions_spells_v1", "control_hz": 10,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "obs_norm": normalizer.state_dict(), "freeze_obs_norm": True,
            "update": 550, "global_step": 1128448, "episode": 3200,
            "training_config": {
                "reward_profile": TARGET_REWARD_PROFILE,
                "reward_config": asdict(get_reward_config(TARGET_REWARD_PROFILE)),
                "episode_time_limit": 150, "observation_time_limit": 60,
                "boss_level": 2, "lr": .0001, "ent_coef": .005,
                "game_speed": 1.0, "train_mode": "all", "freeze_obs_norm": True,
                "observation_schema": schema, "frame_stack": 4, "observation_size": width,
            },
            "pretraining": {"kind": "previous transfer"},
            "evaluation": {"update": 550, "nohit_wins": 2, "episodes": 40},
        }
        return model, source

    def migrate(self, target_schema=LEGACY_SCHEMA):
        destination = self.root / f"expanded_{target_schema}.pth"
        return migrate_checkpoint(self.source, destination, target_schema=target_schema)

    def load_model(self, result):
        model = ActorCritic(result["observation_size"], [1] * 7 + [4], hidden=256,
                            bernoulli=True, exclusive_directions=True)
        model.load_state_dict(result["model"], strict=True)
        return model

    def assert_tree_equal(self, expected, actual):
        if torch.is_tensor(expected):
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(expected, actual)
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.assert_tree_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for a, b in zip(expected, actual):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(expected, actual)

    def assert_behavior_preserved(self, source_model, source, result):
        schema = source.get("observation_schema", LEGACY_SCHEMA)
        indices = source_column_indices(schema, result["observation_schema"])
        expanded = self.load_model(result)
        old_norm, new_norm = RunningNorm(source["observation_size"]), RunningNorm(result["observation_size"])
        old_norm.load_state_dict(source["obs_norm"])
        new_norm.load_state_dict(result["obs_norm"])
        random = np.random.default_rng(42)
        old_raw = (source["obs_norm"]["mean"]
                   + random.normal(size=(32, source["observation_size"]))
                   * np.sqrt(source["obs_norm"]["var"]))
        new_raw = random.normal(size=(32, result["observation_size"])) * 10000
        new_raw[:, indices] = old_raw
        old_input, new_input = old_norm.normalize(old_raw), new_norm.normalize(new_raw)
        old_logits, old_value = source_model(torch.as_tensor(old_input))
        new_logits, new_value = expanded(torch.as_tensor(new_input))
        for before, after in zip(old_logits + [old_value], new_logits + [new_value]):
            torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
        self.assertEqual([source_model.greedy_act(x) for x in old_input],
                         [expanded.greedy_act(x) for x in new_input])
        for old, new in zip(old_input[:5], new_input[:5]):
            self.assertAlmostEqual(source_model.value_only(old), expanded.value_only(new), places=6)
        # Arbitrary extra history and optional combat features contribute zero.
        changed = new_raw.copy()
        extra = np.setdiff1d(np.arange(result["observation_size"]), indices)
        changed[:, extra] = random.uniform(-1e9, 1e9, size=(32, len(extra)))
        changed_logits, changed_value = expanded(torch.as_tensor(new_norm.normalize(changed)))
        for before, after in zip(new_logits + [new_value], changed_logits + [changed_value]):
            torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_legacy_and_combat_targets_preserve_real_forward_greedy_and_value(self):
        for schema in (LEGACY_SCHEMA, COMBAT_SCHEMA):
            with self.subTest(schema=schema):
                self.assert_behavior_preserved(self.model, self.saved, self.migrate(schema))

    def test_statistics_calibrate_new_history_and_preserve_recent_frames_and_freeze(self):
        for schema in (LEGACY_SCHEMA, COMBAT_SCHEMA):
            with self.subTest(schema=schema):
                result = self.migrate(schema)
                frame_width = observation_width(schema)
                indices = source_column_indices(LEGACY_SCHEMA, schema)
                for key in ("mean", "var"):
                    old, new = self.saved["obs_norm"][key], result["obs_norm"][key]
                    np.testing.assert_array_equal(new[indices], old)
                    for frame in range(12):
                        np.testing.assert_array_equal(new[frame * frame_width:frame * frame_width + 118], old[:118])
                    if schema == COMBAT_SCHEMA:
                        for frame in range(16):
                            np.testing.assert_array_equal(new[frame * frame_width + 118:(frame + 1) * frame_width],
                                                          0 if key == "mean" else 1)
                for key in ("count", "clip"):
                    self.assertEqual(result["obs_norm"][key], self.saved["obs_norm"][key])
                self.assertIs(result["freeze_obs_norm"], self.saved["freeze_obs_norm"])

    def test_all_model_and_adam_states_are_preserved_and_real_adam_step_can_learn_history(self):
        for schema in (LEGACY_SCHEMA, COMBAT_SCHEMA):
            with self.subTest(schema=schema):
                result = self.migrate(schema)
                indices = source_column_indices(LEGACY_SCHEMA, schema)
                extra = np.setdiff1d(np.arange(result["observation_size"]), indices)
                for name, expected in self.saved["model"].items():
                    actual = result["model"][name]
                    if name == "trunk.0.weight":
                        self.assertTrue(torch.equal(actual[:, indices], expected))
                        self.assertEqual(torch.count_nonzero(actual[:, extra]).item(), 0)
                    else:
                        self.assertTrue(torch.equal(expected, actual))
                first_id = self.saved["optimizer"]["param_groups"][0]["params"][0]
                for identifier, old_state in self.saved["optimizer"]["state"].items():
                    new_state = result["optimizer"]["state"][identifier]
                    for key, expected in old_state.items():
                        if identifier == first_id and key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                            self.assertTrue(torch.equal(expected, new_state[key][:, indices]))
                            self.assertEqual(torch.count_nonzero(new_state[key][:, extra]).item(), 0)
                        else:
                            self.assert_tree_equal(expected, new_state[key])
                self.assert_tree_equal(self.saved["optimizer"]["param_groups"], result["optimizer"]["param_groups"])
                model = self.load_model(result)
                optimizer = torch.optim.Adam(model.parameters(), lr=.0001, amsgrad=True)
                optimizer.load_state_dict(result["optimizer"])
                logits, values = model(torch.randn(8, result["observation_size"]))
                (sum(x.square().mean() for x in logits) + values.square().mean()).backward()
                optimizer.step()
                self.assertGreater(torch.count_nonzero(model.trunk[0].weight[:, extra]).item(), 0)
                self.assertEqual(optimizer.state[model.trunk[0].weight]["step"].item(), 2)

    def test_source_counters_reward_metadata_rng_and_exclusive_destination(self):
        original = self.source.read_bytes()
        rng = torch.get_rng_state().clone()
        result = migrate_checkpoint(self.source, self.root / "expanded.pth")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(original, self.source.read_bytes())
        self.assertEqual(result["frame_stack"], 16)
        self.assertEqual(result["observation_schema"], LEGACY_SCHEMA)
        self.assertEqual(result["observation_size"], 1888)
        self.assertEqual(result["observation_frame_size"], LEGACY_FRAME_WIDTH)
        for key in ("update", "global_step", "episode"):
            self.assertEqual(result[key], self.saved[key])
        config = result["training_config"]
        self.assertEqual(config["frame_stack"], 16)
        self.assertEqual(config["observation_size"], 1888)
        self.assertEqual(config["start_update"], 551)
        self.assertEqual(config["reward_profile"], TARGET_REWARD_PROFILE)
        self.assertEqual(config["reward_config"], asdict(get_reward_config(TARGET_REWARD_PROFILE)))
        self.assertNotIn("potential", config["reward_config"])
        self.assertNotIn("evaluation", result)
        provenance = result["frame_stack_migration"]
        self.assertEqual(provenance["source_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(provenance["previous_evaluation"], self.saved["evaluation"])
        self.assertEqual(provenance["previous_pretraining"], self.saved["pretraining"])
        self.assertFalse(provenance["optimizer_reset"])
        self.assertFalse(provenance["critic_reset"])
        destination = self.root / "expanded.pth"
        payload = destination.read_bytes()
        with self.assertRaises(FileExistsError):
            migrate_checkpoint(self.source, destination)
        self.assertEqual(payload, destination.read_bytes())
        with self.assertRaisesRegex(ValueError, "source must remain unchanged"):
            migrate_checkpoint(self.source, self.source)

    def test_existing_combat_source_preserves_all_128_recent_features_and_restores_old_reward(self):
        model, source = self.make_source(COMBAT_SCHEMA)
        source["training_config"].update(reward_profile="radiant_potential_v1",
                                         reward_config=asdict(get_reward_config("radiant_potential_v1")))
        torch.save(source, self.source)
        result = self.migrate(COMBAT_SCHEMA)
        self.assertEqual(result["observation_size"], 16 * COMBAT_FRAME_WIDTH)
        self.assert_behavior_preserved(model, source, result)
        self.assertEqual(result["training_config"]["reward_profile"], TARGET_REWARD_PROFILE)
        self.assertEqual(result["training_config"]["reward_config"],
                         asdict(get_reward_config(TARGET_REWARD_PROFILE)))
        with self.assertRaisesRegex(ValueError, "cannot discard"):
            self.migrate(LEGACY_SCHEMA)
        self.assertFalse((self.root / f"expanded_{LEGACY_SCHEMA}.pth").exists())

    def test_incompatible_incomplete_unfrozen_or_non21_source_is_rejected(self):
        invalid = []
        for key, value in (("format_version", 8), ("action_sizes", [1] * 7),
                           ("frame_stack", 8), ("model_hidden", 128),
                           ("freeze_obs_norm", False), ("initialization_only", True),
                           ("observation_size", 512), ("nail_damage", 33)):
            source = copy.deepcopy(self.saved)
            source[key] = value
            invalid.append(source)
        source = copy.deepcopy(self.saved)
        source.pop("optimizer")
        invalid.append(source)
        for key, value in (("nail_damage", 33), ("combat_charms", [25]),
                           ("freeze_obs_norm", False), ("observation_schema", COMBAT_SCHEMA)):
            source = copy.deepcopy(self.saved)
            source["training_config"][key] = value
            invalid.append(source)
        source = copy.deepcopy(self.saved)
        source["training_config"]["reward_config"]["nail_damage_reference"] = 33
        invalid.append(source)
        source = copy.deepcopy(self.saved)
        source["optimizer"]["state"] = {}
        invalid.append(source)
        source = copy.deepcopy(self.saved)
        source["optimizer"]["state"][0]["exp_avg"] = torch.zeros(256, 1)
        invalid.append(source)
        source = copy.deepcopy(self.saved)
        source["obs_norm"]["var"][0] = np.nan
        invalid.append(source)
        for index, source in enumerate(invalid):
            with self.subTest(case=index):
                torch.save(source, self.source)
                with self.assertRaises(ValueError):
                    self.migrate()
                self.assertFalse((self.root / f"expanded_{LEGACY_SCHEMA}.pth").exists())


if __name__ == "__main__":
    unittest.main()
