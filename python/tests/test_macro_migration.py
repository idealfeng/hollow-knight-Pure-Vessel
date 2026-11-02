"""Canonical conditional probabilities, preserved critic, and fresh Adam."""

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
from macro_actions import (MACRO_ACTIONS, MACRO_DISTRIBUTION, NUM_MACRO_ACTIONS,
                           macro_metadata, validate_macro_checkpoint)
from migrate_frame_stack import source_column_indices
from migrate_macro_policy import migrate_checkpoint
from obs_norm import RunningNorm
from ppo_model import ActorCritic


class MacroMigrationTests(unittest.TestCase):
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
        self.destination = self.root / "macro.pth"
        torch.manual_seed(13)
        self.old_model = ActorCritic(472, [1] * 7 + [4], hidden=256,
                                     bernoulli=True, exclusive_directions=True)
        optimizer = torch.optim.Adam(self.old_model.parameters(), lr=.0001, eps=1e-5, amsgrad=True)
        # Fit every parameter and ensure there really are Adam histories to reset.
        logits, value = self.old_model(torch.randn(16, 472))
        (sum(x.square().mean() for x in logits) + value.square().mean()).backward()
        optimizer.step()
        norm = RunningNorm(472, clip=8.0)
        norm.mean = np.linspace(10, 100, 472)
        norm.var = np.linspace(.5, 30, 472)
        norm.count = 12000.0
        profile = "radiant_spells_dodge_v1"
        self.saved = {
            "format_version": 9, "observation_size": 472, "frame_stack": 4,
            "observation_schema": "legacy_v1", "action_names": list(SPELL_ACTION_NAMES),
            "action_sizes": [1] * 7 + [4], "model_hidden": 256,
            "action_distribution": "exclusive_directions_spells_v1", "control_hz": 10,
            "model": self.old_model.state_dict(), "optimizer": optimizer.state_dict(),
            "obs_norm": norm.state_dict(), "freeze_obs_norm": True,
            "update": 550, "global_step": 1128448, "episode": 3200,
            "training_config": {
                "reward_profile": profile, "reward_config": asdict(get_reward_config(profile)),
                "episode_time_limit": 150, "observation_time_limit": 60, "boss_level": 2,
                "lr": .0001, "ent_coef": .005, "game_speed": 1.0, "train_mode": "all",
                "freeze_obs_norm": True, "observation_schema": "legacy_v1",
                "frame_stack": 4, "observation_size": 472,
            },
            "inference_config": {"observation_schema": "legacy_v1", "frame_stack": 4},
            "pretraining": {"kind": "previous transfer"},
            "evaluation": {"update": 550, "nohit_wins": 2, "episodes": 40},
        }
        torch.save(self.saved, self.source)

    def migrate(self):
        return migrate_checkpoint(self.source, self.destination)

    def model(self, saved):
        model = ActorCritic(1888, [NUM_MACRO_ACTIONS], hidden=256)
        model.load_state_dict(saved["model"], strict=True)
        return model

    def inputs(self, migrated):
        random = np.random.default_rng(8)
        old_raw = self.saved["obs_norm"]["mean"] + random.normal(size=(32, 472)) * np.sqrt(self.saved["obs_norm"]["var"])
        new_raw = random.uniform(-10000, 10000, size=(32, 1888))
        new_raw[:, source_column_indices()] = old_raw
        old_norm, new_norm = RunningNorm(472), RunningNorm(1888)
        old_norm.load_state_dict(self.saved["obs_norm"])
        new_norm.load_state_dict(migrated["obs_norm"])
        return torch.tensor(old_norm.normalize(old_raw)), torch.tensor(new_norm.normalize(new_raw))

    def test_new_distribution_is_exact_old_joint_conditioned_on_canonical_table(self):
        migrated = self.migrate()
        new_model = self.model(migrated)
        old_input, new_input = self.inputs(migrated)
        choices = torch.tensor([a.buttons for a in MACRO_ACTIONS], dtype=torch.long)
        old_batch = old_input.repeat_interleave(35, dim=0)
        old_actions = choices.repeat(len(old_input), 1)
        old_log_probs, _, old_values = self.old_model.evaluate(old_batch, old_actions)
        old_joint = old_log_probs.sum(dim=1).reshape(len(old_input), 35)
        new_logits, new_values = new_model(new_input)
        torch.testing.assert_close(new_logits[0].softmax(-1), old_joint.softmax(-1), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(new_values, old_values.reshape(-1, 35)[:, 0], atol=1e-6, rtol=1e-6)
        self.assertEqual([new_model.greedy_act(row.numpy())[0] for row in new_input],
                         old_joint.argmax(dim=1).tolist())
        # Selecting one table vector is a restricted policy, not a claim that
        # arbitrary old greedy button combinations remain possible.
        self.assertFalse(migrated["macro_migration"]["unrestricted_source_policy_preserved"])

    def test_backbone_critic_and_recent_input_columns_are_preserved_extra_columns_zero(self):
        migrated = self.migrate()
        indices = source_column_indices()
        for name, expected in self.saved["model"].items():
            if name.startswith("heads."):
                continue
            actual = migrated["model"][name]
            if name == "trunk.0.weight":
                self.assertTrue(torch.equal(actual[:, indices], expected))
                self.assertEqual(torch.count_nonzero(actual[:, :12 * 118]).item(), 0)
            else:
                self.assertTrue(torch.equal(actual, expected))
        new_model = self.model(migrated)
        _, new_input = self.inputs(migrated)
        logits, values = new_model(new_input)
        new_input[:, :12 * 118] = torch.randn(32, 12 * 118) * 1e6
        changed_logits, changed_values = new_model(new_input)
        torch.testing.assert_close(changed_logits[0], logits[0], atol=0, rtol=0)
        torch.testing.assert_close(changed_values, values, atol=0, rtol=0)

    def test_normalization_copies_oldest_statistics_for_new_history_and_freezes(self):
        migrated = self.migrate()
        for key in ("mean", "var"):
            old, new = self.saved["obs_norm"][key], migrated["obs_norm"][key]
            np.testing.assert_array_equal(new[source_column_indices()], old)
            for frame in range(12):
                np.testing.assert_array_equal(new[frame * 118:(frame + 1) * 118], old[:118])
        for key in ("count", "clip"):
            self.assertEqual(migrated["obs_norm"][key], self.saved["obs_norm"][key])
        self.assertIs(migrated["freeze_obs_norm"], True)

    def test_all_adam_states_restart_and_real_training_step_can_learn_history(self):
        migrated = self.migrate()
        self.assertTrue(self.saved["optimizer"]["state"])
        self.assertEqual(migrated["optimizer"]["state"], {})
        for key, value in self.saved["optimizer"]["param_groups"][0].items():
            if key != "params":
                self.assertEqual(migrated["optimizer"]["param_groups"][0][key], value)
        model = self.model(migrated)
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(migrated["optimizer"])
        inputs = torch.randn(16, 1888)
        actions = torch.arange(16).reshape(-1, 1)
        log_probs, entropy, value = model.evaluate(inputs, actions)
        loss = -log_probs.mean() - .005 * entropy.mean() + .5 * value.square().mean()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(torch.count_nonzero(model.trunk[0].weight[:, :12 * 118]).item(), 0)
        self.assertTrue(all(state["step"].item() == 1 for state in optimizer.state.values()))
        self.assertEqual(model.act(inputs[0].numpy())[0].__len__(), 1)

    def test_metadata_counters_source_hash_rng_and_exclusive_destination(self):
        original = self.source.read_bytes()
        rng = torch.get_rng_state().clone()
        migrated = self.migrate()
        self.assertEqual(self.source.read_bytes(), original)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(validate_macro_checkpoint(migrated))
        self.assertEqual(migrated["observation_size"], 1888)
        self.assertEqual(migrated["observation_schema"], "legacy_v1")
        self.assertEqual(migrated["action_distribution"], MACRO_DISTRIBUTION)
        self.assertEqual(migrated["execution_profile"], "legacy_v1")
        for key in ("update", "global_step", "episode"):
            self.assertEqual(migrated[key], self.saved[key])
        self.assertEqual(migrated["training_config"]["start_update"], 551)
        self.assertEqual(migrated["training_config"]["reward_config"], self.saved["training_config"]["reward_config"])
        for key, value in macro_metadata().items():
            self.assertEqual(migrated["training_config"][key], value)
            self.assertEqual(migrated["inference_config"][key], value)
        self.assertNotIn("evaluation", migrated)
        provenance = migrated["macro_migration"]
        self.assertEqual(provenance["source_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(provenance["previous_evaluation"], self.saved["evaluation"])
        self.assertTrue(provenance["optimizer_reset"])
        self.assertFalse(provenance["critic_reset"])
        self.assertFalse(provenance["reward_changed"])
        self.assertEqual(torch.load(self.destination, weights_only=False)["format_version"], 10)
        target_bytes = self.destination.read_bytes()
        with self.assertRaises(FileExistsError):
            self.migrate()
        self.assertEqual(self.destination.read_bytes(), target_bytes)
        with self.assertRaisesRegex(ValueError, "source must remain unchanged"):
            migrate_checkpoint(self.source, self.source)

    def test_execution_profile_is_explicitly_selectable_without_changing_weights(self):
        results = []
        for profile in ("legacy_v1", "repeat_dash_v1", "soul_guard_v1"):
            result = migrate_checkpoint(self.source, self.root / f"{profile}.pth", execution_profile=profile)
            self.assertEqual(result["execution_profile"], profile)
            self.assertEqual(result["training_config"]["execution_profile"], profile)
            self.assertEqual(result["inference_config"]["execution_profile"], profile)
            self.assertEqual(result["macro_migration"]["destination_execution_profile"], profile)
            results.append(result)
        for result in results[1:]:
            for name in results[0]["model"]:
                self.assertTrue(torch.equal(results[0]["model"][name], result["model"][name]))
        with self.assertRaisesRegex(ValueError, "unknown execution profile"):
            migrate_checkpoint(self.source, self.destination, execution_profile="combined")
        self.assertFalse(self.destination.exists())

    def test_incompatible_or_incomplete_sources_are_rejected_without_target(self):
        invalid = []
        for key, value in (("format_version", 10), ("frame_stack", 16),
                           ("optimizer", {}), ("action_sizes", [35]),
                           ("freeze_obs_norm", False), ("nail_damage", 33)):
            source = copy.deepcopy(self.saved)
            source[key] = value
            invalid.append(source)
        source = copy.deepcopy(self.saved)
        source["training_config"]["reward_profile"] = "radiant_spells_v1"
        source["training_config"]["reward_config"] = asdict(get_reward_config("radiant_spells_v1"))
        invalid.append(source)
        source = copy.deepcopy(self.saved)
        source["model"]["heads.0.weight"][0, 0] = float("nan")
        invalid.append(source)
        for number, source in enumerate(invalid):
            with self.subTest(number=number):
                torch.save(source, self.source)
                with self.assertRaises(ValueError):
                    self.migrate()
                self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
