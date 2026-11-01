"""Check the eight-action trainer's save/resume boundary without a game."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agent.action import SPELL_ACTION_NAMES
import main_multi
from obs_norm import RunningNorm
from ppo_model import ActorCritic


class SpellTrainerTests(unittest.TestCase):
    def test_initializer_creates_optimizer_and_fixed_evaluation_saves_format9(self):
        actions = []

        class Pool:
            def __init__(self, endpoints, *args):
                self.connections = [0]
                self.pending = {}
                self.steps = 0
                self.schema = args[-1]

            def send(self, index, operation, payload=None):
                self.pending[index] = operation
                if operation == "step":
                    actions.append(payload)

            def ready(self, pending):
                return list(pending)

            def recv(self, index):
                operation = self.pending.pop(index)
                info = {"hits": 2, "damage": 0, "episode_damage": 0,
                        "misses": 0, "staggers": 0, "min_boss_health": 0,
                        "won": True, "reset_seconds": 0, "reset_attempts": 0,
                        "boss_level": 2, "reward_parts": {"boss damage": 1.0}}
                observation = np.zeros(8, dtype=np.float32)
                if operation == "reset":
                    self.steps = 0
                    return observation, info
                if operation == "step":
                    self.steps += 1
                    return observation, 1.0, self.steps == 2, False, info
                return {}

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "initializer.pth"
            normalizer = RunningNorm(8)
            model = ActorCritic(8, [1] * 7 + [4], hidden=256,
                                bernoulli=True, exclusive_directions=True)
            torch.save({
                "format_version": 9, "observation_size": 8, "frame_stack": 4,
                "action_names": list(SPELL_ACTION_NAMES), "action_sizes": [1] * 7 + [4],
                "model_hidden": 256, "action_distribution": main_multi.SPELL_DISTRIBUTION,
                "model": model.state_dict(), "obs_norm": normalizer.state_dict(),
                "initialization_only": True, "freeze_obs_norm": True,
                "update": -1, "global_step": 0, "episode": 0,
            }, source)
            run = root / "run"
            argv = ["main_multi.py", "--ports", "9001", "--run-dir", str(run),
                    "--init-checkpoint", str(source), "--reward-profile", "radiant_spells_v1",
                    "--boss-level", "2", "--steps-per-env", "3", "--total-updates", "1",
                    "--eval-every", "1", "--no-monitor"]
            stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                     "approx_kl": 0.0, "explained_variance": 0.0, "n_samples": 3}
            with patch("sys.argv", argv), patch.object(main_multi, "prepare_fights"), \
                    patch.object(main_multi, "ExternalEnvPool", Pool), \
                    patch.object(main_multi, "ppo_update", return_value=stats), \
                    contextlib.redirect_stdout(io.StringIO()):
                main_multi.main()
            saved = torch.load(run / "ppo_checkpoint.pth", map_location="cpu", weights_only=False)
            best = torch.load(run / "best_eval_checkpoint.pth", map_location="cpu", weights_only=False)
            self.assertEqual(saved["format_version"], 9)
            self.assertEqual(best["action_sizes"], [1] * 7 + [4])
            self.assertEqual(saved["action_names"], list(SPELL_ACTION_NAMES))
            self.assertIn("optimizer", saved)
            self.assertEqual(saved["update"], 0)
            self.assertTrue(actions and all(len(action) == 8 for action in actions))
            config = json.loads((run / "run_config.json").read_text())
            self.assertEqual(config["reward_config"]["nail_damage_reference"], 21)
            self.assertEqual(config["action_distribution"], main_multi.SPELL_DISTRIBUTION)


if __name__ == "__main__":
    unittest.main()
