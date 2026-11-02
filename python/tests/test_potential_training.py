"""Exercise the new schema through the actual trainer save/evaluation boundary."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

import main_multi
from agent.action import SPELL_ACTION_NAMES
from obs_norm import RunningNorm
from ppo_model import ActorCritic


class PotentialTrainingTests(unittest.TestCase):
    def test_new_input_schema_survives_checkpoint_and_fixed_evaluation(self):
        schemas = []

        class Pool:
            def __init__(self, endpoints, *args, **kwargs):
                schemas.append(kwargs.get("observation_schema"))
                self.connections, self.pending, self.steps = [0], {}, 0

            def send(self, index, operation, payload=None):
                self.pending[index] = operation

            def ready(self, pending):
                return list(pending)

            def recv(self, index):
                operation = self.pending.pop(index)
                info = dict(hits=2, damage=0, episode_damage=0, damage_tracking_valid=True,
                            misses=0, staggers=0, min_boss_health=0, won=True,
                            reset_seconds=0, reset_attempts=0, boss_level=2,
                            reward_parts={"potential shaping": .01})
                observation = np.zeros(512, dtype=np.float32)
                if operation == "reset":
                    self.steps = 0
                    return observation, info
                if operation == "step":
                    self.steps += 1
                    return observation, 1., self.steps == 2, False, info
                return {}

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, run = root / "source.pth", root / "run"
            model = ActorCritic(512, [1] * 7 + [4], hidden=256,
                                bernoulli=True, exclusive_directions=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
            torch.save(dict(format_version=9, observation_schema="combat_v2", observation_size=512,
                            frame_stack=4, action_names=list(SPELL_ACTION_NAMES), action_sizes=[1] * 7 + [4],
                            model_hidden=256, action_distribution=main_multi.SPELL_DISTRIBUTION,
                            model=model.state_dict(), optimizer=optimizer.state_dict(),
                            obs_norm=RunningNorm(512).state_dict(), freeze_obs_norm=True,
                            update=550, global_step=1128448, episode=3200), source)
            arguments = ["main_multi.py", "--ports", "9001", "--run-dir", str(run),
                         "--init-checkpoint", str(source), "--reward-profile", "radiant_potential_v1",
                         "--boss-level", "2", "--episode-time-limit", "150", "--observation-time-limit", "60",
                         "--steps-per-env", "3", "--total-updates", "552", "--eval-every", "1", "--no-monitor"]
            stats = dict(policy_loss=0., value_loss=0., entropy=0., approx_kl=0., explained_variance=0., n_samples=3)
            with patch("sys.argv", arguments), patch.object(main_multi, "prepare_fights"), \
                    patch.object(main_multi, "ExternalEnvPool", Pool), \
                    patch.object(main_multi, "ppo_update", return_value=stats), \
                    contextlib.redirect_stdout(io.StringIO()):
                main_multi.main()
            saved = torch.load(run / "ppo_checkpoint.pth", map_location="cpu", weights_only=False)
            self.assertEqual(schemas, ["combat_v2"])
            self.assertEqual((saved["observation_schema"], saved["observation_size"], saved["update"]),
                             ("combat_v2", 512, 551))
            config = json.loads((run / "run_config.json").read_text())
            self.assertEqual(config["reward_config"]["potential"]["gamma"], config["gamma"])
            self.assertEqual(config["reward_config"]["clean_dodge"], .1)
            self.assertEqual(config["reward_config"]["empty_attack"], -.005)
            self.assertTrue((run / "best_eval_checkpoint.pth").exists())


if __name__ == "__main__":
    unittest.main()
