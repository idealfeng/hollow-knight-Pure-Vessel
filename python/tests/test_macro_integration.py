"""Offline macro environment, trainer, and frozen-player integration checks."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import game_state
with patch.object(game_state, "start"):
    import hollow_knight_env as environment
import main_multi
import play
from agent.action import describe_execution
from macro_actions import (MACRO_ACTIONS, MACRO_ACTION_NAMES, MACRO_DISTRIBUTION,
                           macro_metadata, decode_macro)
from obs_norm import RunningNorm
from ppo_model import ActorCritic
from .test_reward import make_state


class MacroEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = environment.HollowKnightEnv(
            action_distribution=MACRO_DISTRIBUTION, frame_stack=16,
            reward_profile="radiant_spells_dodge_v1", execution_profile="repeat_dash_v1")
        self.state = make_state(boss_hp=1300, knight_hp=9)
        self.state["mana"] = 99
        self.raw = np.zeros(118, dtype=np.float32)
        self.env.state = self.state
        self.env.raw_observation = self.raw
        self.env.min_boss_hp = 1300
        self.env.stacker.reset(self.raw)

    def run_steps(self, actions):
        dispatched = []
        def execute(buttons, previous, **kwargs):
            plan = describe_execution(buttons, previous, **kwargs)
            dispatched.append(plan)
            return plan["effective_buttons"]
        with patch.object(self.env, "_read", return_value=(self.state, self.raw)), \
                patch.object(self.env, "_wait_while_paused", return_value=False), \
                patch.object(environment, "execute_buttons", side_effect=execute), \
                patch.object(environment.time, "sleep"):
            result = [self.env.step(action) for action in actions]
        return result, dispatched

    def test_discrete_macro_decodes_buttons_and_repeats_dash_without_hidden_movement(self):
        index = next(i for i, action in enumerate(MACRO_ACTIONS) if action.buttons[6])
        results, plans = self.run_steps([index, (index,)])
        self.assertEqual(self.env.action_space.n, len(MACRO_ACTIONS))
        self.assertEqual(results[-1][0].shape, (1888,))
        self.assertTrue(all(plan["dash_tap"] for plan in plans))
        info = results[-1][-1]
        self.assertEqual(info["macro_id"], index)
        self.assertEqual(info["macro_name"], MACRO_ACTION_NAMES[index])
        self.assertEqual(info["decoded_buttons"], list(decode_macro(index)))
        self.assertEqual(info["macro_counts"][MACRO_ACTION_NAMES[index]], 2)
        self.assertEqual(info["execution_counts"]["dash_taps"], 2)

    def test_spell_request_and_soul_guard_plan_remain_distinct(self):
        self.env.execution_profile = "soul_guard_v1"
        self.state["mana"] = 0
        index = next(i for i, action in enumerate(MACRO_ACTIONS) if action.buttons[7] == 3)
        results, plans = self.run_steps([(index,)])
        self.assertEqual(plans[0]["spell_requested"], 3)
        self.assertEqual(plans[0]["spell_sent"], 0)
        info = results[0][-1]
        self.assertEqual(info["spell_requests"]["descending_dark"], 1)
        self.assertEqual(info["execution_counts"]["spell_soul_suppressed"], 1)

    def test_invalid_macro_index_fails_before_any_input_is_sent(self):
        for value in (-1, len(MACRO_ACTIONS), (0, 1), 1.0, True):
            with self.subTest(value=value), patch.object(environment, "execute_buttons") as execute:
                with self.assertRaises((ValueError, IndexError)):
                    self.env.step(value)
                execute.assert_not_called()


class MacroTrainerTests(unittest.TestCase):
    def test_macro_training_rejects_old_33_damage_reward_before_game_access(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sys, "argv", ["main_multi.py", "--action-distribution", MACRO_DISTRIBUTION,
                                          "--frame-stack", "16", "--reward-profile", "victory_v1",
                                          "--run-dir", directory, "--no-monitor"]), \
                patch.object(main_multi, "prepare_fights") as prepare:
            with self.assertRaisesRegex(SystemExit, "21-damage reward reference"):
                main_multi.main()
            prepare.assert_not_called()

    def initializer(self, path):
        torch.set_num_threads(2)
        model = ActorCritic(1888, [len(MACRO_ACTIONS)], hidden=256)
        saved = {"format_version": 10, "observation_size": 1888,
                 "observation_schema": "legacy_v1", "frame_stack": 16,
                 "control_hz": 10, "action_names": ["macro"],
                 "action_sizes": [len(MACRO_ACTIONS)], "model_hidden": 256,
                 "action_distribution": MACRO_DISTRIBUTION,
                 "execution_profile": "repeat_dash_v1", **macro_metadata(),
                 "model": model.state_dict(), "obs_norm": RunningNorm(1888).state_dict(),
                 "initialization_only": True, "freeze_obs_norm": True,
                 "update": -1, "global_step": 0, "episode": 0,
                 "training_config": {"episode_time_limit": 150, "observation_time_limit": 60,
                                     "reward_profile": "radiant_spells_dodge_v1"}}
        torch.save(saved, path)
        return saved

    def test_macro_training_evaluation_and_check_only_save_the_same_schema(self):
        actions, settings = [], []
        class Pool:
            def __init__(self, endpoints, *args, **kwargs):
                self.connections = [0]
                self.pending = {}
                self.steps = 0
                settings.append((args, kwargs))
            def send(self, index, operation, payload=None):
                self.pending[index] = operation
                if operation == "step":
                    actions.append(payload)
            def ready(self, pending):
                return list(pending)
            def recv(self, index):
                operation = self.pending.pop(index)
                info = {"hits": 2, "damage": 0, "episode_damage": 0,
                        "damage_tracking_valid": True, "misses": 0, "staggers": 0,
                        "min_boss_health": 0, "won": True, "reset_seconds": 0,
                        "reset_attempts": 0, "reset_mode": "scene", "boss_level": 2,
                        "reward_parts": {"boss damage": 1}}
                observation = np.zeros(1888, dtype=np.float32)
                if operation == "reset":
                    self.steps = 0
                    return observation, info
                if operation == "step":
                    self.steps += 1
                    return observation, 1, self.steps == 2, False, info
                return {}
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "initializer.pth"
            self.initializer(source)
            run = root / "run"
            argv = ["main_multi.py", "--ports", "9001", "--run-dir", str(run),
                    "--init-checkpoint", str(source), "--reward-profile", "radiant_spells_dodge_v1",
                    "--boss-level", "2", "--steps-per-env", "3", "--total-updates", "1",
                    "--eval-every", "1", "--no-monitor"]
            stats = {"policy_loss": 0, "value_loss": 0, "entropy": 0,
                     "approx_kl": 0, "explained_variance": 0, "n_samples": 3}
            with patch("sys.argv", argv), patch.object(main_multi, "prepare_fights"), \
                    patch.object(main_multi, "ExternalEnvPool", Pool), \
                    patch.object(main_multi, "ppo_update", return_value=stats), \
                    contextlib.redirect_stdout(io.StringIO()):
                main_multi.main()
            saved = torch.load(run / "ppo_checkpoint.pth", map_location="cpu", weights_only=False)
            self.assertEqual(saved["format_version"], 10)
            self.assertEqual(saved["action_sizes"], [len(MACRO_ACTIONS)])
            self.assertEqual(saved["macro_schema_hash"], macro_metadata()["macro_schema_hash"])
            self.assertEqual(saved["execution_profile"], "repeat_dash_v1")
            self.assertIn("optimizer", saved)
            self.assertEqual(settings[0][1]["execution_profile"], "repeat_dash_v1")
            self.assertTrue(actions and all(len(action) == 1 for action in actions))
            records = [json.loads(line) for line in (run / "training_metrics.jsonl").read_text().splitlines()]
            names = {name for row in records if row["event"] == "episode" for name in row["actions"]}
            self.assertTrue(names and names <= set(MACRO_ACTION_NAMES))
            with patch("sys.argv", ["play.py", "--checkpoint", str(run / "best_eval_checkpoint.pth"), "--check-only"]), \
                    patch.object(play, "stage_fresh_fight") as stage, \
                    patch.object(play, "ExternalEnvPool") as connect, \
                    contextlib.redirect_stdout(io.StringIO()):
                play.main()
            stage.assert_not_called()
            connect.assert_not_called()

    def test_tampered_macro_hash_fails_before_game_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bad.pth"
            saved = self.initializer(source)
            saved["macro_schema_hash"] = "changed"
            torch.save(saved, source)
            with patch("sys.argv", ["main_multi.py", "--run-dir", str(Path(temporary) / "run"),
                                      "--init-checkpoint", str(source), "--no-monitor"]), \
                    patch.object(main_multi, "prepare_fights") as stage:
                with self.assertRaises(ValueError):
                    main_multi.main()
            stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
