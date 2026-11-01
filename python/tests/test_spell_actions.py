"""Spell requests use normal game inputs without changing legacy nail actions."""

import unittest
from unittest.mock import call, patch

import gymnasium as gym
import numpy as np

import game_state
from agent.action import (
    BUTTON_ORDER, NUM_BUTTONS, SPELL_ACTION_NAMES, execute_buttons, requested_spell,
)

with patch.object(game_state, "start"):
    from hollow_knight_env import HollowKnightEnv


def buttons(spell=0, **held):
    return tuple(held.get(name, 0) for name in BUTTON_ORDER) + (spell,)


class SpellActionTests(unittest.TestCase):
    def test_legacy_schema_and_repeated_nail_requests_stay_unchanged(self):
        self.assertEqual(NUM_BUTTONS, 7)
        self.assertEqual(SPELL_ACTION_NAMES, BUTTON_ORDER + ("spell",))
        action = (1, 0, 0, 0, 1, 1, 1)
        with patch("agent.action.controls") as controls:
            previous = execute_buttons(action)
            returned = execute_buttons(action, previous)
        self.assertEqual(returned, list(action))
        self.assertEqual(controls.tap.call_args_list, [call("attack"), call("dash"), call("attack")])
        self.assertEqual(controls.key_down.call_args_list, [call("left"), call("jump")])

    def test_each_spell_explicitly_selects_its_vertical_aim(self):
        for spell, expected in ((1, (0, 0)), (2, (1, 0)), (3, (0, 1))):
            with self.subTest(spell=spell), patch("agent.action.controls") as controls:
                previous = execute_buttons(buttons(spell, up=1))
                self.assertEqual(tuple(previous[2:4]), expected)
                controls.tap.assert_called_once_with("quick_cast")
                self.assertNotIn(call("cast"), controls.tap.call_args_list)

    def test_spell_aim_overrides_previous_hold_and_zero_restores_normal_aim(self):
        with patch("agent.action.controls") as controls:
            previous = execute_buttons(buttons(0, up=1))
            controls.reset_mock()
            previous = execute_buttons(buttons(1, up=1), previous)
            self.assertEqual(controls.mock_calls, [call.key_up("up"), call.tap("quick_cast")])
            controls.reset_mock()
            previous = execute_buttons(buttons(1, up=1), previous)
            self.assertEqual(controls.mock_calls, [call.tap("quick_cast")])
            self.assertEqual(previous[-1], 1)
            controls.reset_mock()
            previous = execute_buttons(buttons(0, up=1), previous)
            self.assertEqual(controls.mock_calls, [call.key_down("up")])

    def test_repeated_same_spell_sends_a_request_on_every_nonzero_decision(self):
        with patch("agent.action.controls") as controls:
            previous = execute_buttons(buttons(2))
            previous = execute_buttons(buttons(2), previous)
            previous = execute_buttons(buttons(0), previous)
            execute_buttons(buttons(2), previous)
            self.assertEqual(controls.tap.call_args_list,
                             [call("quick_cast"), call("quick_cast"), call("quick_cast")])

    def test_changing_spell_requests_a_new_cast_without_zero(self):
        with patch("agent.action.controls") as controls:
            previous = execute_buttons(buttons(2))
            previous = execute_buttons(buttons(3), previous)
            self.assertEqual(previous[2:4], [0, 1])
            self.assertEqual(controls.tap.call_args_list, [call("quick_cast"), call("quick_cast")])
            self.assertEqual(requested_spell(buttons(3), previous), 3)

    def test_invalid_action_is_rejected_before_sending_any_input(self):
        invalid = [buttons(spell) for spell in (-1, 4, 0.5, "1")]
        invalid += [buttons(1, up=1, down=1), buttons(1, left=2), (0,) * 9]
        for action in invalid:
            with self.subTest(action=action), patch("agent.action.controls") as controls:
                with self.assertRaises(ValueError):
                    execute_buttons(action)
                self.assertEqual(controls.mock_calls, [])

    def test_cast_request_is_not_hidden_when_no_soul_or_during_cooldown(self):
        # Eligibility belongs to the game. Every new categorical request is
        # issued; no SOUL mask silently changes the PPO action that was sampled.
        with patch("agent.action.controls") as controls:
            execute_buttons(buttons(3, attack=1, dash=1))
        self.assertEqual(controls.tap.call_args_list,
                         [call("attack"), call("dash"), call("quick_cast")])


class SpellEnvironmentTests(unittest.TestCase):
    def test_action_space_changes_only_for_explicit_spell_distribution(self):
        legacy = HollowKnightEnv()
        spells = HollowKnightEnv(action_distribution="exclusive_directions_spells_v1")
        self.assertIsInstance(legacy.action_space, gym.spaces.MultiBinary)
        self.assertIsInstance(spells.action_space, gym.spaces.MultiDiscrete)
        np.testing.assert_array_equal(spells.action_space.nvec, [2] * 7 + [4])
        self.assertEqual(legacy.observation_space.shape, spells.observation_space.shape)
        self.assertEqual(spells.observation_space.shape, (472,))
        self.assertTrue(spells.action_space.contains(np.array(buttons(3))))
        self.assertFalse(spells.action_space.contains(np.array(buttons(4))))

    def test_unknown_distribution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown action distribution"):
            HollowKnightEnv(action_distribution="unknown")

    def test_spell_control_rate_leaves_release_gap_between_quick_cast_taps(self):
        with self.assertRaisesRegex(ValueError, "control_hz <= 10"):
            HollowKnightEnv(control_hz=15, action_distribution="exclusive_directions_spells_v1")
        self.assertEqual(HollowKnightEnv(control_hz=15).control_hz, 15)

    def test_legacy_environment_rejects_eight_actions_before_game_access(self):
        env = HollowKnightEnv()
        env.state = {}
        with patch("hollow_knight_env.controls") as controls:
            with self.assertRaisesRegex(ValueError, "expected 7 actions"):
                env.step(buttons(1))
            self.assertEqual(controls.mock_calls, [])

    def test_episode_info_counts_real_requests_and_observed_net_soul_loss(self):
        env = HollowKnightEnv(action_distribution="exclusive_directions_spells_v1")
        initial = {"boss_control_state": "Idle", "boss_health": 1300,
                   "knight_health": 9, "mana": 99, "attacking": False,
                   "hit_count": 0, "damage_count": 0}
        env.state = initial
        env.min_boss_hp = 1300
        raw = np.zeros(118, dtype=np.float32)
        env.stacker.reset(raw)
        states = [(dict(initial, mana=mana), raw) for mana in (66, 66, 99, 66)]
        with patch.object(env, "_wait_while_paused", return_value=False), \
                patch.object(env, "_read", side_effect=states), \
                patch("hollow_knight_env.is_done", return_value=False), \
                patch("hollow_knight_env.time.sleep"), \
                patch("agent.action.controls"), patch("hollow_knight_env.controls"):
            for spell in (1, 1, 0, 1):
                _, _, _, _, info = env.step(buttons(spell))
        self.assertEqual(info["spell_requests"], {
            "shade_soul": 3, "abyss_shriek": 0, "descending_dark": 0,
        })
        self.assertEqual(info["soul_spent"], 66.0)


if __name__ == "__main__":
    unittest.main()
