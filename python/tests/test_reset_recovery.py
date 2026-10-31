"""Regression checks for missed deaths and Godhome reset navigation."""

from dataclasses import replace
import importlib
import unittest
from unittest.mock import patch

import numpy as np
import game_state

with patch.object(game_state, "start"):
    from hollow_knight_env import HollowKnightEnv
    from build_human_bc_dataset import frame_to_state_dict
    reset_module = importlib.import_module("enviroment.reset")


def live_frame():
    return game_state._parse_frame(
        "0,0,1,2;0;;10,20;0,0,1,800,1,0,0,9,0,1,3,1,10,10;"
        "0,0,0,0,0,0,1,0,0,1;Idle;2,0,4,3"
    )


class ResetRecoveryTests(unittest.TestCase):
    def test_death_survives_missing_collider_and_restored_hp(self):
        alive = live_frame()
        with patch.object(game_state, "_episode_tracking", False), \
             patch.object(game_state, "episode_death_frame", None):
            game_state._publish(alive)
            game_state.reset_episode_counters()
            dead = replace(alive, knight_health=0, knight_box=None, damage_count=29)
            game_state._publish(dead)
            game_state._publish(replace(alive, boss_health=-1, damage_count=29))
            self.assertIs(game_state.episode_death_frame, dead)
            env = HollowKnightEnv()
            env.state = frame_to_state_dict(alive, 10, 20)
            terminal, vector = env._read(steps=1, retries=1)
            self.assertEqual(terminal["knight_health"], 0)
            self.assertEqual(terminal["damage_count"], 9)
            self.assertIsNotNone(terminal["knight_box"])
            self.assertEqual(vector.shape, (118,))
            game_state.reset_episode_counters()
            self.assertIsNone(game_state.episode_death_frame)

    def test_missing_boss_ends_episode_and_releases_keys(self):
        env = HollowKnightEnv()
        env.state = frame_to_state_dict(live_frame(), 10, 20)
        env.raw_observation = np.zeros(118, dtype=np.float32)
        env.stacker.reset(env.raw_observation)
        current = dict(env.state, boss_health=-1, boss_facing_left=None)
        with patch.object(env, "_read", return_value=(current, env.raw_observation)), \
             patch("hollow_knight_env.execute_buttons", return_value=[0] * 7), \
             patch("hollow_knight_env.time.sleep"), \
             patch("hollow_knight_env.controls.release_all") as release:
            _, _, terminated, truncated, info = env.step([0] * 7)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["truncated_reason"], "fight_lost")
        self.assertFalse(info["won"])
        release.assert_called_once()

    def test_statue_reset_uses_observed_position_instead_of_blind_sleep(self):
        with patch.multiple(game_state, knight_box=(187.7, 4.5, 188.7, 6.5),
                            knight_health=9, boss_health=-1, boss_control_state="",
                            boss_box=None, game_speed=1.0), \
             patch.object(reset_module, "wait_until", side_effect=lambda predicate, **kw: predicate()), \
             patch.object(reset_module, "state_is_stable", return_value=True), \
             patch.object(reset_module.time, "sleep") as sleep, \
             patch.object(game_state, "reset_episode_counters"), \
             patch.object(reset_module.controls, "release_all"), \
             patch.object(reset_module.controls, "tap") as tap:
            def enter(action, hold_ms):
                if action == "submit":
                    game_state.boss_health = 800
                    game_state.boss_control_state = "Idle"
                    game_state.boss_box = (2, 0, 4, 3)
            tap.side_effect = enter
            result = reset_module.reset_environment(return_details=True)
            self.assertTrue(result["ok"])
            self.assertEqual(result["attempts"], 1)
            self.assertEqual([call.args[0] for call in tap.call_args_list], ["up", "submit"])
            self.assertTrue(all(call.kwargs["hold_ms"] == 120 for call in tap.call_args_list))
            sleep.assert_called_once_with(1.2)


if __name__ == "__main__":
    unittest.main()
