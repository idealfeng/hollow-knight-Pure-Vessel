"""Changing a fight budget must not silently retime a transferred policy."""

import unittest
from unittest.mock import patch

import numpy as np
import game_state

with patch.object(game_state, "start"):
    from hollow_knight_env import HollowKnightEnv

from test_reward import make_state


class ObservationTimeLimitTests(unittest.TestCase):
    def test_default_observation_scale_tracks_episode_limit(self):
        for limit in (60.0, 120.0):
            with self.subTest(limit=limit):
                env = HollowKnightEnv(episode_time_limit=limit)
                self.assertEqual(env._time_fraction(300), 30.0 / limit)
                self.assertEqual(env._time_fraction(int(limit * 10)), 1.0)

    def test_legacy_scale_is_identical_until_sixty_seconds_then_clamps(self):
        old = HollowKnightEnv(episode_time_limit=60.0)
        extended = HollowKnightEnv(episode_time_limit=120.0, observation_time_limit=60.0)
        for steps in (0, 1, 300, 599, 600, 900, 1200):
            with self.subTest(steps=steps):
                self.assertEqual(extended._time_fraction(steps), old._time_fraction(steps))

    def test_sixty_second_observation_scale_does_not_end_fight_before_120_seconds(self):
        env = HollowKnightEnv(episode_time_limit=120.0, observation_time_limit=60.0)
        state = make_state(boss_hp=1300, knight_hp=9)
        raw = np.zeros(118, dtype=np.float32)
        env.state = state
        env.raw_observation = raw
        env.min_boss_hp = 1300
        env.stacker.reset(raw)
        with patch.object(env, "_read", return_value=(state, raw)), \
                patch.object(env, "_wait_while_paused", return_value=False), \
                patch("hollow_knight_env.execute_buttons", return_value=[0] * 7), \
                patch("hollow_knight_env.time.sleep"), \
                patch("hollow_knight_env.controls.release_all"), \
                patch("hollow_knight_env.controls.kill") as kill:
            for start in (599, 600, 1198):
                env.episode_steps = start
                _, _, terminated, truncated, _ = env.step([0] * 7)
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                kill.assert_not_called()
            env.episode_steps = 1199
            _, _, terminated, truncated, info = env.step([0] * 7)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["truncated_reason"], "timeout")
            kill.assert_called_once_with()

    def test_time_limits_must_be_finite_and_positive(self):
        for value in (0, -1, float("nan"), float("inf")):
            for field in ("episode_time_limit", "observation_time_limit"):
                with self.subTest(value=value, field=field):
                    with self.assertRaises(ValueError):
                        HollowKnightEnv(**{field: value})


if __name__ == "__main__":
    unittest.main()
