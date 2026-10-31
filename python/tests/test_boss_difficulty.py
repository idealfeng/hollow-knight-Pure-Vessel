import unittest
from unittest.mock import patch
import numpy as np
import fight_reset
import game_state

with patch.object(game_state, "start"):
    from hollow_knight_env import HollowKnightEnv


class DifficultyTests(unittest.TestCase):
    def test_level_command_is_exact_and_rejects_old_mod(self):
        with patch("fight_reset.controls.request", return_value="ok reset_v1 level 2") as request:
            fight_reset.set_level(2)
            request.assert_called_once_with("RESET_FIGHT LEVEL 2")
        with patch("fight_reset.controls.request", return_value="err unknown reset operation"):
            with self.assertRaisesRegex(RuntimeError, "tier-switch mod"):
                fight_reset.set_level(2)

    def test_target_distinguishes_ascended_from_radiant(self):
        with patch("fight_reset.controls.request", return_value="target_v1;GG_Traitor_Lord;2;1"):
            self.assertEqual(fight_reset.target()["observed_level"], 1)

    def reset_case(self, observed=2, reset_ok=True):
        env = HollowKnightEnv(boss_level=2)
        env._fast_reset_checked = env._fast_reset_supported = True
        raw = np.zeros(118, dtype=np.float32)
        ordinary = {"knight_box": (0, 0, 1, 2), "knight_health": 9, "boss_health": 800}
        radiant = dict(ordinary, boss_health=1300)
        return env, raw, ordinary, radiant

    def test_initial_ordinary_scene_is_reset_before_sampling(self):
        env, raw, ordinary, radiant = self.reset_case()
        with patch.object(env, "_read", side_effect=[(ordinary, raw), (radiant, raw)]), \
                patch.object(fight_reset, "set_level") as level, \
                patch.object(fight_reset, "reset", return_value={"ok": True, "attempts": 1}) as reset, \
                patch.object(fight_reset, "target", return_value={"target_level": 2, "observed_level": 2}), \
                patch.object(game_state, "reset_episode_counters"):
            _, info = env.reset(options={"reuse_live_fight": True})
        level.assert_called_once_with(2)
        reset.assert_called_once()
        self.assertEqual(info["boss_level"], 2)
        self.assertEqual(env.state["boss_health"], 1300)

    def test_1300_hp_alone_does_not_qualify_as_radiant(self):
        env, raw, ordinary, radiant = self.reset_case()
        with patch.object(env, "_read", side_effect=[(ordinary, raw), (radiant, raw)]), \
                patch.object(fight_reset, "set_level"), \
                patch.object(fight_reset, "reset", return_value={"ok": True, "attempts": 1}), \
                patch.object(fight_reset, "target", return_value={"target_level": 2, "observed_level": 1}), \
                patch("hollow_knight_env.time.monotonic", side_effect=[0, 3]), \
                patch.object(game_state, "reset_episode_counters"):
            with self.assertRaisesRegex(RuntimeError, "difficulty was not applied"):
                env.reset(options={"reuse_live_fight": True})

    def test_failed_tier_reset_cannot_fall_back_to_ordinary_menu(self):
        env, raw, ordinary, _ = self.reset_case()
        with patch.object(env, "_read", return_value=(ordinary, raw)), \
                patch.object(fight_reset, "set_level"), \
                patch.object(fight_reset, "reset", return_value={"ok": False}), \
                patch("hollow_knight_env.reset_environment") as menu:
            with self.assertRaisesRegex(RuntimeError, "menu fallback disabled"):
                env.reset(options={"reuse_live_fight": True})
        menu.assert_not_called()


if __name__ == "__main__":
    unittest.main()
