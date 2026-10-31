"""Protocol compatibility and episode reset boundaries without a live game."""

import unittest
from unittest.mock import patch

import numpy as np
import controls
import fight_reset
import game_state

with patch.object(game_state, "start"):
    from hollow_knight_env import HollowKnightEnv


class FightResetTests(unittest.TestCase):
    def test_old_mod_is_not_mistaken_for_success(self):
        with patch.object(controls, "request", return_value="err unknown command"):
            self.assertFalse(fight_reset.enable())
            self.assertIsNone(fight_reset.reset(lambda: True))

    def test_waits_for_matching_generation_and_fresh_combat_state(self):
        initial = {"state": "ready", "generation": 2}
        current = {"state": "ready", "generation": 3}
        with patch.object(fight_reset, "status", side_effect=[initial, initial, current, current]), \
             patch.object(controls, "request", return_value="ok reset_v1 3"), \
             patch.object(fight_reset.time, "sleep"), \
             patch.object(fight_reset.time, "perf_counter", side_effect=range(20)):
            ready = unittest.mock.Mock(side_effect=[False, True])
            result = fight_reset.reset(ready)
        self.assertTrue(result["ok"])
        self.assertEqual(result["generation"], 3)
        self.assertEqual(ready.call_count, 2)

    def test_supported_failure_is_reported_for_safe_fallback(self):
        with patch.object(fight_reset, "status", side_effect=[
                {"state": "idle", "generation": 0},
                {"state": "failed", "generation": 1, "message": "load failed"}]), \
             patch.object(controls, "request", return_value="ok reset_v1 1"):
            result = fight_reset.reset(lambda: True)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["message"], "load failed")

    def test_scene_reset_skips_kill_and_menu_and_clears_old_terminal(self):
        env = HollowKnightEnv()
        env._has_reset_once = True
        env._fast_reset_checked = True
        env._fast_reset_supported = True
        raw = np.zeros(118, dtype=np.float32)
        dead = {"knight_box": (0, 0, 1, 2), "knight_health": 0, "boss_health": 500}
        alive = dict(dead, knight_health=9, boss_health=800)
        with patch.object(env, "_read", side_effect=[(dead, raw), (alive, raw)]), \
             patch.object(fight_reset, "reset", return_value={"ok": True, "attempts": 1}), \
             patch("hollow_knight_env.controls.kill") as kill, \
             patch("hollow_knight_env.reset_environment") as menu, \
             patch.object(game_state, "reset_episode_counters") as counters:
            observation, info = env.reset()
        kill.assert_not_called()
        menu.assert_not_called()
        counters.assert_called_once()
        self.assertEqual(info["reset_mode"], "scene")
        self.assertEqual(observation.shape, (472,))
        self.assertEqual(env.state["knight_health"], 9)

    def test_command_reply_is_not_truncated_at_64_bytes(self):
        reply = ("reset_v1;failed;1;GG_Traitor_Lord;" + "x" * 100 + "\n").encode()
        sock = unittest.mock.Mock()
        sock.recv.side_effect = [bytes([byte]) for byte in reply]
        with patch.object(controls, "_sock", sock):
            result = controls.request("RESET_FIGHT STATUS")
        self.assertEqual(result, reply.decode().strip())

    def test_failed_scene_reset_disables_guard_and_restarts_live_arena_normally(self):
        env = HollowKnightEnv()
        env._has_reset_once = True
        env._fast_reset_checked = True
        env._fast_reset_supported = True
        raw = np.zeros(118, dtype=np.float32)
        alive = {"knight_box": (0, 0, 1, 2), "knight_health": 9, "boss_health": 800}
        dead = dict(alive, knight_health=0)
        with patch.object(env, "_read", side_effect=[(dead, raw), (alive, raw)]), \
             patch.object(fight_reset, "reset", return_value={"ok": False, "error": "timeout"}), \
             patch.object(fight_reset, "disable") as disable, \
             patch("hollow_knight_env.get_state", return_value=alive), \
             patch("hollow_knight_env.controls.kill") as kill, \
             patch.object(game_state, "knight_health", 0), \
             patch("hollow_knight_env.reset_environment", return_value={"ok": True, "attempts": 1}) as menu:
            _, info = env.reset()
        disable.assert_called_once()
        kill.assert_called_once()
        menu.assert_called_once()
        self.assertFalse(env._fast_reset_supported)
        self.assertEqual(info["reset_mode"], "menu")


if __name__ == "__main__":
    unittest.main()
