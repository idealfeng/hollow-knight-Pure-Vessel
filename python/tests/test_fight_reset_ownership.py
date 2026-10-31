"""A reconnect must recover its own reset lease without stealing another one."""

import unittest
from unittest.mock import call, patch

import fight_reset


class FightResetOwnershipTests(unittest.TestCase):
    def test_level_recovers_missing_socket_owner_and_preserves_requested_tier(self):
        with patch.object(fight_reset.controls, "request", side_effect=[
                "err fast reset is not enabled", "ok reset_v1 enabled",
                "ok reset_v1 level 2"]) as request:
            fight_reset.set_level(2)
        self.assertEqual(request.call_args_list, [
            call("RESET_FIGHT LEVEL 2"), call("RESET_FIGHT ENABLE"),
            call("RESET_FIGHT LEVEL 2")])

    def test_waits_briefly_for_old_disconnected_owner_cleanup(self):
        with patch.object(fight_reset.controls, "request", side_effect=[
                "err fast reset is not enabled",
                "err another trainer owns fast reset", "ok reset_v1 enabled",
                "ok reset_v1 level 2"]) as request, \
                patch.object(fight_reset.time, "sleep") as sleep:
            fight_reset.set_level(2)
        sleep.assert_called_once_with(0.1)
        self.assertEqual(request.call_count, 4)

    def test_other_trainer_keeps_owner_and_recovery_is_bounded(self):
        with patch.object(fight_reset.controls, "request", side_effect=[
                "err fast reset is not enabled"] + [
                "err another trainer owns fast reset"] * 3) as request, \
                patch.object(fight_reset.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "another trainer owns"):
                fight_reset.set_level(2)
        self.assertEqual(request.call_count, 4)
        self.assertFalse(any("DISABLE" in args.args[0] for args in request.call_args_list))
        self.assertEqual(request.call_args_list.count(call("RESET_FIGHT LEVEL 2")), 1)

    def test_repeated_owner_loss_does_not_loop_forever(self):
        with patch.object(fight_reset.controls, "request", side_effect=[
                "err fast reset is not enabled", "ok reset_v1 enabled",
                "err fast reset is not enabled"]) as request:
            with self.assertRaisesRegex(RuntimeError, "not enabled"):
                fight_reset.set_level(2)
        self.assertEqual(request.call_count, 3)

    def test_disabled_config_is_respected(self):
        with patch.object(fight_reset.controls, "request", return_value=
                          "err fast reset is not enabled") as request, \
                patch.object(fight_reset, "enabled_by_config", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "not enabled"):
                fight_reset.set_level(2)
        request.assert_called_once_with("RESET_FIGHT LEVEL 2")

    def test_start_recovers_only_after_unambiguous_rejection(self):
        with patch.object(fight_reset, "status", side_effect=[
                {"state": "idle", "generation": 4},
                {"state": "ready", "generation": 5}]), \
                patch.object(fight_reset.controls, "request", side_effect=[
                "err fast reset is not enabled", "ok reset_v1 enabled",
                "ok reset_v1 5"]) as request:
            result = fight_reset.reset(lambda: True)
        self.assertTrue(result["ok"])
        self.assertEqual(request.call_args_list, [
            call("RESET_FIGHT", retry=False), call("RESET_FIGHT ENABLE"),
            call("RESET_FIGHT", retry=False)])

    def test_missing_start_reply_is_never_replayed(self):
        with patch.object(fight_reset, "status", return_value={
                "state": "idle", "generation": 4}), \
                patch.object(fight_reset.controls, "request", return_value=None) as request:
            result = fight_reset.reset(lambda: True)
        self.assertFalse(result["ok"])
        request.assert_called_once_with("RESET_FIGHT", retry=False)

    def test_other_failures_are_not_misdiagnosed_as_lost_ownership(self):
        for reply in ("err reset is busy", "err reset target not captured",
                      "err unknown reset operation"):
            with self.subTest(reply=reply), \
                    patch.object(fight_reset.controls, "request", return_value=reply) as request:
                with self.assertRaises(RuntimeError):
                    fight_reset.set_level(2)
            request.assert_called_once_with("RESET_FIGHT LEVEL 2")


if __name__ == "__main__":
    unittest.main()
