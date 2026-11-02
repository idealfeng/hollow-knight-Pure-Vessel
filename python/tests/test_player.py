"""Regression checks for fresh launches and frozen single-game playback."""

from contextlib import ExitStack
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import play
import policy_runtime
import torch


def write_checkpoint(path, episode_time_limit=120.0):
    """Keep playback tests independent of whichever real model is packaged."""
    size, hidden = 472, 8
    model = play.ActorCritic(size, [1] * 7, hidden=hidden,
                             bernoulli=True, exclusive_directions=True)
    saved = {
        "format_version": 8, "observation_size": size, "frame_stack": 4,
        "model_hidden": hidden, "action_distribution": "exclusive_directions_v1",
        "action_names": list(play.BUTTON_ORDER), "action_sizes": [1] * 7,
        "model": model.state_dict(), "obs_norm": play.RunningNorm(size).state_dict(),
        "update": 650, "control_hz": 10,
        "training_config": {"episode_time_limit": episode_time_limit,
                            "observation_time_limit": 60.0,
                            "reward_profile": "radiant_spells_dodge_v1"},
    }
    torch.save(saved, path)


class PlayerTests(unittest.TestCase):
    def run_player(self, outcomes, budget, *, level=2, stop_after="flawless",
                   invalid=None, expect_error=None, close_error=None):
        pool = MagicMock()
        if close_error:
            pool.close.side_effect = close_error
        completed = []

        def evaluate(_pool, model, normalize, count, record):
            outcome = outcomes[len(completed)]
            won, damage = outcome if isinstance(outcome, tuple) else (outcome, 0 if outcome else 9)
            result = {"steps": 500, "info": {
                "boss_level": level, "reset_mode": "scene", "truncated_reason": None,
                "won": won, "episode_damage": damage,
                "damage_tracking_valid": True, "min_boss_health": 0 if won else 400,
            }}
            if invalid and len(completed) == len(outcomes) - 1:
                result["info"].update(invalid)
            completed.append(result)
            record(result)
            return [result]

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            checkpoint = Path(temporary) / "policy.pth"
            write_checkpoint(checkpoint)
            stack.enter_context(patch.object(play, "ROOT", Path(temporary)))
            stage = stack.enter_context(patch.object(play, "stage_fresh_fight"))
            freeze = stack.enter_context(patch.object(play, "freeze_game", return_value={"paused": True}))
            stack.enter_context(patch.object(play, "prepare_fight"))
            factory = stack.enter_context(patch.object(play, "ExternalEnvPool", return_value=pool))
            stack.enter_context(patch.object(play, "evaluate_pool", side_effect=evaluate))
            stack.enter_context(patch("sys.argv", ["play.py", "--checkpoint", str(checkpoint),
                "--episodes", str(budget), "--boss-level", str(level), "--stop-after", stop_after]))
            stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
            if expect_error:
                with self.assertRaisesRegex(RuntimeError, expect_error):
                    play.main()
            else:
                play.main()
            stage.assert_called_once_with(("127.0.0.1", 9001), level)
            self.assertEqual(factory.call_args.args[0], [("127.0.0.1", 9001)])
            self.assertEqual(factory.call_args.args[3:5], (120.0, 1.0))
            self.assertEqual(factory.call_args.args[6], level)
            pool.close.assert_called_once()
            freeze.assert_called_once_with(("127.0.0.1", 9001))
            summary = json.loads(next(Path(temporary).rglob("summary.json")).read_text())
            self.assertTrue(summary["checkpoint_unchanged"])
            self.assertTrue(summary["normalizer_unchanged"])
            return len(completed), summary

    def test_loss_retries_and_flawless_win_stops_remaining_attempts(self):
        completed, summary = self.run_player([False, True], 100)
        self.assertEqual(completed, 2)
        self.assertEqual(summary["flawless_wins"], 1)

    def test_episode_limit_stops_without_requiring_victory(self):
        completed, summary = self.run_player([False], 1)
        self.assertEqual(completed, 1)
        self.assertEqual(summary["flawless_wins"], 0)

    def test_attuned_fixed_budget_continues_after_both_types_of_win(self):
        completed, summary = self.run_player([(True, 0), (True, 2), (False, 9)], 3,
                                             level=0, stop_after="episodes")
        self.assertEqual(completed, 3)
        self.assertEqual(summary["wins"], 2)
        self.assertEqual(summary["flawless_wins"], 1)
        self.assertAlmostEqual(summary["win_rate"], 2 / 3)
        self.assertAlmostEqual(summary["nohit_rate"], 1 / 3)
        self.assertEqual(summary["boss_level"], 0)
        self.assertEqual(summary["action_selection"], "greedy")
        self.assertTrue(summary["budget_completed"])
        self.assertTrue(summary["valid"])

    def test_any_win_mode_stops_at_damaged_attuned_victory(self):
        completed, summary = self.run_player([(False, 9), (True, 2)], 0,
                                             level=0, stop_after="win")
        self.assertEqual(completed, 2)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["flawless_wins"], 0)
        self.assertFalse(summary["retry_until_flawless"])

    def test_wrong_difficulty_and_disconnect_do_not_enter_denominator(self):
        for invalid in ({"boss_level": 1}, {"truncated_reason": "disconnect"},
                        {"truncated_reason": "fight_lost"}, {"damage_tracking_valid": False}):
            with self.subTest(invalid=invalid):
                _, summary = self.run_player([False, True], 2, level=0,
                    stop_after="episodes", invalid=invalid, expect_error="Invalid native ATTUNED")
                self.assertEqual(summary["completed"], 1)
                self.assertEqual(summary["wins"], 0)
                self.assertEqual(summary["win_rate"], 0)
                self.assertFalse(summary["valid"])
                self.assertFalse(summary["budget_completed"])

    def test_fixed_budget_rejects_infinite_retry_before_any_connection(self):
        with patch("sys.argv", ["play.py", "--episodes", "0", "--stop-after", "episodes"]), \
                patch.object(play, "stage_fresh_fight") as stage, \
                patch.object(play, "freeze_game") as freeze, \
                patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            play.main()
        stage.assert_not_called()
        freeze.assert_not_called()

    def test_pool_close_failure_still_pauses_only_owned_game(self):
        _, summary = self.run_player([True], 1, expect_error="Playback cleanup failed",
                                     close_error=RuntimeError("worker close failed"))
        self.assertFalse(summary["valid"])
        self.assertTrue(summary["pause_after_exit"]["paused"])
        self.assertIn("worker close failed", summary["cleanup_errors"][0])

    def test_check_only_never_connects_to_game(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "policy.pth"
            write_checkpoint(checkpoint)
            with patch("sys.argv", ["play.py", "--checkpoint", str(checkpoint), "--check-only"]), \
                    patch.object(play, "stage_fresh_fight") as stage, \
                    patch.object(play, "freeze_game") as freeze, \
                    patch.object(play, "ExternalEnvPool") as factory, \
                    patch("sys.stdout", new_callable=io.StringIO):
                play.main()
        stage.assert_not_called()
        factory.assert_not_called()
        freeze.assert_not_called()


class FreshFightTests(unittest.TestCase):
    def stage(self, observed=2, *, level=2, boss_health=None, mana=0):
        commands = []
        captured = False
        generation = 0
        reply = b""

        def capture(*_):
            nonlocal captured
            commands.append("CAPTURE")
            captured = True

        def write(value):
            nonlocal reply, generation
            command = value.decode().strip()
            commands.append(command)
            answer = "ok"
            if command == "RESET_FIGHT STATUS":
                answer = f"reset_v1;{'ready' if captured else 'uncaptured'};{generation};GG_Traitor_Lord;ok"
            elif command == f"RESET_FIGHT LEVEL {level}" and not captured:
                answer = "err reset target not captured"
            elif command == "RESET_FIGHT":
                generation += 1
                answer = f"ok reset_v1 started {generation}"
            elif command == "pos":
                health = boss_health if boss_health is not None else 800 if level == 0 else 1300
                answer = f";;;;0,0,0,{health},0,0,0,9,{mana},0"
            elif command == "RESET_FIGHT TARGET":
                answer = f"target_v1;GG_Traitor_Lord;{level};{observed}"
            reply = (answer + "\n").encode()

        stream = MagicMock()
        stream.write.side_effect = write
        stream.readline.side_effect = lambda *_: reply
        sock = MagicMock()
        sock.makefile.return_value.__enter__.return_value = stream
        with patch.object(policy_runtime.socket, "create_connection") as connect, \
                patch("main_multi.prepare_fight", side_effect=capture), \
                patch.object(policy_runtime.time, "monotonic", side_effect=[0, 0, 41]), \
                patch.object(policy_runtime.time, "sleep"):
            connect.return_value.__enter__.return_value = sock
            policy_runtime.stage_fresh_fight(("127.0.0.1", 9001), level)
        return commands

    def test_new_game_captures_before_selecting_native_tier(self):
        commands = self.stage()
        self.assertLess(commands.index("CAPTURE"), commands.index("RESET_FIGHT LEVEL 2"))
        self.assertIn("RESET_FIGHT TARGET", commands)

    def test_ascended_1300_hp_cannot_pass_as_radiant(self):
        with self.assertRaisesRegex(RuntimeError, "native fight reset timed out"):
            self.stage(observed=1)

    def test_attuned_selects_and_checks_native_800_hp(self):
        self.assertIn("RESET_FIGHT LEVEL 0", self.stage(observed=0, level=0))
        with self.assertRaisesRegex(RuntimeError, "native fight reset timed out"):
            self.stage(observed=0, level=0, boss_health=1300)

    def test_fresh_fight_requires_zero_soul(self):
        with self.assertRaisesRegex(RuntimeError, "native fight reset timed out"):
            self.stage(mana=33)


class FreezeOwnedGameTests(unittest.TestCase):
    def freeze(self, speeds):
        commands, remaining = [], iter(speeds)
        reply = b""

        def write(value):
            nonlocal reply
            command = value.decode().strip()
            commands.append(command)
            answer = f";;;;0,0,0,800,0,0,0,9,0,{next(remaining)}" if command == "pos" else "ok"
            reply = (answer + "\n").encode()

        stream, sock = MagicMock(), MagicMock()
        stream.write.side_effect = write
        stream.readline.side_effect = lambda *_: reply
        sock.makefile.return_value.__enter__.return_value = stream
        with patch.object(policy_runtime.socket, "create_connection") as connect, \
                patch.object(policy_runtime.time, "sleep"):
            connect.return_value.__enter__.return_value = sock
            result = policy_runtime.freeze_game(("127.0.0.1", 9002))
            connect.assert_called_once_with(("127.0.0.1", 9002), timeout=5.0)
        return commands, result

    def test_cleanup_only_touches_owned_port_and_verifies_actual_pause(self):
        commands, result = self.freeze([1, 0])
        self.assertEqual(commands, ["CLEAR", "SPEED 0", "pos", "SPEED 0", "pos"])
        self.assertEqual(result, {"endpoint": "127.0.0.1:9002", "paused": True,
                                  "inputs_cleared": True})


if __name__ == "__main__":
    unittest.main()
