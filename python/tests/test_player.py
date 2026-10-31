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


class PlayerTests(unittest.TestCase):
    def run_player(self, outcomes, budget):
        pool = MagicMock()
        completed = []

        def evaluate(_pool, model, normalize, count, record):
            won = outcomes[len(completed)]
            result = {"steps": 500, "info": {
                "boss_level": 2, "reset_mode": "scene", "truncated_reason": None,
                "won": won, "episode_damage": 0 if won else 9,
                "damage_tracking_valid": True, "min_boss_health": 0 if won else 400,
            }}
            completed.append(result)
            record(result)
            return [result]

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            stack.enter_context(patch.object(play, "ROOT", Path(temporary)))
            stage = stack.enter_context(patch.object(play, "stage_fresh_fight"))
            stack.enter_context(patch.object(play, "prepare_fight"))
            factory = stack.enter_context(patch.object(play, "ExternalEnvPool", return_value=pool))
            stack.enter_context(patch.object(play, "evaluate_pool", side_effect=evaluate))
            stack.enter_context(patch("sys.argv", ["play.py", "--episodes", str(budget)]))
            stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
            play.main()
            stage.assert_called_once_with(("127.0.0.1", 9001), 2)
            self.assertEqual(factory.call_args.args[0], [("127.0.0.1", 9001)])
            self.assertEqual(factory.call_args.args[3:5], (120.0, 1.0))
            pool.close.assert_called_once()
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

    def test_check_only_never_connects_to_game(self):
        with patch("sys.argv", ["play.py", "--check-only"]), \
                patch.object(play, "stage_fresh_fight") as stage, \
                patch.object(play, "ExternalEnvPool") as factory, \
                patch("sys.stdout", new_callable=io.StringIO):
            play.main()
        stage.assert_not_called()
        factory.assert_not_called()


class FreshFightTests(unittest.TestCase):
    def stage(self, observed=2):
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
            elif command == "RESET_FIGHT LEVEL 2" and not captured:
                answer = "err reset target not captured"
            elif command == "RESET_FIGHT":
                generation += 1
                answer = f"ok reset_v1 started {generation}"
            elif command == "pos":
                answer = ";;;;0,0,0,1300,0,0,0,9,0,0"
            elif command == "RESET_FIGHT TARGET":
                answer = f"target_v1;GG_Traitor_Lord;2;{observed}"
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
            policy_runtime.stage_fresh_fight(("127.0.0.1", 9001), 2)
        return commands

    def test_new_game_captures_before_selecting_native_tier(self):
        commands = self.stage()
        self.assertLess(commands.index("CAPTURE"), commands.index("RESET_FIGHT LEVEL 2"))
        self.assertIn("RESET_FIGHT TARGET", commands)

    def test_ascended_1300_hp_cannot_pass_as_radiant(self):
        with self.assertRaisesRegex(RuntimeError, "native fight reset timed out"):
            self.stage(observed=1)


if __name__ == "__main__":
    unittest.main()
