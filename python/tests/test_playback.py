"""Offline checks for unlimited frozen playback and checkpoint settings."""
from contextlib import ExitStack
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import play


class UnlimitedPlaybackTests(unittest.TestCase):
    def run_playback(self, outcomes, *, check_only=False):
        original, digest = play.load_checkpoint(play.DEFAULT_CHECKPOINT)
        saved = dict(original, training_config={"episode_time_limit": 150.0,
                                                "observation_time_limit": 60.0,
                                                "reward_profile": "radiant_spells_dodge_v1"})
        completed = []
        pool = MagicMock()

        def evaluate(_pool, model, normalize, count, callback):
            won, damage = outcomes[len(completed)]
            result = {"steps": 500, "info": {"boss_level": 2, "reset_mode": "scene",
                      "truncated_reason": None, "won": won, "episode_damage": damage,
                      "damage_tracking_valid": True, "min_boss_health": 0 if won else 400}}
            completed.append(result)
            callback(result)
            return [result]

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            stack.enter_context(patch.object(play, "ROOT", Path(temporary)))
            stack.enter_context(patch.object(play, "load_checkpoint", return_value=(saved, digest)))
            stage = stack.enter_context(patch.object(play, "stage_fresh_fight"))
            prepare = stack.enter_context(patch.object(play, "prepare_fight"))
            factory = stack.enter_context(patch.object(play, "ExternalEnvPool", return_value=pool))
            evaluate_mock = stack.enter_context(patch.object(play, "evaluate_pool", side_effect=evaluate))
            stack.enter_context(patch("sys.argv", ["play.py", "--episodes", "0"]
                                      + (["--check-only"] if check_only else [])))
            stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
            play.main()
            if check_only:
                stage.assert_not_called()
                prepare.assert_not_called()
                factory.assert_not_called()
                evaluate_mock.assert_not_called()
                return
            self.assertEqual(factory.call_args.args[3], 150.0)
            self.assertEqual(factory.call_args.kwargs["observation_time_limit"], 60.0)
            self.assertEqual(factory.call_args.args[5], "radiant_spells_dodge_v1")
            pool.close.assert_called_once()
            summary = json.loads(next(Path(temporary).rglob("summary.json")).read_text())
            return completed, summary

    def test_unlimited_retries_only_stop_after_undamaged_victory(self):
        completed, summary = self.run_playback([(False, 9), (True, 1), (False, 9), (True, 0)])
        self.assertEqual(len(completed), 4)
        self.assertEqual(summary["flawless_wins"], 1)
        self.assertEqual(summary["episodes_limit"], 0)
        self.assertTrue(summary["retry_until_flawless"])
        self.assertEqual(summary["episode_time_limit"], 150.0)
        self.assertTrue(summary["checkpoint_unchanged"])
        self.assertTrue(summary["normalizer_unchanged"])

    def test_check_only_with_unlimited_budget_stays_offline(self):
        self.run_playback([], check_only=True)


if __name__ == "__main__":
    unittest.main()
