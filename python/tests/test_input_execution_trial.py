import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import input_execution_trial as trial


class InputExecutionTrialTests(unittest.TestCase):
    def test_profiles_use_same_frozen_source_and_endpoint_strata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "python/runs/recordings").mkdir(parents=True)
            (root / "python/runs/traitor_4env_iframes_fixed").mkdir(parents=True)
            source = root / "source.pth"
            source.write_bytes(b"unchanged policy")
            target = root / "trial"
            endpoints = [f"127.0.0.1:{port}" for port in (9001, 9002, 9003, 9004)]
            calls = []

            class Evaluator:
                def __init__(self, directory, checkpoint, digest, entropy, names, stopped,
                             episode_time_limit):
                    self.directory = directory
                    self.assertions = (entropy, names, episode_time_limit)

                def evaluate(self, checkpoint, label, **options):
                    calls.append((checkpoint.read_bytes(), label, options, self.assertions))
                    rows = [{"env": index, "info": {"won": False, "damage_tracking_valid": True,
                             "damage": 9, "min_boss_health": 500 + index,
                             "truncated_reason": None}} for index in range(4) for _ in range(10)]
                    (target / f"{label}_fights.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                    return {"valid": True, "episodes": 40, "label": label}

                def status(self, *args, **kwargs):
                    pass

            saved = {"frame_stack": 4, "observation_size": 472, "update": 550,
                     "training_config": {"reward_profile": "radiant_spells_dodge_v1"}}
            with patch.object(trial, "ROOT", root), \
                    patch.object(trial.torch, "load", return_value=saved), \
                    patch.object(trial, "verified_checkpoint"), \
                    patch.object(trial, "existing_controllers", return_value=[]), \
                    patch.object(trial, "DodgeRewardTrial", Evaluator), \
                    patch.object(trial, "freeze_endpoint", side_effect=lambda name: {"endpoint": name}):
                trial.run(source, target, endpoints)
            self.assertEqual(source.read_bytes(), b"unchanged policy")
            self.assertEqual([row[1] for row in calls], list(trial.PROFILES))
            for payload, label, options, arguments in calls:
                self.assertEqual(payload, b"unchanged policy")
                self.assertEqual(options, {"execution_profile": label,
                                          "reward_profile": "radiant_spells_dodge_v1"})
                self.assertEqual(arguments, (.005, endpoints, 150))
            result = json.loads((target / "comparison.json").read_text())
            self.assertTrue(result["valid"])
            self.assertFalse(result["trained"])
            for summary in result["summaries"]:
                self.assertEqual([group["fights"] for group in summary["by_endpoint"]], [10] * 4)

    def test_existing_controller_check_uses_literal_python_and_script_suffixes(self):
        class Result:
            stdout = '[{"ProcessId": 123,"CommandLine":"python play.py"}]'
        with patch.object(trial.subprocess, "run", return_value=Result()) as run:
            self.assertEqual(trial.existing_controllers()[0]["ProcessId"], 123)
        command = run.call_args.args[0][-1]
        self.assertIn("^python(w)?\\.exe$", command)
        self.assertIn("main_multi\\.py|play\\.py", command)
        self.assertNotIn("\\\\.exe", command)


if __name__ == "__main__":
    unittest.main()
