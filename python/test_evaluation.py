"""Check evaluation scheduling and training episode boundaries without a game."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluation import evaluate_pool, interrupt_episode_stats


class FakeModel:
    def greedy_act(self, observation):
        return (observation,)


class FakePool:
    def __init__(self):
        self.connections = ["local", "remote"]
        self.commands = []
        self.pending = {}
        self.step_counts = [0, 0]
        self.reset_counts = [0, 0]
        self.frozen = set()

    def send(self, index, operation, payload=None):
        self.commands.append((index, operation))
        self.pending[index] = (operation, payload)

    def ready(self, pending):
        # Return the remote worker first to exercise out-of-order replies.
        return list(reversed(list(pending)))

    def recv(self, index):
        operation, payload = self.pending.pop(index)
        if operation == "reset":
            if not any(op == "start" for _, op in self.commands):
                assert (0, "reset") in self.commands and (1, "reset") in self.commands
            self.step_counts[index] = 0
            self.reset_counts[index] += 1
            return index, {}
        if operation == "step":
            assert payload == (index + 10,)
            self.step_counts[index] += 1
            done = self.step_counts[index] == index + 1
            return index, 1.0, done, False, {"hits": self.step_counts[index]}
        if operation == "freeze":
            self.frozen.add(index)
        return {}


class EvaluationTests(unittest.TestCase):
    def test_workers_evaluate_independently_and_finish_frozen(self):
        pool = FakePool()
        callbacks = []
        results = evaluate_pool(pool, FakeModel(), lambda obs: obs + 10, 2, callbacks.append)
        self.assertEqual(len(results), 4)
        self.assertEqual(callbacks, results)
        self.assertEqual(pool.reset_counts, [2, 2])
        self.assertEqual(pool.frozen, {0, 1})
        for result in results:
            self.assertEqual(result["steps"], result["env"] + 1)
            self.assertEqual(result["reward"], result["env"] + 1)
        self.assertEqual(sorted(r["episode_in_eval"] for r in results), [1, 1, 2, 2])

    def test_evaluation_does_not_carry_partial_training_scores(self):
        rewards, steps = [2.5, 0.0], [40, 0]
        actions = [Counter(attack=9), Counter()]
        parts = [Counter({"boss damage": 3, "time": -0.5}), Counter()]
        records = interrupt_episode_stats(rewards, steps, actions, parts)
        self.assertEqual(records, [{"env": 0, "steps": 40, "reward": 2.5,
                                    "reward_parts": {"boss damage": 3, "time": -0.5}}])
        self.assertEqual(rewards, [0.0, 0.0])
        self.assertEqual(steps, [0, 0])
        self.assertTrue(all(not counter for counter in actions + parts))

    def test_training_loop_keeps_complete_scores_and_saves_best_evaluation(self):
        import main_multi

        class TrainingPool:
            def __init__(self, endpoints, *args):
                self.connections = list(range(len(endpoints)))
                self.pending = {}
                self.steps = [0] * len(endpoints)

            def send(self, index, operation, payload=None):
                self.pending[index] = operation

            def ready(self, pending):
                return list(reversed(list(pending)))

            def recv(self, index):
                operation = self.pending.pop(index)
                observation = np.zeros(8, dtype=np.float32)
                info = {"hits": 2, "damage": 0, "misses": 0, "staggers": 0,
                        "min_boss_health": 0, "won": True,
                        "reset_seconds": 0, "reset_attempts": 0,
                        "reward_parts": {"boss damage": 1.0}}
                if operation == "reset":
                    self.steps[index] = 0
                    return observation, info
                if operation == "step":
                    self.steps[index] += 1
                    return observation, 1.0, self.steps[index] == 2, False, info
                return {}

            def close(self):
                pass

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "explained_variance": 0.0, "n_samples": 6}
        with tempfile.TemporaryDirectory() as directory:
            argv = ["main_multi.py", "--ports", "9001", "9002", "--steps-per-env", "3",
                    "--total-updates", "2", "--run-dir", directory,
                    "--eval-every", "1", "--no-monitor"]
            with patch("sys.argv", argv), patch.object(main_multi, "prepare_fights"), \
                 patch.object(main_multi, "ExternalEnvPool", TrainingPool), \
                 patch.object(main_multi, "ppo_update", return_value=stats):
                main_multi.main()
            root = Path(directory)
            records = [json.loads(line) for line in
                       (root / "training_metrics.jsonl").read_text().splitlines()]
            completed = [r for r in records if r["event"] == "episode"]
            interrupted = [r for r in records if r["event"] == "episode_interrupted"]
            evaluations = [r for r in records if r["event"] == "eval_summary"]
            self.assertEqual(len(completed), 4)
            self.assertTrue(all(r["steps"] == 2 and r["reward"] == 2 for r in completed))
            self.assertEqual(len(interrupted), 4)
            self.assertTrue(all(r["steps"] == 1 for r in interrupted))
            self.assertEqual(len(evaluations), 2)
            self.assertEqual(evaluations[0]["episodes"], 2)
            self.assertTrue(evaluations[0]["new_best"])
            self.assertFalse(evaluations[1]["new_best"])
            self.assertTrue((root / "best_eval_checkpoint.pth").exists())
            config = json.loads((root / "run_config.json").read_text())
            self.assertEqual(config["endpoints"], ["127.0.0.1:9001", "127.0.0.1:9002"])


if __name__ == "__main__":
    unittest.main()
