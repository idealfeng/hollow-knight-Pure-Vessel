import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from combat_trace import CombatTrace, trace_state


def state(**changes):
    data = {"knight_health": 9, "boss_health": 1300, "damage_count": 0,
            "boss_control_state": "Waves", "knight_box": (0, 0, 1, 2),
            "boss_box": (4, 0, 6, 4), "can_shade_cloak_now": True,
            "hazards": [SimpleNamespace(instance_id=i, box=(i, 0, i + 1, 2),
                                        velocity=(-3, 0), boss_attached=False) for i in range(10)]}
    data.update(changes)
    return data


class CombatTraceTests(unittest.TestCase):
    def test_hazard_ids_and_all_hazards_survive_serialization(self):
        serialized = json.loads(json.dumps(trace_state(state())))
        self.assertEqual([h["instance_id"] for h in serialized["hazards"]], list(range(10)))
        self.assertEqual(serialized["boss_move"], "SHOCKWAVE")
        self.assertEqual(serialized["hazards"][0]["velocity"], [-3, 0])

    def test_disabled_configuration_creates_no_trace_writer(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(CombatTrace.from_environment(0.1))

    def test_terminal_is_flushed_and_button_hold_is_not_a_new_dash(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = CombatTrace(directory, host="127.0.0.1", port=9001, step_dt=0.1)
            trace.begin(state(), 1, boss_level=2, reward_profile="radiant_spacing_v1")
            trace.transition(state(), state(), [0] * 6 + [1], None, step=1, reward=0,
                             reward_parts=[], terminated=False, truncated=False,
                             won=False, truncated_reason=None)
            trace.transition(state(), state(knight_health=0, damage_count=9), [0] * 6 + [1],
                             [0] * 6 + [1], step=2, reward=-47, reward_parts=[("die", -20)],
                             terminated=True, truncated=False, won=False, truncated_reason=None)
            self.assertIsNone(trace.stream)
            rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(rows[1]["dash_pressed"])
            self.assertFalse(rows[2]["dash_pressed"])
            self.assertTrue(rows[2]["terminated"])
            self.assertEqual(rows[2]["after"]["damage_count"], 9)

    def test_reset_closes_previous_partial_fight_with_distinct_files(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = CombatTrace(directory, host="127.0.0.1", port=9001, step_dt=0.1)
            trace.begin(state(), 1, boss_level=2, reward_profile="radiant_spacing_v1")
            first = trace.path
            trace.begin(state(), 2, boss_level=2, reward_profile="radiant_spacing_v1")
            trace.close()
            self.assertEqual(len(list(Path(directory).glob("*.jsonl"))), 2)
            rows = [json.loads(line) for line in first.read_text().splitlines()]
            self.assertEqual(rows[-1], {"event": "interrupted", "reason": "reset_before_terminal"})


if __name__ == "__main__":
    unittest.main()
