"""Compare execution profiles through a fake command sink, without a game."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from agent.action import BUTTON_ORDER, describe_execution, execute_buttons


def buttons(spell=0, **selected):
    return [selected.get(name, 0) for name in BUTTON_ORDER] + [spell]


class FakeControls:
    def __init__(self):
        self.commands = []

    def key_down(self, name):
        self.commands.append(f"KD {name}")

    def key_up(self, name):
        self.commands.append(f"KU {name}")

    def tap(self, name, hold_ms=60):
        self.commands.append(f"TAP {name} {hold_ms}")


class ExecutionProfileTests(unittest.TestCase):
    def test_default_and_explicit_legacy_keep_existing_command_sequence(self):
        expected = ["KD left", "KD up", "TAP attack 60", "KD jump", "TAP dash 60",
                    "TAP attack 60", "KU up", "TAP attack 60", "TAP quick_cast 60",
                    "KD up", "TAP attack 60", "KU left", "KU up", "KU jump"]
        sequence = [buttons(left=1, up=1, attack=1, jump=1, dash=1),
                    buttons(left=1, up=1, attack=1, jump=1, dash=1),
                    buttons(1, left=1, up=1, attack=1, jump=1, dash=1),
                    buttons(left=1, up=1, attack=1, jump=1, dash=1), buttons()]
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                fake, previous = FakeControls(), None
                with patch("agent.action.controls", fake):
                    for action in sequence:
                        kwargs = {"execution_profile": "legacy_v1"} if explicit else {}
                        previous = execute_buttons(action, previous, **kwargs)
                self.assertEqual(fake.commands, expected)

    def test_repeated_dash_decisions_send_one_legacy_tap_and_three_candidate_taps(self):
        for profile, count in (("legacy_v1", 1), ("repeat_dash_v1", 3), ("soul_guard_v1", 1)):
            with self.subTest(profile=profile):
                fake, previous = FakeControls(), None
                with patch("agent.action.controls", fake):
                    for _ in range(3):
                        previous = execute_buttons(buttons(dash=1), previous,
                                                   execution_profile=profile, mana=99)
                self.assertEqual(fake.commands, ["TAP dash 60"] * count)

    def test_repeat_dash_retains_jump_hold_attack_taps_and_dash_release_then_repress(self):
        fake, previous = FakeControls(), None
        with patch("agent.action.controls", fake):
            for action in (buttons(attack=1, jump=1, dash=1),
                           buttons(attack=1, jump=1, dash=1), buttons(), buttons(dash=1)):
                previous = execute_buttons(action, previous, execution_profile="repeat_dash_v1")
        self.assertEqual(fake.commands, ["TAP attack 60", "KD jump", "TAP dash 60",
                                         "TAP attack 60", "TAP dash 60", "KU jump", "TAP dash 60"])

    def test_legacy_and_repeat_still_send_low_soul_spells_and_override_nail_aim(self):
        for profile in ("legacy_v1", "repeat_dash_v1"):
            with self.subTest(profile=profile):
                fake = FakeControls()
                with patch("agent.action.controls", fake):
                    effective = execute_buttons(buttons(3, up=1, attack=1),
                                                execution_profile=profile, mana=0)
                self.assertEqual(effective[2:4], [0, 1])
                self.assertEqual(effective[7], 3)
                self.assertEqual(fake.commands, ["KD down", "TAP attack 60", "TAP quick_cast 60"])

    def test_low_soul_guard_suppresses_entire_cast_without_overriding_aim_or_other_actions(self):
        fake = FakeControls()
        with patch("agent.action.controls", fake):
            effective = execute_buttons(buttons(3, up=1, attack=1, jump=1, dash=1),
                                        execution_profile="soul_guard_v1", mana=32.999)
        self.assertEqual(effective, buttons(up=1, attack=1, jump=1, dash=1))
        self.assertEqual(fake.commands, ["KD up", "TAP attack 60", "KD jump", "TAP dash 60"])
        self.assertFalse(any("quick_cast" in command for command in fake.commands))

    def test_guard_at_33_casts_then_low_soul_restores_normal_aim_then_retries(self):
        fake = FakeControls()
        requested = buttons(3, up=1)
        with patch("agent.action.controls", fake):
            previous = execute_buttons(requested, execution_profile="soul_guard_v1", mana=33)
            self.assertEqual(previous[2:4], [0, 1])
            previous = execute_buttons(requested, previous, execution_profile="soul_guard_v1", mana=32)
            self.assertEqual(previous[2:4], [1, 0])
            self.assertEqual(previous[7], 0)
            previous = execute_buttons(requested, previous, execution_profile="soul_guard_v1", mana=33)
        self.assertEqual(fake.commands, ["KD down", "TAP quick_cast 60", "KD up", "KU down",
                                         "KU up", "KD down", "TAP quick_cast 60"])
        self.assertEqual(previous[7], 3)

    def test_guard_requires_known_valid_mana_before_sending_any_command(self):
        for mana in (None, float("nan"), float("inf"), -1, "unknown"):
            with self.subTest(mana=mana):
                fake = FakeControls()
                with patch("agent.action.controls", fake), self.assertRaisesRegex(ValueError, "mana"):
                    execute_buttons(buttons(2, left=1, attack=1),
                                    execution_profile="soul_guard_v1", mana=mana)
                self.assertEqual(fake.commands, [])

    def test_descriptions_are_pure_json_and_match_dispatch_gates(self):
        fake = FakeControls()
        action = np.asarray(buttons(3, up=1, dash=1), dtype=np.int64)
        previous = np.asarray(buttons(0, up=1, dash=1), dtype=np.int64)
        original, original_previous = action.copy(), previous.copy()
        for profile, mana, dash_tap, spell_sent, overridden in (
                ("legacy_v1", None, False, 3, True),
                ("repeat_dash_v1", None, True, 3, True),
                ("soul_guard_v1", 0, False, 0, False),
                ("soul_guard_v1", 33, False, 3, True)):
            with self.subTest(profile=profile, mana=mana), patch("agent.action.controls", fake):
                fake.commands.clear()
                plan = describe_execution(action, previous, profile, mana)
                self.assertEqual(fake.commands, [])
                self.assertEqual(json.loads(json.dumps(plan, allow_nan=False)), plan)
                self.assertEqual(plan["spell_requested"], 3)
                self.assertEqual(plan["spell_sent"], spell_sent)
                self.assertEqual(plan["dash_tap"], dash_tap)
                self.assertEqual(plan["aim_overridden"], overridden)
                effective = execute_buttons(action, previous, execution_profile=profile, mana=mana)
                self.assertEqual(effective, plan["effective_buttons"])
                self.assertEqual(fake.commands.count("TAP dash 60"), int(dash_tap))
                self.assertEqual(fake.commands.count("TAP quick_cast 60"), int(bool(spell_sent)))
                if profile == "soul_guard_v1" and mana == 0:
                    self.assertEqual(plan["spell_gate_reason"], "soul_below_33")
                    self.assertEqual(plan["dash_gate_reason"], "held_without_new_edge")
        np.testing.assert_array_equal(action, original)
        np.testing.assert_array_equal(previous, original_previous)

    def test_seven_button_vectors_remain_supported_and_invalid_profiles_send_nothing(self):
        fake = FakeControls()
        with patch("agent.action.controls", fake):
            self.assertEqual(execute_buttons([0] * 6 + [1]), [0] * 6 + [1])
            fake.commands.clear()
            with self.assertRaisesRegex(ValueError, "execution profile"):
                execute_buttons(buttons(1, left=1), execution_profile="unknown")
            self.assertEqual(fake.commands, [])


if __name__ == "__main__":
    unittest.main()
