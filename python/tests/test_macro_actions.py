"""The versioned macro table covers combat primitives without conflicts."""

from dataclasses import FrozenInstanceError
import hashlib
import json
import unittest

import numpy as np

from macro_actions import (MACRO_ACTIONS, MACRO_ACTION_HASH, MACRO_ACTION_NAMES,
                           MACRO_DISTRIBUTION, MACRO_SCHEMA, NUM_MACRO_ACTIONS,
                           decode_macro, macro_metadata, validate_macro_checkpoint)


class MacroActionTests(unittest.TestCase):
    def checkpoint(self):
        return {"format_version": 10, "action_distribution": MACRO_DISTRIBUTION,
                "action_names": ["macro"], "action_sizes": [35], **macro_metadata()}

    def test_table_is_immutable_unique_and_has_exactly_35_one_step_choices(self):
        self.assertIsInstance(MACRO_ACTIONS, tuple)
        self.assertEqual(NUM_MACRO_ACTIONS, 35)
        self.assertEqual(len(set(MACRO_ACTION_NAMES)), 35)
        self.assertEqual(len({a.buttons for a in MACRO_ACTIONS}), 35)
        for action in MACRO_ACTIONS:
            self.assertIsInstance(action.buttons, tuple)
            self.assertEqual(len(action.buttons), 8)
            self.assertTrue(all(type(x) is int and x in (0, 1) for x in action.buttons[:7]))
            self.assertIn(action.buttons[7], (0, 1, 2, 3))
        with self.assertRaises(FrozenInstanceError):
            MACRO_ACTIONS[0].name = "changed"

    def test_every_directional_and_jumping_nail_primitive_is_retained(self):
        attacks = [a.buttons for a in MACRO_ACTIONS if a.buttons[4]]
        self.assertEqual(len(attacks), 18)
        for horizontal in ((0, 0), (1, 0), (0, 1)):
            for aim in ((0, 0), (1, 0), (0, 1)):
                for jump in (0, 1):
                    self.assertIn(horizontal + aim + (1, jump, 0, 0), attacks)

    def test_conflicting_requests_are_removed_but_jump_dash_retained(self):
        for a in MACRO_ACTIONS:
            left, right, up, down, attack, jump, dash, spell = a.buttons
            self.assertFalse(left and right)
            self.assertFalse(up and down)
            self.assertFalse(attack and dash)
            self.assertFalse(attack and spell)
            self.assertFalse(dash and spell)
            self.assertFalse(jump and spell)
        self.assertEqual(sum(a.buttons[5] and a.buttons[6] for a in MACRO_ACTIONS), 3)

    def test_all_three_spells_have_canonical_effective_aim(self):
        spells = [a.buttons for a in MACRO_ACTIONS if a.buttons[7]]
        self.assertEqual(len(spells), 5)
        self.assertEqual(sum(a[7] == 1 for a in spells), 3)
        for a in spells:
            self.assertEqual(a[2:4], (int(a[7] == 2), int(a[7] == 3)))

    def test_decode_requires_a_valid_scalar_integer_and_returns_stable_tuple(self):
        for index, macro in enumerate(MACRO_ACTIONS):
            self.assertIs(decode_macro(index), macro.buttons)
            self.assertEqual(decode_macro(np.int64(index)), macro.buttons)
        for invalid in (-1, 35, 99, True, False, 1.0, "1", None, (1,), [1]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                decode_macro(invalid)

    def test_hash_describes_schema_and_entire_ordered_table_and_metadata_is_fresh(self):
        metadata = macro_metadata()
        expected = hashlib.sha256(json.dumps(
            {"schema": MACRO_SCHEMA, "actions": metadata["macro_actions"]},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii")).hexdigest()
        self.assertEqual(MACRO_ACTION_HASH, expected)
        self.assertEqual(metadata["macro_schema_hash"], expected)
        metadata["macro_actions"][0]["buttons"][0] = 1
        metadata["macro_actions"][0]["name"] = "changed"
        self.assertEqual(macro_metadata()["macro_actions"][0],
                         {"name": "idle", "buttons": [0] * 8})

    def test_checkpoint_validation_rejects_schema_table_order_and_head_changes(self):
        self.assertTrue(validate_macro_checkpoint(self.checkpoint()))
        mutations = (("format_version", 9), ("action_distribution", "other"),
                     ("action_names", ["attack"]), ("action_sizes", [34]),
                     ("macro_schema", "other"), ("macro_schema_hash", "0" * 64),
                     ("macro_actions", list(reversed(macro_metadata()["macro_actions"]))))
        for key, value in mutations:
            saved = self.checkpoint()
            saved[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_macro_checkpoint(saved)


if __name__ == "__main__":
    unittest.main()
