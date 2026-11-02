"""Offline protocol checks for authoritative dash telemetry and hazard types."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import game_state


BASE = ("0,0,1,2;0;;10,20;0,0,1,1300,1,0,0,9,0,1,3,1,10,10;"
        "0,0,0,0,0,0,1,0,0,1;Idle;2,0,4,3")
HAZARD = ";42,2,0,4,3,-5,0,1"
DASH = ";dash_v1,1,0,1,1.2,0.08,0.08,0,1"


class DashTelemetryTests(unittest.TestCase):
    def test_legacy_feed_does_not_advertise_unknown_dash_timers_as_valid(self):
        frame = game_state._parse_frame(BASE + HAZARD)
        self.assertFalse(frame.dash_state_valid)
        self.assertFalse(frame.invulnerability_time_valid)
        self.assertIsNone(frame.hazards[0].hazard_type)
        self.assertEqual(frame.hazards[0].box, (2, 0, 4, 3))

    def test_black_dash_is_distinct_from_available_next_dash(self):
        frame = game_state._parse_frame(BASE + HAZARD + DASH + ";hazard_types_v1,42:1")
        self.assertTrue(frame.dash_state_valid)
        self.assertFalse(frame.can_dash_now)
        self.assertTrue(frame.is_shadow_dashing)
        self.assertAlmostEqual(frame.shadow_cooldown_remaining, 1.2)
        self.assertAlmostEqual(frame.shadow_dash_remaining, .08)
        self.assertAlmostEqual(frame.dash_remaining, .08)
        self.assertEqual(frame.hazards[0].hazard_type, 1)

    def test_normal_dash_has_zero_shadow_protection_despite_active_dash_time(self):
        frame = game_state._parse_frame(BASE + ";dash_v1,1,0,0,0.7,0,0.06,0,1")
        self.assertFalse(frame.is_shadow_dashing)
        self.assertEqual(frame.shadow_dash_remaining, 0)
        self.assertAlmostEqual(frame.dash_remaining, .06)

    def test_direct_fsm_invulnerability_can_have_unknown_remaining_duration(self):
        frame = game_state._parse_frame(BASE + ";dash_v1,1,0,0,0,0,0,0,0")
        self.assertTrue(frame.dash_state_valid)
        self.assertFalse(frame.invulnerability_time_valid)
        self.assertEqual(frame.invulnerability_remaining, 0)

    def test_known_timed_invulnerability_is_carried_without_rounding_to_a_flag(self):
        frame = game_state._parse_frame(BASE + ";dash_v1,1,1,0,0,0,0,0.25,1")
        self.assertTrue(frame.invulnerability_time_valid)
        self.assertAlmostEqual(frame.invulnerability_remaining, .25)

    def test_hazard_types_are_matched_by_id_independent_of_section_order(self):
        response = (BASE + ";hazard_types_v1,43:2,42:1" + HAZARD
                    + ";43,10,0,12,3,-3,0,0" + DASH)
        frame = game_state._parse_frame(response)
        self.assertEqual([h.hazard_type for h in frame.hazards], [1, 2])
        self.assertEqual([h.instance_id for h in frame.hazards], [42, 43])

    def test_no_hero_suppresses_a_stale_valid_extension(self):
        response = BASE.replace("0,0,1,2;", ";", 1) + DASH
        frame = game_state._parse_frame(response)
        self.assertFalse(frame.dash_state_valid)
        self.assertFalse(frame.is_shadow_dashing)
        self.assertFalse(frame.invulnerability_time_valid)

    def test_invalid_extension_values_and_duplicates_are_rejected(self):
        for suffix in (DASH + DASH, DASH.replace("1.2", "nan"),
                       DASH.replace("1.2", "inf"), DASH.replace("1.2", "-0.1"),
                       DASH.replace("dash_v1,1", "dash_v1,yes"),
                       ";hazard_types_v1,42:1,42:2"):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                game_state._parse_frame(BASE + suffix)

    def test_unknown_protocol_version_is_not_silently_treated_as_new_dash_schema(self):
        frame = game_state._parse_frame(BASE + ";dash_v2,1,1,0,0" + HAZARD)
        self.assertFalse(frame.dash_state_valid)
        self.assertEqual(len(frame.hazards), 1)

    def test_publish_disconnect_and_missing_hero_do_not_preserve_dash_validity(self):
        tracked = [key for key, value in vars(game_state).items()
                   if isinstance(value, (bool, int, float, str, tuple, list)) or value is None]
        original = {key: getattr(game_state, key) for key in tracked}
        try:
            with patch.object(game_state, "_episode_tracking", False):
                frame = game_state._parse_frame(BASE + DASH)
                game_state._publish(frame)
                self.assertTrue(game_state.dash_state_valid)
                self.assertTrue(game_state.is_shadow_dashing)
                game_state._publish(replace(frame, knight_box=None))
                self.assertFalse(game_state.dash_state_valid)
                game_state._publish(frame)
                game_state._on_disconnect()
                self.assertFalse(game_state.dash_state_valid)
                self.assertFalse(game_state.is_shadow_dashing)
                self.assertEqual(game_state.shadow_dash_remaining, 0)
                self.assertFalse(game_state.invulnerability_time_valid)
        finally:
            for key, value in original.items():
                setattr(game_state, key, value)


if __name__ == "__main__":
    unittest.main()
