"""Geometry and discounted-return checks; no live game or policy access."""

from dataclasses import asdict, replace
from types import SimpleNamespace
import unittest

from combat_potential import PotentialConfig, potential, potential_reward, swept_aabb_interval


def state(*, knight=(0, 0, 1, 2), boss=(4, 0, 6, 4), velocity=(0, 0),
          boss_velocity=(0, 0), hazards=(), phase="Idle", **extra):
    return {"knight_box": knight, "boss_box": boss, "knight_velocity": velocity,
            "boss_velocity": boss_velocity, "hazards": list(hazards),
            "knight_health": 9, "boss_health": 1300,
            "boss_control_state": phase, "dash_state_valid": True,
            "invulnerability_time_valid": False, **extra}


class SweptAabbTests(unittest.TestCase):
    def test_approach_predicts_both_entry_and_exit(self):
        self.assertEqual(swept_aabb_interval((0, 0, 1, 2), (2, 0, 3, 2), (-2, 0)), (.5, 1.5))

    def test_parallel_vertical_separation_and_moving_away_are_safe(self):
        self.assertIsNone(swept_aabb_interval((0, 0, 1, 2), (2, 3, 3, 5), (-2, 0)))
        self.assertIsNone(swept_aabb_interval((0, 0, 1, 2), (2, 0, 3, 2), (2, 0)))

    def test_stationary_overlap_and_diagonal_nonoverlapping_time_windows(self):
        self.assertEqual(swept_aabb_interval((0, 0, 1, 2), (.5, 0, 1.5, 2), (0, 0)), (0, float("inf")))
        self.assertIsNone(swept_aabb_interval((0, 0, 1, 1), (2, 5, 3, 6), (-2, -1)))

    def test_invalid_data_never_counts_as_no_threat(self):
        for box in (None, (0, 0, float("nan"), 2), (0, 0, 0, 2)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                swept_aabb_interval((0, 0, 1, 2), box, (0, 0))


class CombatPotentialTests(unittest.TestCase):
    def setUp(self):
        self.config = PotentialConfig(range_weight=0, protection_time_margin=0)

    def test_no_threat_is_saturated_zero_and_closer_collision_is_worse(self):
        safe = potential(state(), self.config)
        distant = potential(state(boss=(2, 0, 3, 2), boss_velocity=(-4, 0)), self.config)
        close = potential(state(boss=(1.4, 0, 2.4, 2), boss_velocity=(-4, 0)), self.config)
        self.assertEqual(safe.value, 0)
        self.assertAlmostEqual(distant.earliest_unprotected_ttc, .25)
        self.assertGreater(distant.value, close.value)

    def test_full_safe_crossing_can_be_covered_by_confirmed_protection(self):
        snapshot = state(boss=(1.2, 0, 1.4, 2), velocity=(10, 0),
                         is_shadow_dashing=True, shadow_dash_remaining=.2,
                         hazards=[{"box": (1.2, 0, 1.4, 2), "velocity": (0, 0),
                                   "boss_attached": True, "hazard_type": 1}])
        self.assertEqual(potential(snapshot, self.config).value, 0)

    def test_still_overlapping_when_protection_ends_remains_a_threat(self):
        snapshot = state(boss=(.5, 0, 1.5, 2), is_shadow_dashing=True,
                         shadow_dash_remaining=.2,
                         hazards=[{"box": (.5, 0, 1.5, 2), "velocity": (0, 0),
                                   "boss_attached": True, "hazard_type": 1}])
        result = potential(snapshot, self.config)
        self.assertAlmostEqual(result.earliest_unprotected_ttc, .2)
        self.assertAlmostEqual(result.value, -.6)

    def test_cooldown_readiness_does_not_grant_protection(self):
        snapshot = state(boss=(.5, 0, 1.5, 2), can_shade_cloak_now=True,
                         shadow_dash_remaining=.5, invulnerability_time_valid=True)
        self.assertEqual(potential(snapshot, self.config).value, -1)

    def test_unknown_protection_duration_does_not_hide_collision(self):
        snapshot = state(boss=(.5, 0, 1.5, 2), is_invuln=True,
                         invulnerability_remaining=.5, invulnerability_time_valid=False)
        self.assertEqual(potential(snapshot, self.config).value, -1)

    def test_generic_confirmed_invulnerability_uses_remaining_contact_window(self):
        snapshot = state(boss=(1.2, 0, 1.4, 2), velocity=(10, 0), is_invuln=True,
                         invulnerability_remaining=.2, invulnerability_time_valid=True)
        self.assertEqual(potential(snapshot, self.config).value, 0)

    def test_shadow_protection_never_applies_to_unknown_or_non_enemy_hazard_types(self):
        for hazard_type in (None, 2, 3):
            with self.subTest(hazard_type=hazard_type):
                snapshot = state(boss=(.5, 0, 1.5, 2), is_shadow_dashing=True,
                                 shadow_dash_remaining=.2,
                                 hazards=[{"box": (.5, 0, 1.5, 2), "velocity": (0, 0),
                                           "boss_attached": True, "hazard_type": hazard_type}])
                self.assertEqual(potential(snapshot, self.config).value, -1)

    def test_dedicated_shadow_protection_does_not_require_generic_invulnerability(self):
        snapshot = state(boss=(1.2, 0, 1.4, 2), velocity=(10, 0), is_invuln=False,
                         invulnerability_time_valid=False, is_shadow_dashing=True,
                         shadow_dash_remaining=.2,
                         hazards=[SimpleNamespace(box=(1.2, 0, 1.4, 2), velocity=(0, 0),
                                                  boss_attached=True, hazard_type=1)])
        self.assertEqual(potential(snapshot, self.config).value, 0)

    def test_hazard_velocity_is_relative_and_nearest_threat_wins(self):
        near = {"box": (2, 0, 3, 2), "velocity": (-4, 0)}
        far = SimpleNamespace(box=(5, 0, 6, 2), velocity=(-2, 0))
        snapshot = state(boss=(20, 0, 22, 4), velocity=(1, 0), hazards=[far, near])
        result = potential(snapshot, self.config)
        self.assertAlmostEqual(result.earliest_unprotected_ttc, .2)
        self.assertAlmostEqual(result.value, -.6)

    def test_body_is_included_without_detached_hazards_and_duplicates_do_not_multiply_risk(self):
        body = (.5, 0, 1.5, 2)
        no_hazards = potential(state(boss=body), self.config)
        duplicates = potential(state(boss=body, hazards=[{"box": body, "velocity": (0, 0)}] * 3), self.config)
        self.assertEqual(no_hazards.value, -1)
        self.assertEqual(no_hazards.value, duplicates.value)

    def test_range_uses_interval_and_vertical_attack_gate(self):
        config = replace(self.config, range_weight=1, range_min_gap=.75, range_max_gap=2.5)
        ideal = potential(state(boss=(2.5, 0, 4.5, 4)), config)
        too_far = potential(state(boss=(7, 0, 9, 4)), config)
        too_close = potential(state(boss=(1.2, 0, 3.2, 4)), config)
        above = potential(state(knight=(0, 8, 1, 10), boss=(7, 0, 9, 4)), config)
        self.assertEqual(ideal.value, 0)
        self.assertLess(too_far.value, ideal.value)
        self.assertLess(too_close.value, ideal.value)
        self.assertEqual(above.value, 0)

    def test_default_attack_interval_does_not_make_body_contact_an_ideal_position(self):
        config = PotentialConfig()
        ideal = potential(state(boss=(3.5, 0, 5.5, 4)), config)
        contact = potential(state(boss=(1, 0, 3, 4)), config)
        self.assertEqual(ideal.range_value, 0)
        self.assertLess(contact.range_value, 0)
        self.assertLess(contact.value, ideal.value)

    def test_range_does_not_force_approach_during_imminent_collision(self):
        config = replace(self.config, range_weight=1)
        snapshot = state(boss=(20, 0, 22, 4), hazards=[{"box": (.5, 0, 1.5, 2), "velocity": (0, 0)}])
        result = potential(snapshot, config)
        self.assertEqual(result.value, -1)

    def test_missing_telemetry_and_nonfinite_hazard_are_not_safe(self):
        broken = [state(dash_state_valid=False),
                  state(hazards=[{"box": (1, 0, 2, 3), "velocity": (float("nan"), 0)}]),
                  state(boss=None), state(knight_health=0)]
        for snapshot in broken:
            with self.subTest(snapshot=snapshot):
                self.assertFalse(potential(snapshot, self.config).valid)
                self.assertIsNone(potential_reward(state(), snapshot, config=self.config))

    def test_unknown_fsm_keeps_geometry_risk_without_attack_range_guidance(self):
        config = replace(self.config, range_weight=1, attack_phase_range_multiplier=.5)
        safe_unknown = potential(state(boss=(20, 0, 22, 4), phase="Unmapped Move"), config)
        self.assertTrue(safe_unknown.valid)
        self.assertEqual(safe_unknown.value, 0)
        dangerous_unknown = potential(state(boss=(.5, 0, 1.5, 2), phase="Unmapped Move"), config)
        self.assertTrue(dangerous_unknown.valid)
        self.assertEqual(dangerous_unknown.value, -1)
        self.assertAlmostEqual(potential_reward(state(), state(phase="Unmapped Move"), config=config),
                               config.eta * (config.gamma * potential(state(phase="Unmapped Move"), config).value
                                             - potential(state(), config).value))

    def test_default_physics_margin_does_not_claim_countdown_covers_exact_contact_exit(self):
        snapshot = state(boss=(1.2, 0, 1.4, 2), velocity=(10, 0),
                         is_shadow_dashing=True, shadow_dash_remaining=.15,
                         hazards=[{"box": (1.2, 0, 1.4, 2), "velocity": (0, 0),
                                   "boss_attached": True, "hazard_type": 1}])
        self.assertEqual(potential(snapshot, self.config).value, 0)
        conservative = potential(snapshot, PotentialConfig(range_weight=0))
        self.assertLess(conservative.value, 0)
        self.assertAlmostEqual(conservative.earliest_unprotected_ttc, .13)

    def test_danger_disappearing_keeps_transition_bonus_and_reentry_keeps_cost(self):
        threat = state(boss=(.5, 0, 1.5, 2))
        safe = state()
        self.assertAlmostEqual(potential_reward(threat, safe, config=self.config), .1)
        self.assertAlmostEqual(potential_reward(safe, threat, config=self.config), -.09955)

    def test_phase_weight_is_inside_complete_potential_difference(self):
        config = replace(self.config, range_weight=1)
        a = state(boss=(8, 0, 10, 4), phase="Idle")
        b = state(boss=(8, 0, 10, 4), phase="Attack Antic")
        expected = config.eta * (config.gamma * potential(b, config).value - potential(a, config).value)
        self.assertEqual(potential_reward(a, b, config=config), expected)
        self.assertGreater(expected, 0)

    def test_terminal_boundary_is_zero_even_when_final_geometry_disappears(self):
        threat = state(boss=(.5, 0, 1.5, 2))
        terminal = {"boss_health": 0}
        self.assertAlmostEqual(potential_reward(threat, terminal, terminal=True, config=self.config), .1)
        self.assertIsNone(potential_reward(threat, terminal, terminal=False, config=self.config))

    def test_discounted_full_trajectory_telescopes_including_terminal(self):
        config = replace(self.config, range_weight=.25)
        snapshots = [state(boss=(8, 0, 10, 4)), state(boss=(4, 0, 6, 4)),
                     state(boss=(.5, 0, 1.5, 2)), state(boss=(2.5, 0, 4.5, 4)), {}]
        scores = [potential_reward(a, b, terminal=i == len(snapshots) - 2, config=config)
                  for i, (a, b) in enumerate(zip(snapshots, snapshots[1:]))]
        total = sum(config.gamma ** i * reward for i, reward in enumerate(scores))
        self.assertAlmostEqual(total, -config.eta * potential(snapshots[0], config).value)

    def test_configuration_is_checkpoint_serializable_and_rejects_bad_scales(self):
        self.assertEqual(PotentialConfig(**asdict(self.config)), self.config)
        for kwargs in ({"ttc_horizon": 0}, {"gamma": 0}, {"eta": -1}, {"range_max_gap": .1},
                       {"range_weight": float("inf")}, {"protection_time_margin": -.01}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PotentialConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
