"""Conservative dense shaping for the uncharmed 21-damage spell policy."""

from types import SimpleNamespace
import unittest

from agent.reward import RewardTracker, calculate_reward, get_reward_config
from .test_reward import make_state


def snapshot(*, gap=8.5, fsm="Idle", hazards=(), **kwargs):
    result = make_state(fsm=fsm, knight_box=(0.0, 0.0, 1.0, 2.0),
                        boss_box=(1.0 + gap, 0.0, 3.0 + gap, 4.0), **kwargs)
    result["hazards"] = list(hazards)
    return result


class SpellRewardTests(unittest.TestCase):
    def setUp(self):
        self.config = get_reward_config("radiant_spells_v1")

    def score(self, previous=None, current=None, **kwargs):
        return calculate_reward(previous or snapshot(), current or snapshot(),
                                done=kwargs.pop("done", False),
                                return_breakdown=True, reward_config=self.config,
                                **kwargs)

    def test_profile_normalizes_actual_damage_and_disables_proxy_bonuses(self):
        self.assertEqual(self.config.nail_damage_reference, 21)
        for field in ("shockwave_spacing_bonus", "shockwave_clear_bonus",
                      "clean_dodge", "empty_attack", "flawless_win_bonus"):
            self.assertEqual(getattr(self.config, field), 0)
        self.assertEqual((self.config.damage_per_mask, self.config.death,
                          self.config.boss_defeated, self.config.time_penalty_per_second),
                         (3, -20, 100, 0.1))
        for damage, expected in ((21, 1), (30, 30 / 21), (63, 3), (80, 80 / 21)):
            with self.subTest(damage=damage):
                _, parts = self.score(snapshot(boss_hp=1300),
                                      snapshot(boss_hp=1300 - damage))
                self.assertAlmostEqual(dict(parts)["boss damage"], expected)

    def test_distance_cost_is_thresholded_linear_and_capped(self):
        for gap, expected in ((0, 0), (5, 0), (8.5, -0.0075),
                              (12, -0.015), (50, -0.015)):
            with self.subTest(gap=gap):
                _, parts = self.score(current=snapshot(gap=gap))
                self.assertAlmostEqual(dict(parts).get("far from boss", 0), expected)
                self.assertLessEqual(sum(value for _, value in parts), 0)

    def test_dangerous_unknown_and_windup_phases_do_not_urge_approach(self):
        for fsm in ("Attack 1", "Waves", "Slam Antic", "unknown"):
            with self.subTest(fsm=fsm):
                _, parts = self.score(current=snapshot(gap=50, fsm=fsm))
                self.assertNotIn("far from boss", dict(parts))
        for fsm in ("Idle", "Attack Recover"):
            with self.subTest(fsm=fsm):
                _, parts = self.score(current=snapshot(gap=50, fsm=fsm))
                self.assertAlmostEqual(dict(parts)["far from boss"], -0.015)

    def test_near_detached_wave_suspends_cost_after_fsm_recovery(self):
        wave = SimpleNamespace(box=(3.0, 0.0, 4.0, 20.0), boss_attached=False)
        for fsm in ("Slam End", "Idle"):
            with self.subTest(fsm=fsm):
                _, parts = self.score(current=snapshot(gap=50, fsm=fsm, hazards=[wave]))
                self.assertNotIn("far from boss", dict(parts))
                self.assertAlmostEqual(dict(parts)["time"], -0.01)

    def test_only_confirmed_attached_colliders_are_ignored(self):
        box = (3.0, 0.0, 4.0, 20.0)
        for marker in (False, None, "true", 1):
            with self.subTest(marker=marker):
                _, parts = self.score(current=snapshot(gap=50, hazards=[
                    {"box": box, "boss_attached": marker}]))
                self.assertNotIn("far from boss", dict(parts))
        _, parts = self.score(current=snapshot(gap=50, hazards=[
            {"box": box, "boss_attached": True}]))
        self.assertAlmostEqual(dict(parts)["far from boss"], -0.015)

    def test_unknown_geometry_or_missing_hazard_feed_never_implies_safe(self):
        cases = [snapshot(hazards=[{"boss_attached": False}]),
                 snapshot(hazards=[{"box": (0, float("nan"), 2, 4)}]),
                 snapshot(hazards=[{"box": (0, 0, -1, 2)}]),
                 snapshot(hazards=[{"box": ("0", "0", "2", "4")}])]
        missing = snapshot()
        del missing["hazards"]
        cases.append(missing)
        for current in cases:
            with self.subTest(current=current):
                _, parts = self.score(current=current)
                self.assertNotIn("far from boss", dict(parts))

    def test_far_detached_hazard_does_not_permanently_disable_cost(self):
        _, parts = self.score(current=snapshot(gap=50, hazards=[
            {"box": (20.0, 0.0, 21.0, 20.0), "boss_attached": False}]))
        self.assertAlmostEqual(dict(parts)["far from boss"], -0.015)

    def test_equal_simulated_time_has_equal_distance_and_time_cost(self):
        for hz in (5, 10, 20):
            total = sum(self.score(current=snapshot(gap=50), step_dt=1 / hz)[0]
                        for _ in range(hz))
            self.assertAlmostEqual(total, -0.25)

    def test_no_reward_for_soul_spending_or_claimed_dodge(self):
        previous, current = snapshot(gap=0), snapshot(gap=0)
        previous["mana"], current["mana"] = 99, 66
        reward, parts = self.score(previous, current, successful_dodge=True,
                                   missed_attack=True, successful_wave_spacing=True,
                                   successful_wave_clear=True)
        self.assertAlmostEqual(reward, -0.01)
        self.assertEqual(set(dict(parts)), {"time"})
        self.assertAlmostEqual(dict(parts)["time"], -0.01)

    def test_radiant_fatal_hit_and_timeout_keep_existing_meaning(self):
        _, parts = self.score(snapshot(), snapshot(knight_hp=0, damage_count=9), done=True)
        self.assertEqual(dict(parts)["took damage"], -27)
        self.assertEqual(dict(parts)["die"], -20)
        _, parts = self.score(snapshot(), snapshot(knight_hp=0), done=True, timed_out=True)
        self.assertNotIn("took damage", dict(parts))
        self.assertNotIn("die", dict(parts))

    def test_dodge_and_miss_diagnostics_survive_without_proxy_reward(self):
        tracker = RewardTracker(self.config)
        tracker.transition(snapshot(fsm="Idle"), snapshot(fsm="Attack 1", attacking=True), False)
        tracker.transition(snapshot(fsm="Attack 1", attacking=True),
                           snapshot(fsm="Attack Recover"), False)
        self.assertEqual(tracker.dodges, 1)
        self.assertEqual(tracker.misses, 1)
        self.assertNotIn("clean dodge", dict(tracker.last_breakdown))
        self.assertNotIn("empty attack", dict(tracker.last_breakdown))

    def test_existing_profiles_retain_original_reference_and_phase_only_cost(self):
        for profile in ("victory_v1", "nohit_v1", "radiant_spacing_v1", "radiant_wave_clear_v1"):
            with self.subTest(profile=profile):
                config = get_reward_config(profile)
                self.assertEqual(config.nail_damage_reference, 33)
                self.assertEqual(config.distance_hazard_exemption, 0)
                state = snapshot(gap=50, hazards=[{"box": (2, 0, 3, 20)}])
                _, parts = calculate_reward(state, state, False, reward_config=config,
                                            return_breakdown=True)
                self.assertAlmostEqual(dict(parts)["far from boss"], -0.005)


if __name__ == "__main__":
    unittest.main()
