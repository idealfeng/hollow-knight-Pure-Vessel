"""Bounded shockwave-spacing shaping independent of live games."""
from dataclasses import asdict
import unittest

from agent.reward import RewardTracker, calculate_reward, get_reward_config
from test_reward import make_state


def wave_state(fsm="Waves", gap=4.0, side="right", height=0.0, **kwargs):
    knight_box = (0.0, height, 1.0, height + 2.0)
    boss_box = ((1.0 + gap, 0.0, 3.0 + gap, 4.0) if side == "right"
                else (-2.0 - gap, 0.0, -gap, 4.0))
    return make_state(fsm=fsm, knight_box=knight_box, boss_box=boss_box, **kwargs)


class WaveSpacingTests(unittest.TestCase):
    def tracker(self, profile="radiant_spacing_v1"):
        return RewardTracker(get_reward_config(profile))

    def move(self, tracker, previous, current, **kwargs):
        tracker.transition(previous, current, done=kwargs.pop("done", False), **kwargs)
        return dict(tracker.last_breakdown)

    def clean_wave(self, tracker=None, **kwargs):
        tracker = tracker or self.tracker()
        self.move(tracker, wave_state("Slam Antic", **kwargs), wave_state("Slamming", **kwargs))
        self.move(tracker, wave_state("Slamming", **kwargs), wave_state(**kwargs))
        return self.move(tracker, wave_state(**kwargs), wave_state("Slam End", **kwargs))

    def test_profile_and_existing_objectives(self):
        config = asdict(get_reward_config("radiant_spacing_v1"))
        self.assertEqual(config["shockwave_spacing_bonus"], 0.20)
        self.assertEqual((config["shockwave_min_gap"], config["shockwave_max_gap"]), (3, 6))
        self.assertEqual((config["damage_per_mask"], config["boss_defeated"],
                          config["death"], config["flawless_win_bonus"]), (3, 100, -20, 0))
        for profile in ("victory_v1", "nohit_v1"):
            with self.subTest(profile=profile):
                self.assertNotIn("shockwave spacing", self.clean_wave(self.tracker(profile)))

    def test_reward_occurs_once_only_after_normal_completion(self):
        tracker = self.tracker()
        self.assertNotIn("shockwave spacing", self.move(
            tracker, wave_state("Slam Antic"), wave_state("Slamming")))
        for _ in range(12):
            self.assertNotIn("shockwave spacing", self.move(tracker, wave_state(), wave_state()))
        self.assertEqual(self.move(tracker, wave_state(), wave_state("Slam End"))["shockwave spacing"], 0.20)
        self.assertNotIn("shockwave spacing", self.move(tracker, wave_state("Slam End"), wave_state("Idle")))
        self.assertEqual(tracker.wave_spacing_rewards, 1)

    def test_active_entry_without_windup_is_supported(self):
        tracker = self.tracker()
        self.move(tracker, wave_state("Idle"), wave_state())
        self.assertEqual(self.move(tracker, wave_state(), wave_state("Slam End"))["shockwave spacing"], 0.20)

    def test_gap_is_horizontal_and_bounded_on_either_side(self):
        for gap in (3.0, 4.0, 6.0):
            for side in ("left", "right"):
                with self.subTest(gap=gap, side=side):
                    self.assertEqual(self.clean_wave(gap=gap, side=side)["shockwave spacing"], 0.20)
        for gap in (0.0, 2.9, 6.1, 25.0):
            with self.subTest(gap=gap):
                self.assertNotIn("shockwave spacing", self.clean_wave(gap=gap))

    def test_jumping_high_does_not_turn_close_horizontal_gap_into_reward(self):
        self.assertNotIn("shockwave spacing", self.clean_wave(gap=0.0, height=50.0))

    def test_entry_gap_is_not_replaced_by_later_retreat(self):
        tracker = self.tracker()
        self.move(tracker, wave_state("Slam Antic", gap=0), wave_state(gap=0))
        self.move(tracker, wave_state(gap=0), wave_state(gap=4))
        self.assertNotIn("shockwave spacing", self.move(tracker, wave_state(gap=4), wave_state("Slam End", gap=4)))

    def test_damage_on_entry_disqualifies_later_clean_recovery(self):
        tracker = self.tracker()
        self.move(tracker, wave_state("Slam Antic"), wave_state(knight_hp=8, damage_count=1))
        self.assertNotIn("shockwave spacing", self.move(
            tracker, wave_state(knight_hp=8, damage_count=1), wave_state("Slam End", knight_hp=8, damage_count=1)))

    def test_damage_and_healing_inside_wave_do_not_erase_history(self):
        tracker = self.tracker()
        self.move(tracker, wave_state("Slam Antic"), wave_state())
        self.move(tracker, wave_state(), wave_state(knight_hp=8, damage_count=1))
        self.move(tracker, wave_state(knight_hp=8, damage_count=1), wave_state(damage_count=1))
        self.assertNotIn("shockwave spacing", self.move(
            tracker, wave_state(damage_count=1), wave_state("Slam End", damage_count=1)))

    def test_damage_counter_detects_loss_hidden_by_healing(self):
        tracker = self.tracker()
        self.move(tracker, wave_state("Slam Antic"), wave_state())
        self.assertNotIn("shockwave spacing", self.move(
            tracker, wave_state(), wave_state("Slam End", damage_count=1)))

    def test_last_transition_health_loss_disqualifies_even_without_counter_tick(self):
        tracker = self.tracker()
        self.move(tracker, wave_state("Slam Antic"), wave_state())
        self.assertNotIn("shockwave spacing", self.move(
            tracker, wave_state(), wave_state("Slam End", knight_hp=8)))

    def test_terminal_timeout_truncation_and_death_cancel_reward(self):
        cases = [({"done": True}, {}), ({"timed_out": True}, {}),
                 ({"truncated": True}, {}), ({}, {"knight_hp": 0})]
        for flags, current_args in cases:
            with self.subTest(flags=flags, current=current_args):
                tracker = self.tracker()
                self.move(tracker, wave_state("Slam Antic"), wave_state())
                self.assertNotIn("shockwave spacing", self.move(
                    tracker, wave_state(), wave_state("Slam End", **current_args), **flags))
                self.assertEqual(tracker.wave_spacing_rewards, 0)

    def test_killing_boss_during_wave_does_not_claim_completed_dodge(self):
        tracker = self.tracker()
        self.move(tracker, wave_state("Slam Antic"), wave_state())
        self.assertNotIn("shockwave spacing", self.move(
            tracker, wave_state(), wave_state("Slam End", boss_hp=0), done=True))

    def test_missing_unknown_state_or_counter_reset_cannot_finish_wave(self):
        for current in (wave_state("", boss_hp=-1), wave_state("Unrecognized"),
                        wave_state("Slam End", damage_count=0)):
            tracker = self.tracker()
            start_count = 1 if current["boss_control_state"] == "Slam End" else 0
            self.move(tracker, wave_state("Slam Antic", damage_count=start_count),
                      wave_state(damage_count=start_count))
            self.assertNotIn("shockwave spacing", self.move(tracker, wave_state(damage_count=start_count), current))

    def test_starting_mid_active_and_resetting_mid_active_cannot_claim_wave(self):
        tracker = self.tracker()
        self.assertNotIn("shockwave spacing", self.move(tracker, wave_state(), wave_state("Slam End")))
        self.move(tracker, wave_state("Slam Antic"), wave_state())
        tracker.reset()
        self.assertNotIn("shockwave spacing", self.move(tracker, wave_state(), wave_state("Slam End")))

    def test_separate_waves_can_each_earn_one_bounded_bonus(self):
        tracker = self.tracker()
        self.assertEqual(self.clean_wave(tracker)["shockwave spacing"], 0.20)
        self.assertEqual(self.clean_wave(tracker)["shockwave spacing"], 0.20)
        self.assertEqual(tracker.wave_spacing_rewards, 2)

    def test_stateless_event_flag_does_not_reward_invalid_terminal(self):
        _, parts = calculate_reward(wave_state(), wave_state("Slam End"), True,
                                    reward_config=get_reward_config("radiant_spacing_v1"),
                                    successful_wave_spacing=True, return_breakdown=True)
        self.assertNotIn("shockwave spacing", dict(parts))


if __name__ == "__main__":
    unittest.main()
