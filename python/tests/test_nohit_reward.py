"""Reward-objective checks independent of the live games."""

from dataclasses import asdict
import unittest

from agent.reward import RewardTracker, calculate_reward, get_reward_config
from .test_reward import make_state


class NoHitRewardTests(unittest.TestCase):
    def tracker(self):
        return RewardTracker(reward_config=get_reward_config("nohit_v1"))

    def finish(self, tracker, previous=None, current=None, **kwargs):
        previous = previous or make_state(boss_hp=8)
        current = current or make_state(boss_hp=0)
        tracker.transition(previous, current, done=True, **kwargs)
        return dict(tracker.last_breakdown)

    def test_default_profile_is_backward_compatible(self):
        previous = make_state(boss_hp=33, knight_hp=9)
        current = make_state(boss_hp=0, knight_hp=7, damage_count=2)
        legacy = calculate_reward(previous, current, True, return_breakdown=True)
        explicit = calculate_reward(previous, current, True, return_breakdown=True,
                                    reward_config=get_reward_config("victory_v1"), episode_damage=2)
        self.assertEqual(legacy, explicit)
        self.assertEqual(dict(legacy[1])["took damage"], -6)
        self.assertNotIn("flawless victory", dict(legacy[1]))

    def test_profile_is_serializable_and_rejects_unknown_name(self):
        self.assertEqual(asdict(get_reward_config("nohit_v1"))["flawless_win_bonus"], 25)
        with self.assertRaises(ValueError):
            get_reward_config("typo")

    def test_two_masks_cost_twelve_and_other_rewards_stay_equal(self):
        tracker = self.tracker()
        parts = self.finish(tracker, make_state(boss_hp=33),
                            make_state(boss_hp=0, knight_hp=7, damage_count=2))
        self.assertEqual(parts["took damage"], -12)
        self.assertEqual(parts["boss defeated"], 100)
        self.assertEqual(parts["boss damage"], 1)
        self.assertNotIn("flawless victory", parts)
        self.assertEqual(tracker.episode_damage, 2)

    def test_whole_fight_clean_win_adds_twenty_five(self):
        tracker = self.tracker()
        tracker.transition(make_state(boss_hp=800), make_state(boss_hp=8), False)
        parts = self.finish(tracker)
        self.assertEqual(parts["flawless victory"], 25)
        self.assertEqual(parts["boss defeated"], 100)

    def test_clean_final_step_does_not_erase_earlier_damage_or_healing(self):
        tracker = self.tracker()
        tracker.transition(make_state(), make_state(knight_hp=7, damage_count=2), False)
        tracker.transition(make_state(knight_hp=7, damage_count=2),
                           make_state(knight_hp=9, damage_count=2), False)
        parts = self.finish(tracker, make_state(boss_hp=8, damage_count=2),
                            make_state(boss_hp=0, damage_count=2))
        self.assertNotIn("flawless victory", parts)
        self.assertEqual(tracker.episode_damage, 2)

    def test_damage_counter_catches_loss_and_heal_between_steps(self):
        tracker = self.tracker()
        tracker.transition(make_state(), make_state(damage_count=1), False)
        self.assertEqual(tracker.episode_damage, 1)
        self.assertNotIn("flawless victory", self.finish(
            tracker, make_state(boss_hp=8, damage_count=1), make_state(boss_hp=0, damage_count=1)))

    def test_counter_discontinuity_or_preexisting_damage_disqualifies_bonus(self):
        for previous_count, current_count in ((2, 2), (1, 0), (-1, 0)):
            with self.subTest(previous_count=previous_count, current_count=current_count):
                parts = self.finish(self.tracker(), make_state(boss_hp=8, damage_count=previous_count),
                                    make_state(boss_hp=0, damage_count=current_count))
                self.assertNotIn("flawless victory", parts)

    def test_simultaneous_kill_and_damage_disqualifies_bonus(self):
        parts = self.finish(self.tracker(), make_state(boss_hp=8),
                            make_state(boss_hp=0, knight_hp=8, damage_count=1))
        self.assertNotIn("flawless victory", parts)

    def test_timeout_truncation_missing_boss_and_mutual_death_never_get_bonus(self):
        cases = [({}, {"timed_out": True}), ({}, {"truncated": True}),
                 ({"boss_hp": -1}, {}), ({"knight_hp": 0}, {})]
        for state_args, flags in cases:
            with self.subTest(state_args=state_args, flags=flags):
                parts = self.finish(self.tracker(), current=make_state(boss_hp=state_args.pop("boss_hp", 0),
                                                                       **state_args), **flags)
                self.assertNotIn("flawless victory", parts)

    def test_tracker_reset_starts_new_flawless_attempt(self):
        tracker = self.tracker()
        tracker.transition(make_state(), make_state(knight_hp=8, damage_count=1), False)
        tracker.reset()
        self.assertEqual(self.finish(tracker)["flawless victory"], 25)

    def test_final_transition_alone_does_not_claim_whole_fight(self):
        _, parts = calculate_reward(make_state(boss_hp=8), make_state(boss_hp=0), True,
                                    reward_config=get_reward_config("nohit_v1"), return_breakdown=True)
        self.assertNotIn("flawless victory", dict(parts))

    def test_same_terminal_transition_cannot_award_bonus_twice(self):
        tracker = self.tracker()
        self.assertEqual(self.finish(tracker)["flawless victory"], 25)
        self.assertNotIn("flawless victory", self.finish(tracker))


if __name__ == "__main__":
    unittest.main()
