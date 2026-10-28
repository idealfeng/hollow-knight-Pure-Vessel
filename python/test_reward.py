"""Unit tests for calculate_reward and RewardTracker (Traitor Lord)."""

import unittest

from agent.reward import (
    BOSS_DEFEATED,
    DAMAGE_PER_MASK,
    DEATH,
    EMPTY_ATTACK,
    NAIL_DAMAGE_REFERENCE,
    RewardTracker,
    calculate_reward,
)


def make_state(fsm="Idle", boss_hp=800, knight_hp=9, attacking=False,
               hit_count=0, damage_count=0, boss_box=None, knight_box=None):
    return {
        "boss_control_state": fsm,
        "boss_health": boss_hp,
        "knight_health": knight_hp,
        "attacking": attacking,
        "hit_count": hit_count,
        "damage_count": damage_count,
        "boss_box": boss_box,
        "knight_box": knight_box,
    }


class CalculateRewardTests(unittest.TestCase):
    def test_hit_reward_is_proportional_to_hp_drop(self):
        reward, parts = calculate_reward(
            make_state(boss_hp=800), make_state(boss_hp=767),
            done=False, return_breakdown=True,
        )
        self.assertAlmostEqual(dict(parts)["boss damage"], 33 / NAIL_DAMAGE_REFERENCE)

    def test_boss_defeated_awarded_only_on_terminal_zero_hp(self):
        reward, parts = calculate_reward(
            make_state(boss_hp=10), make_state(boss_hp=0),
            done=True, return_breakdown=True,
        )
        self.assertEqual(dict(parts)["boss defeated"], BOSS_DEFEATED)

    def test_took_damage_and_death(self):
        reward, parts = calculate_reward(
            make_state(knight_hp=9), make_state(knight_hp=0),
            done=True, return_breakdown=True,
        )
        parts = dict(parts)
        self.assertEqual(parts["took damage"], -9 * DAMAGE_PER_MASK)
        self.assertEqual(parts["die"], DEATH)

    def test_timed_out_does_not_double_count_death(self):
        reward, parts = calculate_reward(
            make_state(knight_hp=9), make_state(knight_hp=0),
            done=True, timed_out=True, return_breakdown=True,
        )
        parts = dict(parts)
        self.assertNotIn("die", parts)
        self.assertNotIn("took damage", parts)

    def test_missing_fsm_state_never_awards_dodge(self):
        tracker = RewardTracker()
        tracker.transition(
            make_state(fsm="", damage_count=0),
            make_state(fsm="", damage_count=0),
            done=False,
        )
        self.assertEqual(tracker.dodges, 0)
        self.assertNotIn("clean dodge", dict(tracker.last_breakdown))


class RewardTrackerTests(unittest.TestCase):
    def test_empty_attack_counted_once_after_swing_ends(self):
        tracker = RewardTracker()
        tracker.transition(
            make_state(attacking=False), make_state(attacking=True), done=False)
        tracker.transition(
            make_state(attacking=True), make_state(attacking=False), done=False)
        self.assertEqual(tracker.misses, 1)
        self.assertEqual(dict(tracker.last_breakdown).get("empty attack"), EMPTY_ATTACK)

    def test_landed_hit_is_not_a_miss(self):
        tracker = RewardTracker()
        tracker.transition(
            make_state(attacking=False, hit_count=0, boss_hp=800),
            make_state(attacking=True, hit_count=0, boss_hp=800),
            done=False)
        tracker.transition(
            make_state(attacking=True, hit_count=0, boss_hp=800),
            make_state(attacking=False, hit_count=1, boss_hp=767),
            done=False)
        self.assertEqual(tracker.misses, 0)

    def test_clean_dodge_awarded_after_undamaged_active_phase(self):
        tracker = RewardTracker()
        tracker.transition(
            make_state(fsm="Idle", damage_count=0),
            make_state(fsm="Attack 1", damage_count=0),
            done=False)
        tracker.transition(
            make_state(fsm="Attack 1", damage_count=0),
            make_state(fsm="Attack Recover", damage_count=0),
            done=False)
        self.assertEqual(tracker.dodges, 1)

    def test_no_dodge_when_damaged_during_active_phase(self):
        tracker = RewardTracker()
        tracker.transition(
            make_state(fsm="Idle", damage_count=0),
            make_state(fsm="Attack 1", damage_count=0),
            done=False)
        tracker.transition(
            make_state(fsm="Attack 1", damage_count=0),
            make_state(fsm="Attack 1", damage_count=1),
            done=False)
        tracker.transition(
            make_state(fsm="Attack 1", damage_count=1),
            make_state(fsm="Attack Recover", damage_count=1),
            done=False)
        self.assertEqual(tracker.dodges, 0)


if __name__ == "__main__":
    unittest.main()
