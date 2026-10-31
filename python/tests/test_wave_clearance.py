"""Observable-front tests; no live games or calibrated production classifier."""
import unittest

from game_state import Hazard
from wave_clearance import WaveClearanceTracker, WaveFrontCriteria


def front(identifier, left, *, velocity=-20, height=12, attached=False):
    return Hazard(identifier, (left, 0, left + 1, height), (velocity, 0), attached)


def state(fsm="Waves", hazards=(), knight_hp=9, damage=0, knight=(0, 1, 1, 3)):
    return {"boss_control_state": fsm, "boss_health": 1300,
            "boss_box": (7, 0, 10, 4), "knight_box": knight,
            "knight_health": knight_hp, "damage_count": damage,
            "knight_velocity": (0, 0), "hazards": list(hazards)}


class WaveClearanceTests(unittest.TestCase):
    def tracker(self, maximum_rewards=2):
        return WaveClearanceTracker(lambda hazard, snapshot: not hazard.boss_attached,
                                    threat_gap=3, clear_gap=1, maximum_rewards=maximum_rewards)

    def complete(self, tracker, identifier=1):
        a = state("Slam Antic")
        b = state("Slamming", [front(identifier, 4)])
        c = state("Waves", [front(identifier, 2)])
        d = state("Slam End", [front(identifier, -3)])
        e = state("Idle", [front(identifier, -5)])
        self.assertIsNone(tracker.transition(a, b))
        self.assertIsNone(tracker.transition(b, c))
        self.assertIsNone(tracker.transition(c, d))
        return tracker.transition(d, e)

    def test_same_front_approaches_crosses_then_survives_confirmation(self):
        tracker = self.tracker()
        event = self.complete(tracker)
        self.assertEqual(event["event"], "clean_wave_pass")
        self.assertEqual(event["passed_ids"], [1])
        self.assertTrue(event["reward_eligible"])
        self.assertEqual(tracker.reward_count, 1)

    def test_fsm_end_without_actual_front_never_rewards(self):
        tracker = self.tracker()
        snapshots = [state("Slam Antic"), state("Waves"), state("Slam End"), state("Idle")]
        for a, b in zip(snapshots, snapshots[1:]):
            self.assertIsNone(tracker.transition(a, b))
        self.assertEqual(tracker.reward_count, 0)

    def test_disappearing_approaching_front_is_ambiguous_not_success(self):
        tracker = self.tracker()
        a, b, c, d = state("Slam Antic"), state(hazards=[front(1, 2)]), state("Slam End"), state("Idle")
        for x, y in [(a, b), (b, c), (c, d)]:
            self.assertIsNone(tracker.transition(x, y))
        self.assertEqual(tracker.reward_count, 0)

    def test_existing_detached_sickle_cannot_be_claimed_as_new_wave(self):
        tracker = self.tracker()
        a = state("Slam Antic", [front(1, 4)])
        b = state("Waves", [front(1, 2)])
        c = state("Slam End", [front(1, -3)])
        d = state("Idle", [front(1, -5)])
        for x, y in [(a, b), (b, c), (c, d)]:
            self.assertIsNone(tracker.transition(x, y))

    def test_irrelevant_opposite_wave_moving_away_does_not_block(self):
        tracker = self.tracker()
        snapshots = [state("Slam Antic"),
                     state("Slamming", [front(1, 4), front(2, 11, velocity=20)]),
                     state("Waves", [front(1, 2), front(2, 13, velocity=20)]),
                     state("Slam End", [front(1, -3), front(2, 15, velocity=20)]),
                     state("Idle", [front(1, -5), front(2, 17, velocity=20)])]
        events = [tracker.transition(a, b) for a, b in zip(snapshots, snapshots[1:])]
        self.assertEqual(sum(event is not None for event in events), 1)
        self.assertEqual(events[-1]["passed_ids"], [1])

    def test_reward_waits_for_front_past_fsm_end(self):
        tracker = self.tracker()
        snapshots = [state("Slam Antic"), state("Waves", [front(1, 4)]),
                     state("Slam End", [front(1, 2)]), state("Idle", [front(1, -3)]),
                     state("Idle", [front(1, -5)])]
        events = [tracker.transition(a, b) for a, b in zip(snapshots, snapshots[1:])]
        self.assertTrue(all(event is None for event in events[:-1]))
        self.assertEqual(events[-1]["event"], "clean_wave_pass")

    def test_fast_side_crossing_between_samples_still_proves_pass(self):
        tracker = self.tracker()
        a, b = state("Slam Antic"), state(hazards=[front(1, 6)])
        c, d = state("Slam End", [front(1, -3)]), state("Idle", [front(1, -6)])
        self.assertIsNone(tracker.transition(a, b))
        self.assertIsNone(tracker.transition(b, c))
        self.assertEqual(tracker.transition(c, d)["passed_ids"], [1])

    def test_front_on_other_vertical_line_does_not_establish_threat(self):
        tracker = self.tracker()
        a = state("Slam Antic", knight=(0, 20, 1, 22))
        b = state(hazards=[front(1, 2)], knight=(0, 20, 1, 22))
        c = state("Slam End", [front(1, -3)], knight=(0, 20, 1, 22))
        d = state("Idle", [front(1, -5)], knight=(0, 20, 1, 22))
        for x, y in [(a, b), (b, c), (c, d)]:
            self.assertIsNone(tracker.transition(x, y))

    def test_death_timeout_truncation_or_counter_change_disqualifies(self):
        for changes, flags in [({"knight_hp": 0, "damage": 9}, {"done": True}),
                               ({}, {"timed_out": True}), ({}, {"truncated": True}),
                               ({"knight_hp": 8, "damage": 1}, {}),
                               ({"damage": -1}, {})]:
            with self.subTest(changes=changes, flags=flags):
                tracker = self.tracker()
                a, b = state("Slam Antic"), state(hazards=[front(1, 2)])
                c = state("Slam End", [front(1, -3)], **changes)
                d = state("Idle", [front(1, -5)], **changes)
                tracker.transition(a, b)
                self.assertIsNone(tracker.transition(b, c, **flags))
                self.assertIsNone(tracker.transition(c, d, **flags))
                self.assertEqual(tracker.reward_count, 0)

    def test_two_fronts_in_one_cycle_award_once_and_episode_cap_is_bounded(self):
        tracker = self.tracker()
        for identifier in (1, 2, 3):
            event = self.complete(tracker, identifier)
            self.assertEqual(event["reward_eligible"], identifier <= 2)
        self.assertEqual(tracker.completed_passes, 3)
        self.assertEqual(tracker.reward_count, 2)
        tracker.reset()
        self.assertTrue(self.complete(tracker, 4)["reward_eligible"])

    def test_geometry_classifier_requires_tall_fast_new_front_near_boss(self):
        criteria = WaveFrontCriteria(10, 2, 10, 1, 3)
        self.assertTrue(criteria(front(1, 6), state()))
        self.assertFalse(criteria(front(1, 6, height=2), state()))
        self.assertFalse(criteria(front(1, 6, velocity=1), state()))
        self.assertFalse(criteria(front(1, 6, attached=True), state()))
        self.assertFalse(criteria(front(1, -10), state()))

    def test_first_zero_speed_spawn_can_be_classified_after_motion_appears(self):
        tracker = WaveClearanceTracker(WaveFrontCriteria(10, 2, 10, 1, 4),
                                       threat_gap=6, clear_gap=1)
        a = state("Slam Antic")
        b = state("Waves", [front(1, 6, velocity=0)])
        c = state("Waves", [front(1, 4)])
        d = state("Waves", [front(1, 2)])
        e = state("Slam End", [front(1, -3)])
        f = state("Idle", [front(1, -5)])
        events = [tracker.transition(x, y) for x, y in [(a,b),(b,c),(c,d),(d,e),(e,f)]]
        self.assertEqual(events[-1]["passed_ids"], [1])

    def test_far_opposite_front_briefly_closing_then_receding_does_not_block(self):
        tracker = self.tracker()
        snapshots = [state("Slam Antic"),
                     state("Slamming", [front(1, 4), front(2, -12, velocity=20)]),
                     state("Waves", [front(1, 2), front(2, -10, velocity=20)]),
                     state("Slam End", [front(1, -3), front(2, -13, velocity=-20)]),
                     state("Idle", [front(1, -5), front(2, -15, velocity=-20)])]
        events = [tracker.transition(a, b) for a, b in zip(snapshots, snapshots[1:])]
        self.assertEqual(events[-1]["passed_ids"], [1])

    def test_live_front_reentry_cancels_historical_clearance(self):
        tracker = self.tracker()
        snapshots = [state("Slam Antic"), state("Waves", [front(1, 4)]),
                     state("Waves", [front(1, 2)]), state("Waves", [front(1, -3)]),
                     state("Slam End", [front(1, -0.5)]), state("Idle", [front(1, -0.5)])]
        events = [tracker.transition(a, b) for a, b in zip(snapshots, snapshots[1:])]
        self.assertTrue(all(event is None for event in events))
        self.assertEqual(tracker.last_diagnostic["confirmation_steps"], 0)

    def test_stationary_front_does_not_reward_knight_crossing_it(self):
        tracker = self.tracker()
        static = front(1, 6, velocity=0)
        snapshots = [state("Slam Antic", knight=(0,1,1,3)),
                     state("Waves", [static], knight=(0,1,1,3)),
                     state("Waves", [static], knight=(4,1,5,3)),
                     state("Slam End", [static], knight=(9,1,10,3)),
                     state("Idle", [static], knight=(11,1,12,3))]
        events = [tracker.transition(a,b) for a,b in zip(snapshots,snapshots[1:])]
        self.assertTrue(all(event is None for event in events))


if __name__ == "__main__":
    unittest.main()
