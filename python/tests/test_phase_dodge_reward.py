"""Offline behavior checks for the experimental animation-completion reward."""

from dataclasses import asdict
import unittest

from agent.reward import RewardTracker, calculate_reward, get_reward_config


def state(fsm="Idle", *, hp=9, damage=0, boss_hp=1300, **extra):
    return {"boss_control_state": fsm, "knight_health": hp,
            "damage_count": damage, "boss_health": boss_hp,
            "hit_count": 0, "attacking": False, **extra}


class PhaseDodgeTests(unittest.TestCase):
    def setUp(self):
        self.config = get_reward_config("radiant_spells_dodge_v1")
        self.tracker = RewardTracker(self.config)

    def step(self, previous, current, **flags):
        _, parts = self.tracker.transition(previous, current,
                                           done=flags.pop("done", False),
                                           return_breakdown=True, **flags)
        return dict(parts).get("clean dodge", 0)

    def play(self, states):
        return [self.step(a, b) for a, b in zip(states, states[1:])]

    def test_profile_changes_only_dodge_and_empty_swing_objective_and_keeps_old_metadata(self):
        base = asdict(get_reward_config("radiant_spells_v1"))
        candidate = asdict(self.config)
        self.assertNotIn("clean_dodge_max_rewards", base)
        self.assertNotIn("clean_dodge_version", base)
        self.assertIsNone(candidate.pop("clean_dodge_max_rewards"))
        self.assertEqual(candidate.pop("clean_dodge_version"), "observed_move_completion_v1")
        self.assertEqual(candidate["clean_dodge"], 0.1)
        self.assertEqual(candidate["empty_attack"], -0.005)
        self.assertEqual(candidate["shockwave_spacing_bonus"], 0)
        self.assertEqual(candidate["shockwave_clear_bonus"], 0)
        candidate["profile"] = base["profile"]
        candidate["clean_dodge"] = base["clean_dodge"]
        candidate["empty_attack"] = base["empty_attack"]
        self.assertEqual(candidate, base)

    def test_confirmed_empty_nail_swing_costs_only_once_after_animation_ends(self):
        self.step(state(attacking=False), state(attacking=True))
        self.assertNotIn("empty attack", dict(self.tracker.last_breakdown))
        self.step(state(attacking=True), state(attacking=False))
        self.assertAlmostEqual(dict(self.tracker.last_breakdown)["empty attack"], -0.005)
        self.assertEqual(self.tracker.misses, 1)
        self.step(state(attacking=False), state(attacking=False))
        self.assertNotIn("empty attack", dict(self.tracker.last_breakdown))
        self.assertEqual(self.tracker.misses, 1)

    def test_hit_counter_or_actual_boss_hp_loss_prevents_empty_swing_penalty(self):
        for end in (state(attacking=False, hit_count=1),
                    state(attacking=False, boss_hp=1279)):
            with self.subTest(end=end):
                self.tracker.reset()
                self.step(state(attacking=False), state(attacking=True))
                self.step(state(attacking=True), end)
                self.assertNotIn("empty attack", dict(self.tracker.last_breakdown))
                self.assertEqual(self.tracker.misses, 0)

    def test_rejected_button_request_without_attack_animation_is_not_empty_swing(self):
        for _ in range(3):
            self.step(state(attack_requested=True), state(attack_requested=True))
            self.assertNotIn("empty attack", dict(self.tracker.last_breakdown))
        self.assertEqual(self.tracker.misses, 0)

    def test_each_move_completes_once_despite_multiple_active_states(self):
        sequences = (
            ("Attack Antic", "Attack 1", "Attack Swipe", "Attack Recover"),
            ("Jump Antic", "Jump", "DSlash Antic", "DSlash", "Land"),
            ("Slam Antic", "Slamming", "Waves", "Slam End"),
            ("Sickle Antic", "Sickle Throw", "Throw?", "Sickle Throw Recover"),
        )
        for names in sequences:
            with self.subTest(move=names[0]):
                self.tracker.reset()
                rewards = self.play([state("Idle")] + [state(name) for name in names]
                                    + [state(names[-1]), state("Idle")])
                self.assertAlmostEqual(sum(rewards), 0.1)
                self.assertEqual(self.tracker.dodges, 1)
                self.assertEqual(self.tracker.dodge_rewards, 1)
                self.assertAlmostEqual(rewards[-3], 0.1)

    def test_dive_intermediate_windup_is_not_an_end(self):
        rewards = self.play([state(fsm) for fsm in (
            "Jump Antic", "Jump", "DSlash Antic", "DSlash", "Land")])
        self.assertEqual(rewards, [0, 0, 0, 0.1])

    def test_beginning_in_active_or_middle_of_dive_is_not_complete(self):
        for names in (("Attack 1", "Attack Swipe", "Attack Recover"),
                      ("Jump", "DSlash Antic", "DSlash", "Land"),
                      ("DSlash Antic", "DSlash", "Land")):
            with self.subTest(names=names):
                self.tracker.reset()
                self.assertEqual(sum(self.play([state(name) for name in names])), 0)

    def test_known_non_active_to_active_edge_can_survive_missed_antic(self):
        self.assertEqual(self.play([state("Idle"), state("Attack 1"),
                                    state("Attack Recover")]), [0, 0.1])

    def test_damage_at_any_observed_stage_cancels_whole_move(self):
        names = ("Idle", "Attack Antic", "Attack 1", "Attack Swipe", "Attack Recover")
        for index in range(1, len(names)):
            for signal in ("hp", "counter"):
                with self.subTest(index=index, signal=signal):
                    self.tracker.reset()
                    snapshots = [state(name, hp=8 if i >= index and signal == "hp" else 9,
                                       damage=1 if i >= index and signal == "counter" else 0)
                                 for i, name in enumerate(names)]
                    self.assertEqual(sum(self.play(snapshots)), 0)

    def test_healing_after_damage_cannot_restore_clean_cycle(self):
        snapshots = [state("Attack Antic"), state("Attack 1", hp=8),
                     state("Attack Swipe"), state("Attack Recover")]
        self.assertEqual(sum(self.play(snapshots)), 0)

    def test_dead_missing_or_unknown_boss_never_counts_as_completion(self):
        for final in (state("Attack Recover", hp=0, damage=9),
                      state("Attack Recover", boss_hp=0),
                      state("Attack Recover", boss_hp=-1), state(""),
                      state("Unmapped State"), state("Roar")):
            with self.subTest(final=final):
                self.tracker.reset()
                self.step(state("Attack Antic"), state("Attack 1"))
                self.assertEqual(self.step(state("Attack 1"), final), 0)

    def test_unknown_mid_dive_cannot_rejoin_at_dslash_antic(self):
        self.assertEqual(sum(self.play([state(name) for name in (
            "Jump Antic", "Jump", "Unknown", "DSlash Antic", "DSlash", "Land")])), 0)

    def test_counter_discontinuity_invalidates_reward_until_reset(self):
        snapshots = [state("Attack Antic", damage=1), state("Attack 1", damage=0),
                     state("Attack Recover"), state("Attack Antic"),
                     state("Attack 1"), state("Attack Recover")]
        self.assertEqual(sum(self.play(snapshots)), 0)
        self.assertFalse(self.tracker.damage_tracking_valid)

    def test_terminal_timeout_or_truncation_never_awards(self):
        for flags in ({"done": True}, {"timed_out": True}, {"truncated": True}):
            with self.subTest(flags=flags):
                self.tracker.reset()
                self.step(state("Attack Antic"), state("Attack 1"))
                self.assertEqual(self.step(state("Attack 1"), state("Attack Recover"), **flags), 0)

    def test_no_reward_after_episode_ended_and_reset_rearms_tracker(self):
        snapshots = [state(name) for name in ("Attack Antic", "Attack 1", "Attack Recover")]
        for flags in ({"done": True}, {"timed_out": True}, {"truncated": True}):
            with self.subTest(flags=flags):
                self.tracker.reset()
                self.step(state("Idle"), state("Idle"), **flags)
                self.assertEqual(sum(self.play(snapshots)), 0)
                self.tracker.reset()
                self.assertAlmostEqual(sum(self.play(snapshots)), 0.1)

    def test_same_move_can_repeat_with_or_without_observed_recovery(self):
        for names in (("Attack Antic", "Attack 1", "Attack Recover", "Attack Antic",
                       "Attack 1", "Attack Recover"),
                      ("Attack Antic", "Attack 1", "Attack Antic", "Attack 1", "Attack Recover")):
            with self.subTest(names=names):
                self.tracker.reset()
                self.assertAlmostEqual(sum(self.play([state(name) for name in names])), 0.2)

    def test_twenty_clean_moves_all_receive_small_reward_without_episode_cap(self):
        names = ["Idle"] + [name for _ in range(20) for name in (
            "Attack Antic", "Attack 1", "Attack Recover")]
        self.assertAlmostEqual(sum(self.play([state(name) for name in names])), 2.0)
        self.assertEqual(self.tracker.dodge_rewards, 20)
        self.assertEqual(self.tracker.dodges, 20)

    def test_fsm_completion_does_not_claim_detached_wave_disappeared(self):
        # The user's chosen approximation intentionally allows residual waves.
        hazards = [{"box": (0, 0, 1, 10), "boss_attached": False}]
        self.assertEqual(sum(self.play([state(name, hazards=hazards) for name in (
            "Slam Antic", "Waves", "Slam End")])), 0.1)

    def test_direct_event_flag_still_rejects_terminal_and_invalid_transitions(self):
        for flags, current in (({"done": True}, state("Attack Recover")),
                               ({"timed_out": True}, state("Attack Recover")),
                               ({"truncated": True}, state("Attack Recover")),
                               ({}, state("Attack Recover", hp=0)),
                               ({}, state("Attack Recover", damage=1)),
                               ({}, state("Attack Recover", boss_hp=-1)),
                               ({}, state(""))):
            with self.subTest(flags=flags, current=current):
                _, parts = calculate_reward(state("Attack 1"), current,
                                            done=flags.pop("done", False),
                                            successful_dodge=True, reward_config=self.config,
                                            return_breakdown=True, **flags)
                self.assertNotIn("clean dodge", dict(parts))


if __name__ == "__main__":
    unittest.main()
