"""Schema compatibility and potential reward integration without game access."""
from dataclasses import asdict
import unittest
from unittest.mock import patch
import game_state

with patch.object(game_state, "start"):
    from enviroment.state_to_vector import state_to_vector
    from hollow_knight_env import HollowKnightEnv

from agent.reward import RewardTracker, get_reward_config
from combat_potential import potential_reward
from observation_schema import COMBAT_SCHEMA, LEGACY_SCHEMA, DASH_FEATURES


def snapshot(**changes):
    value = dict(knight_box=(0, 0, .5, 1.3), boss_box=(3, 0, 5, 4),
                 knight_health=9, mana=66, knight_facing_right=True,
                 knight_velocity=(0, 0), attacking=False, is_dashing=False,
                 is_jumping=False, is_falling=False, is_recoiling=False,
                 is_on_ground=True, is_touching_wall_left=False, is_touching_wall_right=False,
                 wall_distance_left=10, wall_distance_right=30,
                 can_shade_cloak_now=True, is_invuln=False, attack_box=None,
                 boss_health=1300, boss_facing_left=True, boss_velocity=(0, 0),
                 boss_position=(4, 2), hit_count=0, damage_count=0, boss_state_age=.1,
                 boss_control_state="Idle", hazards=[],
                 dash_state_valid=True, can_dash_now=True, is_shadow_dashing=False,
                 shadow_cooldown_remaining=0, shadow_dash_remaining=0,
                 dash_remaining=0, invulnerability_remaining=0, invulnerability_time_valid=True)
    value.update(changes)
    return value


class CombatObservationTests(unittest.TestCase):
    def test_legacy_columns_remain_exact_and_new_features_are_appended(self):
        state = snapshot(shadow_cooldown_remaining=.8, can_shade_cloak_now=False)
        legacy = state_to_vector(state, .25)
        expanded = state_to_vector(state, .25, schema=COMBAT_SCHEMA)
        self.assertEqual(len(legacy), 118)
        self.assertEqual(len(expanded), 128)
        self.assertEqual(expanded[:118], legacy)
        self.assertEqual(expanded[118:126], [float(state[name]) for name in DASH_FEATURES])
        self.assertEqual(expanded[-2:], [.5, 1.3])

    def test_old_game_telemetry_cannot_silently_enter_new_training(self):
        state = snapshot(dash_state_valid=False)
        self.assertEqual(len(state_to_vector(state)), 118)
        with self.assertRaisesRegex(RuntimeError, "requires the dash_v1"):
            state_to_vector(state, schema=COMBAT_SCHEMA)
        with self.assertRaisesRegex(ValueError, "requires combat_v2"):
            HollowKnightEnv(reward_profile="radiant_potential_v1")
        env = HollowKnightEnv(reward_profile="radiant_potential_v1", observation_schema=COMBAT_SCHEMA,
                              action_distribution="exclusive_directions_spells_v1")
        self.assertEqual(env.observation_space.shape, (512,))

    def test_potential_reward_is_added_without_changing_damage_and_death_rewards(self):
        config = get_reward_config("radiant_potential_v1")
        tracker = RewardTracker(config)
        before = snapshot(knight_velocity=(7, 0))
        after = snapshot(knight_box=(1, 0, 1.5, 1.3), boss_health=1279, hit_count=1)
        _, parts = tracker.transition(before, after, False, return_breakdown=True)
        self.assertEqual(dict(parts)["boss damage"], 1)
        self.assertAlmostEqual(dict(parts)["potential shaping"], potential_reward(before, after, config=config.potential))
        death = snapshot(knight_health=0, damage_count=9, boss_health=1279)
        _, parts = tracker.transition(after, death, True, return_breakdown=True)
        self.assertEqual(dict(parts)["took damage"], -27)
        self.assertEqual(dict(parts)["die"], -20)
        self.assertAlmostEqual(dict(parts)["potential shaping"],
                               potential_reward(after, death, terminal=True, config=config.potential))

    def test_invalid_potential_does_not_award_a_fake_improvement(self):
        tracker = RewardTracker(get_reward_config("radiant_potential_v1"))
        with self.assertRaisesRegex(RuntimeError, "refusing to drop a shaping transition"):
            tracker.transition(snapshot(), snapshot(dash_state_valid=False), False,
                               return_breakdown=True)
        self.assertEqual(tracker.potential_invalid_steps, 1)

    def test_profile_keeps_requested_small_rewards_and_replaces_far_pressure(self):
        config = asdict(get_reward_config("radiant_potential_v1"))
        self.assertEqual((config["clean_dodge"], config["empty_attack"]), (.1, -.005))
        self.assertEqual(config["far_penalty_per_second"], 0)
        self.assertEqual(config["shockwave_clear_bonus"], 0)
        self.assertEqual(config["shockwave_spacing_bonus"], 0)
        self.assertEqual(config["potential"]["gamma"], .9955)


if __name__ == "__main__":
    unittest.main()
