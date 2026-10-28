"""Offline checks for the human-demonstration -> BC dataset bridge."""

import unittest

import numpy as np

from agent.action import MacroAction
from build_human_bc_dataset import (
    downsample_frames,
    frame_to_state_dict,
    held_mask_to_macro,
    stack_observations,
)
from game_state import _parse_frame
from human_demo import BUTTON_NAMES


def bit(*names):
    """Button bitmask from human_demo.BUTTON_NAMES entries."""
    mask = 0
    for name in names:
        mask |= 1 << BUTTON_NAMES.index(name)
    return mask


def make_frame(hp=1600, hero_hp=9, fsm="Idle", hits=0, damages=0,
               boss_box="51,8,53,10"):
    wire = (f"40,5,40.5,6.3;0;;{hits},{damages};"
            f"0,0,1,{hp},0,0,0,{hero_hp},0,1,52,9,10,20;"
            f"0,0,0,0,0,0,1,0,0,1;{fsm};{boss_box}")
    return _parse_frame(wire)


class HeldMaskToMacroTests(unittest.TestCase):
    def test_empty_mask_is_idle(self):
        self.assertEqual(held_mask_to_macro(0), MacroAction.IDLE)

    def test_pure_movement(self):
        self.assertEqual(held_mask_to_macro(bit("left")), MacroAction.MOVE_LEFT)
        self.assertEqual(held_mask_to_macro(bit("right")), MacroAction.MOVE_RIGHT)

    def test_attack_directions_win_over_movement(self):
        self.assertEqual(held_mask_to_macro(bit("attack", "up")), MacroAction.ATTACK_UP)
        self.assertEqual(held_mask_to_macro(bit("attack", "down")), MacroAction.ATTACK_DOWN)
        self.assertEqual(held_mask_to_macro(bit("attack", "left")), MacroAction.ATTACK_LEFT)
        self.assertEqual(held_mask_to_macro(bit("attack", "right")), MacroAction.ATTACK_RIGHT)

    def test_moving_attack_keeps_facing_direction(self):
        # The macro set has no "move + attack"; the attack keeps the direction.
        self.assertEqual(held_mask_to_macro(bit("attack", "right")), MacroAction.ATTACK_RIGHT)
        self.assertEqual(held_mask_to_macro(bit("attack", "left")), MacroAction.ATTACK_LEFT)

    def test_bare_attack_uses_knight_facing(self):
        facing_right = {"knight_facing_right": True}
        facing_left = {"knight_facing_right": False}
        self.assertEqual(held_mask_to_macro(bit("attack"), facing_right), MacroAction.ATTACK_RIGHT)
        self.assertEqual(held_mask_to_macro(bit("attack"), facing_left), MacroAction.ATTACK_LEFT)

    def test_dash_and_jump_chords_are_preserved(self):
        self.assertEqual(held_mask_to_macro(bit("dash", "left")), MacroAction.DASH_LEFT)
        self.assertEqual(held_mask_to_macro(bit("dash", "right")), MacroAction.DASH_RIGHT)
        self.assertEqual(held_mask_to_macro(bit("jump")), MacroAction.JUMP)
        self.assertEqual(held_mask_to_macro(bit("jump", "left")), MacroAction.JUMP_LEFT)
        self.assertEqual(held_mask_to_macro(bit("jump", "right")), MacroAction.JUMP_RIGHT)

    def test_conflicting_directions_collapse_to_idle(self):
        self.assertEqual(held_mask_to_macro(bit("left", "right")), MacroAction.IDLE)


class FrameToStateDictTests(unittest.TestCase):
    def test_hit_and_damage_are_episode_deltas(self):
        frame = make_frame(hits=10, damages=5)
        state = frame_to_state_dict(frame, hit_baseline=3, damage_baseline=2)
        self.assertEqual(state["hit_count"], 7)
        self.assertEqual(state["damage_count"], 3)

    def test_hazards_remain_objects_for_state_to_vector(self):
        frame = make_frame()
        state = frame_to_state_dict(frame)
        self.assertEqual(state["hazards"], [])
        self.assertEqual(state["hazard_boxes"], [])
        self.assertEqual(state["boss_control_state"], "Idle")
        self.assertEqual(state["boss_box"], (51.0, 8.0, 53.0, 10.0))


class DownsampleTests(unittest.TestCase):
    def test_picks_first_frame_at_each_grid_point(self):
        records = [
            ({"game_time": 0.0}, None),
            ({"game_time": 0.016}, None),
            ({"game_time": 0.05}, None),
            ({"game_time": 0.1}, None),
            ({"game_time": 0.11}, None),
            ({"game_time": 0.2}, None),
        ]
        picked = downsample_frames(records, start_game_time=0.0, step_dt=0.1)
        self.assertEqual([r["game_time"] for r, _ in picked], [0.0, 0.1, 0.2])

    def test_negative_relative_time_is_ignored(self):
        records = [({"game_time": 9.9}, None), ({"game_time": 10.0}, None),
                   ({"game_time": 10.05}, None)]
        picked = downsample_frames(records, start_game_time=10.0, step_dt=0.1)
        self.assertEqual([r["game_time"] for r, _ in picked], [10.0])


class StackObservationsTests(unittest.TestCase):
    def test_first_frame_is_replicated_to_fill_the_stack(self):
        observations = [np.full(126, float(i), dtype=np.float32) for i in range(5)]
        stacked = stack_observations(observations, count=4)
        self.assertEqual(len(stacked), 5)
        self.assertEqual(stacked[0].shape, (504,))
        np.testing.assert_array_equal(stacked[0], np.zeros(504, dtype=np.float32))
        np.testing.assert_array_equal(stacked[1][378:], np.full(126, 1.0, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
