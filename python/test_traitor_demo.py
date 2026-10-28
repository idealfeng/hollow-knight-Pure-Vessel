"""Protect the Traitor Lord recording boundaries from intro/menu frames."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from human_demo import EpisodeTracker


def state(fsm, *, boss_hp=800, knight_hp=9):
    return SimpleNamespace(
        knight_box=(0, 0, 1, 1), knight_health=knight_hp,
        boss_health=boss_hp, boss_box=(2, 0, 3, 2),
        boss_control_state=fsm, hit_count=0, damage_count=0,
    )


def record(frame):
    return {"frame": frame, "game_time": frame / 60.0, "pressed": 0}


class TraitorDemoTests(unittest.TestCase):
    def test_intro_is_not_recorded_as_combat(self):
        tracker = EpisodeTracker()
        for frame, fsm in enumerate(("Init", "Emerge Dust", "Roar", "Roar End")):
            began, finished = tracker.observe(record(frame), state(fsm))
            self.assertFalse(began)
            self.assertIsNone(finished)
        began, finished = tracker.observe(record(4), state("Idle"))
        self.assertTrue(began)
        self.assertIsNone(finished)

    def test_death_does_not_start_a_new_fight_before_next_intro(self):
        tracker = EpisodeTracker()
        tracker.observe(record(0), state("Idle"))
        _, finished = tracker.observe(record(1), state("Attack 1", knight_hp=0))
        self.assertIsNotNone(finished)
        began, _ = tracker.observe(record(2), state("Idle", boss_hp=600))
        self.assertFalse(began)
        tracker.observe(record(3), state("Init"))
        began, _ = tracker.observe(record(4), state("Idle"))
        self.assertTrue(began)

    def test_dataset_reward_follows_the_recorded_action(self):
        import build_human_bc_dataset as builder

        episodes = [{"episode": 1, "start_frame": 0, "end_frame": 13,
                     "start_game_time": 0.0, "start_hits": 0, "start_damage": 0}]
        frames = [({"frame": i, "game_time": i / 20.0, "held": 0},
                   SimpleNamespace(index=i, boss_control_state="Idle"))
                  for i in range(14)]
        with patch.object(builder, "load_episodes", return_value=episodes), \
             patch.object(builder, "load_frames", return_value=frames), \
             patch.object(builder, "frame_to_state_dict",
                          side_effect=lambda frame, *args, **kwargs: {"index": frame.index}), \
             patch.object(builder, "state_to_vector",
                          side_effect=lambda state, time_frac: [float(state["index"])] * 118), \
             patch.object(builder, "calculate_reward",
                          side_effect=lambda previous, current, done, step_dt: float(current["index"])):
            data = builder.build_session(Path("synthetic"))

        self.assertEqual(len(data["actions"]), 7)
        np.testing.assert_array_equal(data["rewards"], [2, 4, 6, 8, 10, 12, 13])
        np.testing.assert_array_equal(data["dones"], [False] * 6 + [True])
        self.assertEqual(data["observations"].shape, (7, 472))


if __name__ == "__main__":
    unittest.main()
