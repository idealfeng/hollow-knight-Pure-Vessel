"""Offline checks for demonstration boundaries and observation migration."""

import unittest

import numpy as np

from train_bc import discounted_returns, migrate_observations
from main_multi import CHECKPOINT_FORMAT, REWARD_SCALE


class DemonstrationTests(unittest.TestCase):
    def test_returns_do_not_cross_episode_boundaries(self):
        rewards = np.array([1.0, 2.0, 10.0, 20.0])
        dones = np.array([False, True, False, True])
        episodes = np.array([0, 0, 1, 1])
        result = discounted_returns(rewards, dones, episodes, gamma=0.5)
        np.testing.assert_allclose(result, np.array([2, 2, 20, 20]) * REWARD_SCALE)

    def test_incomplete_episode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            discounted_returns(np.array([1.0, 2.0]),
                               np.array([False, False]), np.array([0, 0]))

    def test_facing_migration_handles_each_frame_and_missing_boss(self):
        frames = np.zeros((2, 4, 126), dtype=np.float32)
        frames[:, :, 22] = 1
        frames[:, :, 24] = [2, 2, -2, -2]
        frames[:, :, 30] = [1, 0, 1, 0]
        frames[:, :, 31] = [0, 1, 1, 0]  # old, inverted flag
        frames[1, 0, 22] = 0
        frames[1, 3, 24] = 0
        original = frames.reshape(2, 504).copy()
        converted = migrate_observations(original, 4).reshape(2, 4, 126)
        np.testing.assert_array_equal(converted[:, :, 31], [[1, 0, 0, 1], [0, 0, 0, 0]])
        np.testing.assert_array_equal(original, frames.reshape(2, 504))
        # A current-format demonstration must not be transformed a second time.
        np.testing.assert_array_equal(
            migrate_observations(converted.reshape(2, 504), CHECKPOINT_FORMAT),
            converted.reshape(2, 504))


if __name__ == "__main__":
    unittest.main()
