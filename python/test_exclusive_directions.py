"""Check the policy's button distribution before a costly live training run."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from ppo_model import ActorCritic
from agent.action import execute_buttons


class ExclusiveDirectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = ActorCritic(8, [1] * 7, hidden=16, bernoulli=True,
                                 exclusive_directions=True)
        self.observation = np.zeros(8, dtype=np.float32)

    def test_sampled_and_greedy_actions_never_conflict(self):
        for _ in range(300):
            action, _, _, _ = self.model.act(self.observation)
            self.assertLessEqual(action[0] + action[1], 1)
            self.assertLessEqual(action[2] + action[3], 1)
        greedy = self.model.greedy_act(self.observation)
        self.assertLessEqual(greedy[0] + greedy[1], 1)
        self.assertLessEqual(greedy[2] + greedy[3], 1)

    def test_rollout_logprob_matches_training_logprob(self):
        action, old_logprobs, _, _ = self.model.act(self.observation)
        obs = torch.as_tensor(self.observation).unsqueeze(0)
        actions = torch.as_tensor(action).unsqueeze(0)
        new_logprobs, entropy, _ = self.model.evaluate(obs, actions)
        torch.testing.assert_close(new_logprobs[0], torch.tensor(old_logprobs))
        self.assertTrue(torch.isfinite(entropy).all())
        loss = -new_logprobs.sum() - 0.01 * entropy.sum()
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.heads[0].weight.grad).all())

    def test_evaluation_rejects_opposing_directions(self):
        obs = torch.zeros((1, 8))
        actions = torch.tensor([[1, 1, 0, 0, 0, 0, 0]])
        with self.assertRaises(ValueError):
            self.model.evaluate(obs, actions)

    def test_held_attack_repeats_but_held_dash_does_not(self):
        action = (0, 0, 0, 0, 1, 0, 1)
        with patch("agent.action.controls.tap") as tap:
            previous = execute_buttons(action)
            execute_buttons(action, previous)
        self.assertEqual([call.args[0] for call in tap.call_args_list],
                         ["attack", "dash", "attack"])


if __name__ == "__main__":
    unittest.main()
