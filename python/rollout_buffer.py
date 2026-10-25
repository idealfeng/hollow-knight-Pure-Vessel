import numpy as np
import torch
 
 
class RolloutBuffer:
 

    def __init__(self, size, state_size, num_action_heads):
        self.size = size
        self.state_size = state_size
        self.num_action_heads = num_action_heads
        self.clear()

    def clear(self):
        self.obs = np.zeros((self.size, self.state_size), dtype=np.float32)
        self.actions = np.zeros((self.size, self.num_action_heads), dtype=np.int64)
        # Per-head, not just the joint sum, so ppo_update can recover a real
        # per-head PPO ratio later. The joint (what the objective actually
        # uses) is just logprobs.sum(axis=1) - storing per-head loses nothing
        # and is the only way to get a head-specific ratio after the fact.
        self.logprobs = np.zeros((self.size, self.num_action_heads), dtype=np.float32)
        self.values = np.zeros(self.size, dtype=np.float32)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.dones = np.zeros(self.size, dtype=np.float32)
        self.ptr = 0
 
    def __len__(self):
        return self.ptr
 
    @property
    def is_full(self):
        return self.ptr >= self.size
 
    def add(self, obs, action, logprob, value, reward, done):
        """
        `logprob`: per-head logprobs from model.act(), length num_action_heads.
        `done` means: the episode ENDED as a result of this step,
        i.e. the state that follows this transition is terminal.
        """
        if self.ptr >= self.size:
            raise RuntimeError("RolloutBuffer overflow - call clear() first")
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.ptr += 1
 
    # ------------------------------------------------------------------
 
    def compute_gae(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        Generalized Advantage Estimation.
 
        The advantage A(s,a) answers: "was this action better or worse than
        what the critic expected from this state?"  Positive -> make it more
        likely.  Negative -> make it less likely.  That signed number is the
        entire learning signal for the actor.
 
        The one-step version is the TD error:
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        Low variance, but biased by however wrong the critic is.
 
        The Monte-Carlo version is (full return - V(s_t)).
        Unbiased, but enormous variance over a 600-step boss fight.
 
        GAE interpolates between them with lambda:
            A_t = delta_t + (gamma*lambda) * A_{t+1}
        lambda=0 -> pure TD (biased, calm).  lambda=1 -> pure MC (noisy, honest).
        lambda=0.95 is the standard compromise.
 
        `last_value` = V(state right after the last stored step).  A rollout
        usually stops mid-fight, so we "bootstrap": we don't know the rest of
        the episode, so we let the critic guess the remaining value.
        When a step really terminated the episode, nonterminal=0 kills both the
        bootstrap and the recursion - future value is genuinely zero there.
        """
        n = self.ptr
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
 
        for t in reversed(range(n)):
            nonterminal = 1.0 - self.dones[t]
            next_value = last_value if t == n - 1 else self.values[t + 1]
 
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
 
        # The critic's regression target.  returns = advantage + old value
        # is just the GAE-smoothed estimate of the discounted return.
        returns = advantages + self.values[:n]
 
        return (
            torch.as_tensor(self.obs[:n]),
            torch.as_tensor(self.actions[:n]),
            torch.as_tensor(self.logprobs[:n]),
            torch.as_tensor(advantages),
            torch.as_tensor(returns),
            torch.as_tensor(self.values[:n]),
        )
 
