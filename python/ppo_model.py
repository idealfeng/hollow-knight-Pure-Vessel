import torch
import torch.nn as nn
from torch.distributions import Categorical
 
 
def _orthogonal(layer, gain):
    """Orthogonal init is the standard for PPO - it keeps activations from
    blowing up or vanishing through the layers early in training."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer
 
 
class ActorCritic(nn.Module):
 
    def __init__(self, state_size, action_sizes, hidden=128):
        """``action_sizes`` is kept generic, but the current environment uses
        ``[12]``: one categorical head over deliberate macro actions."""
        super().__init__()

        self.trunk = nn.Sequential(
            _orthogonal(nn.Linear(state_size, hidden), 2 ** 0.5),
            nn.Tanh(),
            _orthogonal(nn.Linear(hidden, hidden), 2 ** 0.5),
            nn.Tanh(),
        )

        self.heads = nn.ModuleList([
            _orthogonal(nn.Linear(hidden, n), 0.01) for n in action_sizes
        ])
        self.critic = _orthogonal(nn.Linear(hidden, 1), 1.0)
 
    def forward(self, obs):
        h = self.trunk(obs)
        logits = [head(h) for head in self.heads]
        value = self.critic(h).squeeze(-1)
        return logits, value

    # ---------- used while PLAYING (no gradients) ----------

    @torch.no_grad()
    def act(self, obs_vector):
        """Returns (actions_tuple, logprobs_per_head, joint_logprob, value).
        Joint logprob is the sum of each head's logprob - the heads are
        independent given the shared trunk, so log P(a1,...,an) = sum log P(ai).
        logprobs_per_head is cached (not just the joint sum) so ppo_update can
        later compute a genuine per-head KL/clipfrac against the ratio each
        head actually moved by, not just the combined one."""
        obs = torch.as_tensor(obs_vector, dtype=torch.float32).unsqueeze(0)
        logits_list, value = self(obs)

        actions, logprobs = [], []
        for logits in logits_list:
            dist = Categorical(logits=logits)
            a = dist.sample()
            actions.append(int(a.item()))
            logprobs.append(dist.log_prob(a).item())

        joint_logprob = sum(logprobs)
        return tuple(actions), logprobs, joint_logprob, float(value.item())
 
    @torch.no_grad()
    def greedy_act(self, obs_vector):
        obs = torch.as_tensor(obs_vector, dtype=torch.float32).unsqueeze(0)
        logits_list, _ = self(obs)
        return tuple(int(torch.argmax(logits, dim=-1).item()) for logits in logits_list)
 
    @torch.no_grad()
    def value_only(self, obs_vector):
        obs = torch.as_tensor(obs_vector, dtype=torch.float32).unsqueeze(0)
        _, value = self(obs)
        return float(value.item())

 
    # ---------- used while LEARNING (gradients ON) ----------

    def evaluate(self, obs_batch, action_batch):
        """action_batch: shape (N, num_heads), column i = the action taken
        by head i. Returns per-head logprobs (N, num_heads) and per-head
        entropies (N, num_heads) - NOT pre-summed, so callers can compute
        both the joint objective (sum over heads, unchanged from before)
        and genuine per-head diagnostics from the same tensors. value is
        per sample, shared - there is only one critic, not one per head."""
        logits_list, values = self(obs_batch)

        logprobs, entropies = [], []
        for i, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            a_i = action_batch[:, i]
            logprobs.append(dist.log_prob(a_i))
            entropies.append(dist.entropy())

        logprob_per_head = torch.stack(logprobs, dim=1)     # (N, num_heads)
        entropy_per_head = torch.stack(entropies, dim=1)    # (N, num_heads)
        return logprob_per_head, entropy_per_head, values
