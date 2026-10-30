import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
 
 
def _orthogonal(layer, gain):
    """Orthogonal init is the standard for PPO - it keeps activations from
    blowing up or vanishing through the layers early in training."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer
 
 
class ActorCritic(nn.Module):
 
    def __init__(self, state_size, action_sizes, hidden=128, bernoulli=False,
                 exclusive_directions=False):
        """``action_sizes`` is kept generic. With ``bernoulli=True`` each head
        is one Bernoulli logit (multi-binary button actions); otherwise each
        head is a categorical over ``n`` discrete choices."""
        super().__init__()

        self.bernoulli = bool(bernoulli)
        self.exclusive_directions = bool(exclusive_directions)
        if self.exclusive_directions and (not self.bernoulli or len(action_sizes) != 7
                                          or any(size != 1 for size in action_sizes)):
            raise ValueError("exclusive directions require seven Bernoulli button heads")

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

    def configure_trainable_parameters(self, mode="all"):
        """Select parameter updates without changing action sampling or scoring.

        Frozen heads need a frozen trunk as well: otherwise their logits would
        change even though their own weights did not. Keeping every parameter
        in Adam's original groups permits loading a full-policy optimizer;
        frozen parameters have no gradient and Adam skips them, including any
        previously accumulated moments.
        """
        if mode not in ("all", "dash_only"):
            raise ValueError(f"unknown training mode: {mode!r}")
        if mode == "dash_only" and (not self.bernoulli or len(self.heads) != 7):
            raise ValueError("dash_only requires the seven-button Bernoulli policy")
        names = []
        for name, parameter in self.named_parameters():
            trainable = mode == "all" or name.startswith(("heads.6.", "critic."))
            parameter.requires_grad_(trainable)
            parameter.grad = None
            if trainable:
                names.append(name)
        return names
 
    def forward(self, obs):
        h = self.trunk(obs)
        logits = [head(h) for head in self.heads]
        value = self.critic(h).squeeze(-1)
        return logits, value

    @staticmethod
    def _direction_dist(first, second):
        # One categorical choice: neither, first, or second. This removes
        # left+right and up+down from the policy's action space entirely.
        return Categorical(logits=torch.cat((torch.zeros_like(first), first, second), dim=-1))

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
        start = 0
        if self.exclusive_directions:
            for first, second in ((0, 1), (2, 3)):
                dist = self._direction_dist(logits_list[first], logits_list[second])
                choice = dist.sample()
                actions.extend((int((choice == 1).item()), int((choice == 2).item())))
                logprobs.extend((dist.log_prob(choice).item(), 0.0))
            start = 4
        for logits in logits_list[start:]:
            if self.bernoulli:
                dist = Bernoulli(logits=logits)
            else:
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
        if self.exclusive_directions:
            actions = []
            for first, second in ((0, 1), (2, 3)):
                choice = self._direction_dist(logits_list[first], logits_list[second]).probs.argmax(dim=-1)
                actions.extend((int((choice == 1).item()), int((choice == 2).item())))
            actions.extend(int((logits > 0).long().item()) for logits in logits_list[4:])
            return tuple(actions)
        if self.bernoulli:
            return tuple(int((logits > 0).long().item()) for logits in logits_list)
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
        start = 0
        if self.exclusive_directions:
            for first, second in ((0, 1), (2, 3)):
                if torch.any((action_batch[:, first] != 0) & (action_batch[:, second] != 0)):
                    raise ValueError("opposing directions cannot be pressed together")
                choice = (action_batch[:, first] != 0).long() + 2 * (action_batch[:, second] != 0).long()
                dist = self._direction_dist(logits_list[first], logits_list[second])
                pair_logprob = dist.log_prob(choice)
                pair_entropy = dist.entropy()
                logprobs.extend((pair_logprob, torch.zeros_like(pair_logprob)))
                entropies.extend((pair_entropy, torch.zeros_like(pair_entropy)))
            start = 4
        for i in range(start, len(logits_list)):
            logits = logits_list[i]
            if self.bernoulli:
                dist = Bernoulli(logits=logits)
                a_i = action_batch[:, i].float().unsqueeze(-1)
                logprobs.append(dist.log_prob(a_i).squeeze(-1))
                entropies.append(dist.entropy().squeeze(-1))
            else:
                dist = Categorical(logits=logits)
                a_i = action_batch[:, i]
                logprobs.append(dist.log_prob(a_i))
                entropies.append(dist.entropy())

        logprob_per_head = torch.stack(logprobs, dim=1)     # (N, num_heads)
        entropy_per_head = torch.stack(entropies, dim=1)    # (N, num_heads)
        return logprob_per_head, entropy_per_head, values
