import numpy as np
import torch


def ppo_update(model,
               optimizer,
               buffer,
               last_value,
               gamma=0.99,
               gae_lambda=0.95,
               clip_eps=0.2,
               epochs=4,
               minibatch_size=128,
               ent_coef=0.01,
               vf_coef=0.5,
               max_grad_norm=0.5,
               target_kl=0.03,
               clip_value_loss=True,
               value_clip_eps=None):
    """value_clip_eps: clip range for the value function specifically. Kept
    separate from clip_eps (the policy ratio's clip) because they bound
    different units - clip_eps is a dimensionless probability ratio, while
    the value clip bounds a change in return-scaled value. Reusing clip_eps
    for both (the old behaviour, still the default here via None) ties the
    critic's per-update step size to a number chosen for the policy, which
    is where the "critic can't reach the right magnitude" symptom came from
    a few sessions back. Defaults to clip_eps for exact backward compatibility
    when not given."""

    if value_clip_eps is None:
        value_clip_eps = clip_eps

    obs, actions, old_logprobs, advantages, returns, old_values = \
        buffer.compute_gae(last_value, gamma, gae_lambda)
    # old_logprobs: (n, num_heads) - see rollout_buffer.py. The joint used by
    # the actual PPO objective is just the per-head sum; nothing about the
    # clipped surrogate loss changes from before, this only ALSO exposes the
    # per-head decomposition for diagnostics.
    num_heads = old_logprobs.shape[1]

    n = obs.shape[0]
    if n < 2:
        return None

    indices = np.arange(n)

    approx_kl = 0.0
    logs = {"policy_loss": [], "value_loss": [], "entropy": [],
            "clipfrac": [], "approx_kl": []}
    # Per-head diagnostics. These do NOT feed the clipped objective or the
    # early-stop decision - both remain joint, exactly as before - they are
    # read-only visibility into which head is moving, so a head that is
    # collapsing or oscillating shows up even though it can't be "the cause"
    # of a value-loss spike (see grad_norm_trunk/grad_norm_critic below for
    # that - value_loss's gradient does not reach any action head's weights
    # at all, only the shared trunk and the critic layer, so per-head norms
    # are included for completeness but are not where a value-loss problem
    # would show up).
    per_head_logs = {
        "entropy": [[] for _ in range(num_heads)],
        "approx_kl": [[] for _ in range(num_heads)],
        "clipfrac": [[] for _ in range(num_heads)],
        "grad_norm": [[] for _ in range(num_heads)],
    }
    grad_norm_trunk = []
    grad_norm_critic = []
    stopped_early = False

    for epoch in range(epochs):
        np.random.shuffle(indices)

        for start in range(0, n, minibatch_size):
            mb = torch.as_tensor(indices[start:start + minibatch_size],
                                 dtype=torch.long)
            if mb.numel() < 2:
                continue

            # new_logprobs/entropy: (mb_size, num_heads) - per head, not
            # summed. The joint objective below sums them right back to
            # exactly what it used before; per-head values are kept alongside
            # purely for diagnostics.
            new_logprobs, entropy, values = model.evaluate(obs[mb], actions[mb])

            # ---------------- the PPO ratio ----------------
            # ratio = pi_new(a|s) / pi_old(a|s), computed in log space for
            # numerical safety.  On the very first minibatch of the first
            # epoch the policy is unchanged, so ratio == 1 exactly.
            # Summing per-head logratios equals the joint logratio - the
            # clipped objective and the early-stop trigger are unchanged from
            # before this refactor (verified: see test_head_refactor.py).
            logratio_per_head = new_logprobs - old_logprobs[mb]
            logratio = logratio_per_head.sum(dim=1)
            ratio = logratio.exp()

            with torch.no_grad():
                # k3 estimator of KL(old || new): low-variance, always >= 0.
                # This is our "how far did the policy move" alarm.
                approx_kl = ((ratio - 1.0) - logratio).mean().item()
                clipfrac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

                # Per-head versions of the same two diagnostics, computed on
                # each head's OWN ratio in isolation - this is what actually
                # answers "is one head moving much more/less than the others",
                # which the joint numbers above cannot show by construction
                # (a calm head and a wild one can average out to a normal-
                # looking joint KL).
                ratio_per_head = logratio_per_head.exp()
                for i in range(num_heads):
                    head_kl = ((ratio_per_head[:, i] - 1.0) - logratio_per_head[:, i]).mean().item()
                    head_clip = (ratio_per_head[:, i] - 1.0).abs().gt(clip_eps).float().mean().item()
                    per_head_logs["approx_kl"][i].append(head_kl)
                    per_head_logs["clipfrac"][i].append(head_clip)
                    per_head_logs["entropy"][i].append(entropy[:, i].mean().item())

            # ---------------- advantage normalisation ----------------
            # Rescale advantages to mean 0 / std 1 within the minibatch.
            # Only the SIGN and RELATIVE size matter for the policy gradient,
            # and this makes the effective learning rate independent of how
            # big your reward numbers happen to be.
            mb_adv = advantages[mb]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            # ---------------- clipped surrogate objective ----------------
            # Plain policy gradient would be  -(adv * ratio):
            #   adv > 0 -> push ratio up   (make the action more likely)
            #   adv < 0 -> push ratio down (make it less likely)
            # Unclipped, one big advantage can shove the policy off a cliff and
            # it never recovers.  So we also compute the version with ratio
            # clamped to [1-eps, 1+eps] and take the PESSIMISTIC (max of the
            # losses = min of the objectives) one.
            #
            # Effect: once an action's probability has already moved more than
            # ~20%, the clipped branch wins, its gradient is flat, and PPO
            # stops pushing further in that direction this update.  That single
            # trick is the whole reason PPO is stable.  ("Proximal" = stay near
            # the old policy.)
            pg_loss_1 = -mb_adv * ratio
            pg_loss_2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

            # ---------------- value (critic) loss ----------------
            # Plain regression of V(s) onto the GAE returns.  This is the piece
            # that most resembles your old DQN loss. Clipped with its own
            # value_clip_eps, not clip_eps - see the docstring above for why
            # sharing one epsilon between the policy ratio and the value
            # function is a category error, not just a coincidence of naming.
            if clip_value_loss:
                v_unclipped = (values - returns[mb]) ** 2
                v_clip = old_values[mb] + torch.clamp(
                    values - old_values[mb], -value_clip_eps, value_clip_eps)
                v_clipped = (v_clip - returns[mb]) ** 2
                value_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
            else:
                value_loss = 0.5 * ((values - returns[mb]) ** 2).mean()

            # ---------------- entropy bonus ----------------
            # Entropy = how spread out the action distribution is.
            # Max for 8 actions is ln(8) = 2.08 (uniform); 0 means "always the
            # same action".  We SUBTRACT it from the loss, i.e. reward being
            # uncertain, so the policy does not collapse onto one action before
            # it has actually explored.  This is PPO's replacement for epsilon.
            # entropy is (mb_size, num_heads) now - sum over heads first (the
            # joint entropy of independent heads), then average over the
            # batch, reproducing exactly what entropy.mean() did when
            # evaluate() summed the heads internally.
            entropy_loss = entropy.sum(dim=1).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                def _grad_norm(params):
                    gs = [p.grad.norm() for p in params if p.grad is not None]
                    return float(torch.stack(gs).norm()) if gs else 0.0

                grad_norm_trunk.append(_grad_norm(model.trunk.parameters()))
                grad_norm_critic.append(_grad_norm(model.critic.parameters()))
                for i, head in enumerate(model.heads):
                    per_head_logs["grad_norm"][i].append(_grad_norm(head.parameters()))

            # 0.5 not 10.0 - PPO wants small, careful steps.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            logs["policy_loss"].append(policy_loss.item())
            logs["value_loss"].append(value_loss.item())
            logs["entropy"].append(entropy_loss.item())
            logs["clipfrac"].append(clipfrac)
            logs["approx_kl"].append(approx_kl)

        # Safety valve: if the policy has already drifted far from the one that
        # collected this data, the remaining epochs would be extrapolating from
        # stale samples.  Bail out.
        if target_kl is not None and approx_kl > target_kl:
            stopped_early = True
            break

    # Explained variance: how much of the return's variance the critic captures.
    # ~1.0 = excellent critic.  ~0.0 = critic is no better than predicting the
    # mean.  Negative = critic is actively worse than useless.  This is the
    # single most useful PPO diagnostic.
    y_pred = old_values.numpy()
    y_true = returns.numpy()
    var_y = np.var(y_true)
    explained_var = float("nan") if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

    return {
        "policy_loss": float(np.mean(logs["policy_loss"])),
        "value_loss": float(np.mean(logs["value_loss"])),
        "entropy": float(np.mean(logs["entropy"])),
        "clipfrac": float(np.mean(logs["clipfrac"])),
        "approx_kl": float(np.mean(logs["approx_kl"])),
        "explained_variance": explained_var,
        "total_loss": float(np.mean(logs["policy_loss"])
                            + vf_coef * np.mean(logs["value_loss"])),
        "stopped_early": stopped_early,
        "n_samples": n,
        # Per-head diagnostics - read-only, do not feed the objective or the
        # early-stop trigger (both stay joint, as verified against the
        # pre-refactor behaviour). Answers "is one head moving oddly" -
        # NOT "is one head causing the value-loss spikes", which is
        # structurally impossible in this architecture (see grad_norm_head
        # below and the docstring on value_clip_eps).
        "entropy_per_head": [float(np.mean(x)) for x in per_head_logs["entropy"]],
        "approx_kl_per_head": [float(np.mean(x)) for x in per_head_logs["approx_kl"]],
        "clipfrac_per_head": [float(np.mean(x)) for x in per_head_logs["clipfrac"]],
        "grad_norm_per_head": [float(np.mean(x)) for x in per_head_logs["grad_norm"]],
        # These two are what actually explains a value-loss spike, since
        # value_loss's gradient never reaches any action head's weights -
        # only the shared trunk and the critic layer receive it.
        "grad_norm_trunk": float(np.mean(grad_norm_trunk)),
        "grad_norm_critic": float(np.mean(grad_norm_critic)),
    }
