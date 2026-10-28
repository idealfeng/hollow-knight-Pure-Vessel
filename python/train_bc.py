"""Behavior-clone the 12-action PPO actor from synchronized live demonstrations.

The split is by complete episode, so nearby stacked frames from the same
fight cannot leak into validation. The output is a normal multi-env PPO
checkpoint with update=-1; PPO starts at update 0 from the cloned actor.
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from agent.action import MACRO_ACTION_NAMES
from main_multi import (ACTION_SIZES, CHECKPOINT_FORMAT, GAMMA, LR, MODEL_HIDDEN, REWARD_SCALE,
                        default_steps_per_env)
from enviroment.state_to_vector import OBSERVATION_SIZE
from obs_norm import RunningNorm
from ppo_model import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", nargs="+", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ports", type=int, nargs="+", default=[9001, 9002])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--critic-epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.critic_epochs < 1:
        parser.error("epochs, critic-epochs and batch-size must be positive")
    if len(set(args.ports)) != len(args.ports):
        parser.error("ports must be unique")
    return args


def migrate_observations(observations, source_format):
    """Format 4 already stores all inputs needed to correct the facing flag.

    Recompute it in every stacked frame, including absent/zero-offset bosses.
    This does not alter the teacher labels or the recorded trajectories.
    """
    obs = np.array(observations, dtype=np.float32, copy=True)
    if obs.ndim != 2 or obs.shape[1] % OBSERVATION_SIZE:
        raise ValueError("invalid stacked observation width")
    if source_format == 4 and CHECKPOINT_FORMAT == 5:
        frames = obs.reshape(len(obs), -1, OBSERVATION_SIZE)
        dx = frames[:, :, 24]
        left = frames[:, :, 30] > 0
        frames[:, :, 31] = ((frames[:, :, 22] > 0)
                           & (((dx > 0) & left) | ((dx < 0) & ~left)))
    elif source_format != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported demonstration format {source_format}")
    return obs


def discounted_returns(rewards, dones, episode_ids, gamma=GAMMA):
    """Monte Carlo targets, with an explicit terminal boundary for every fight."""
    result = np.zeros(len(rewards), dtype=np.float32)
    end = len(rewards) - 1
    for index in range(len(rewards) - 1, -1, -1):
        episode_end = index == end or episode_ids[index + 1] != episode_ids[index]
        if bool(dones[index]) != episode_end:
            raise ValueError("demonstrations must contain complete, contiguous episodes")
        future = 0.0 if episode_end else result[index + 1]
        result[index] = rewards[index] * REWARD_SCALE + gamma * future
    return result


def load_demonstrations(paths):
    observations, actions, episode_ids, returns = [], [], [], []
    episode_offset = 0
    width = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            current_width = int(data["observation_size"])
            source_format = int(data["format_version"])
            if (source_format not in (4, CHECKPOINT_FORMAT)
                    or list(data["action_names"]) != list(MACRO_ACTION_NAMES)):
                raise ValueError(f"{path}: demonstration format/action names mismatch")
            if str(data["policy"]) not in ("approach_aligned", "human"):
                raise ValueError(f"{path}: expected approach_aligned or human demonstrations")
            obs = migrate_observations(data["observations"], source_format)
            act = np.asarray(data["actions"], dtype=np.int64)
            episodes = np.asarray(data["episode_ids"], dtype=np.int32)
            rewards = np.asarray(data["rewards"], dtype=np.float32)
            dones = np.asarray(data["dones"], dtype=bool)
            if (obs.ndim != 2 or obs.shape[1] != current_width
                    or act.shape != (len(obs),) or episodes.shape != (len(obs),)
                    or len(obs) == 0 or not np.isfinite(obs).all()
                    or rewards.shape != (len(obs),) or dones.shape != (len(obs),)
                    or not np.isfinite(rewards).all()
                    or np.any(act < 0) or np.any(act >= len(MACRO_ACTION_NAMES))):
                raise ValueError(f"{path}: invalid demonstration arrays")
            if width is not None and current_width != width:
                raise ValueError("demonstration observation widths differ")
            width = current_width
            observations.append(obs)
            actions.append(act)
            episode_ids.append(episodes + episode_offset)
            returns.append(discounted_returns(rewards, dones, episodes))
            episode_offset += int(episodes.max()) + 1
    return (np.concatenate(observations), np.concatenate(actions),
            np.concatenate(episode_ids), np.concatenate(returns))


def main():
    args = parse_args()
    checkpoint_path = os.path.join(args.run_dir, "ppo_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        raise FileExistsError(f"{checkpoint_path} already exists; use a new run directory")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    observations, actions, episode_ids, returns = load_demonstrations(args.demos)
    distinct_episodes = np.unique(episode_ids)
    if len(distinct_episodes) < 3:
        raise ValueError("need at least three complete demonstration episodes")
    rng.shuffle(distinct_episodes)
    n_val = max(1, round(len(distinct_episodes) * 0.2))
    val_mask = np.isin(episode_ids, distinct_episodes[:n_val])
    train_mask = ~val_mask

    normalizer = RunningNorm(observations.shape[1])
    normalizer.update(observations[train_mask])
    train_obs = torch.as_tensor(normalizer.normalize(observations[train_mask]))
    val_obs = torch.as_tensor(normalizer.normalize(observations[val_mask]))
    train_actions = torch.as_tensor(actions[train_mask], dtype=torch.long)
    val_actions = torch.as_tensor(actions[val_mask], dtype=torch.long)

    model = ActorCritic(observations.shape[1], ACTION_SIZES,
                        hidden=MODEL_HIDDEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    counts = np.bincount(actions[train_mask], minlength=len(MACRO_ACTION_NAMES))
    weights = np.ones(len(MACRO_ACTION_NAMES), dtype=np.float32)
    present = counts > 0
    weights[present] = np.sqrt(counts[present].sum()
                               / (present.sum() * counts[present]))
    weights = np.minimum(weights, 3.0)
    weights[present] /= weights[present].mean()
    class_weights = torch.as_tensor(weights)

    best_loss = float("inf")
    best_model = None
    best_epoch = 0
    best_accuracy = 0.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_obs))
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            logits, _ = model(train_obs[indices])
            loss = F.cross_entropy(logits[0], train_actions[indices],
                                   weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(val_obs)
            val_loss = F.cross_entropy(logits[0], val_actions).item()
            val_accuracy = (logits[0].argmax(dim=-1) == val_actions).float().mean().item()
        print(f"epoch {epoch:2d}: val loss={val_loss:.4f} "
              f"accuracy={val_accuracy:.1%}", flush=True)
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 10:
                break

    os.makedirs(args.run_dir, exist_ok=True)
    model.load_state_dict(best_model)
    # Fit only the value head: changing the shared trunk here would silently
    # change the cloned action policy. PPO will later train the whole model.
    train_returns = torch.as_tensor(returns[train_mask])
    val_returns = torch.as_tensor(returns[val_mask])
    with torch.no_grad():
        train_features = model.trunk(train_obs).detach()
        val_features = model.trunk(val_obs).detach()
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=1e-3)
    best_critic_loss = float("inf")
    best_critic = copy.deepcopy(model.critic.state_dict())
    for _ in range(args.critic_epochs):
        order = torch.randperm(len(train_features))
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            predicted = model.critic(train_features[indices]).squeeze(-1)
            critic_loss = F.mse_loss(predicted, train_returns[indices])
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_optimizer.step()
        with torch.no_grad():
            critic_loss = F.mse_loss(model.critic(val_features).squeeze(-1), val_returns).item()
        if critic_loss < best_critic_loss:
            best_critic_loss = critic_loss
            best_critic = copy.deepcopy(model.critic.state_dict())
    model.critic.load_state_dict(best_critic)
    best_model = copy.deepcopy(model.state_dict())
    print(f"critic validation MSE={best_critic_loss:.4f}; actor weights preserved", flush=True)
    with torch.no_grad():
        best_logits, _ = model(val_obs)
        predictions = best_logits[0].argmax(dim=-1).numpy()
    validation_counts = np.bincount(val_actions.numpy(), minlength=len(MACRO_ACTION_NAMES))
    per_action = {
        name: {
            "train_count": int(counts[i]),
            "validation_count": int(validation_counts[i]),
            "validation_recall": (float((predictions[val_actions.numpy() == i] == i).mean())
                                  if validation_counts[i] else None),
        }
        for i, name in enumerate(MACRO_ACTION_NAMES)
    }
    # PPO has a different objective. Start its Adam moments from zero rather
    # than carrying cross-entropy momentum into the first policy update.
    ppo_optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT,
        "observation_size": observations.shape[1],
        "frame_stack": observations.shape[1] // OBSERVATION_SIZE,
        "action_names": list(MACRO_ACTION_NAMES),
        "action_sizes": ACTION_SIZES,
        "model_hidden": MODEL_HIDDEN,
        "num_envs": len(args.ports),
        "ports": list(args.ports),
        "steps_per_env": default_steps_per_env(len(args.ports)),
        "control_hz": 10,
        "model": best_model,
        "optimizer": ppo_optimizer.state_dict(),
        "obs_norm": normalizer.state_dict(),
        "freeze_obs_norm": True,
        "update": -1,
        "global_step": 0,
        "episode": 0,
        "pretraining": {
            "method": "behavior_cloning",
            "teacher": "approach_aligned",
            "demos": [os.path.abspath(path) for path in args.demos],
            "train_samples": int(train_mask.sum()),
            "validation_samples": int(val_mask.sum()),
            "train_episodes": int(len(distinct_episodes) - n_val),
            "validation_episodes": int(n_val),
            "best_epoch": best_epoch,
            "validation_loss": best_loss,
            "validation_accuracy": best_accuracy,
            "critic_validation_mse": best_critic_loss,
            "actions": per_action,
        },
    }
    temporary_path = checkpoint_path + ".tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    with open(os.path.join(args.run_dir, "bc_report.json"), "w", encoding="utf-8") as stream:
        json.dump(checkpoint["pretraining"], stream, indent=2)
    print(f"saved {checkpoint_path}; {train_mask.sum()} training samples, "
          f"{val_mask.sum()} validation samples, best epoch {best_epoch}, "
          f"accuracy {best_accuracy:.1%}", flush=True)


if __name__ == "__main__":
    main()
