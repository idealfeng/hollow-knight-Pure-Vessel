"""Compare two saved policies on the same held-out human fight."""

import argparse
from pathlib import Path

import numpy as np
import torch

from agent.action import BUTTON_ORDER, NUM_BUTTONS
from obs_norm import RunningNorm
from ppo_model import ActorCritic
from train_bc import button_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demos", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    args = parser.parse_args()
    data = np.load(args.demos)
    observations = data["observations"]
    actions = data["actions"].astype(bool)
    episode_ids = data["episode_ids"]
    episodes = np.unique(episode_ids)
    np.random.default_rng(42).shuffle(episodes)
    val_mask = np.isin(episode_ids, episodes[:max(1, round(len(episodes) * 0.2))])
    print(f"held-out episode {episodes[0]}, {val_mask.sum()} samples")

    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = ActorCritic(
            observations.shape[1], [1] * NUM_BUTTONS,
            hidden=checkpoint["model_hidden"], bernoulli=True,
            exclusive_directions=True,
        )
        model.load_state_dict(checkpoint["model"])
        model.eval()
        normalizer = RunningNorm(observations.shape[1])
        normalizer.load_state_dict(checkpoint["obs_norm"])
        with torch.no_grad():
            logits, _ = model(torch.as_tensor(normalizer.normalize(observations[val_mask])))
            predicted = button_predictions(logits).numpy()
        target = actions[val_mask]
        recall = {
            name: round(float(predicted[target[:, i], i].mean()), 3)
            if target[:, i].any() else None
            for i, name in enumerate(BUTTON_ORDER)
        }
        print(path, "update", checkpoint["update"],
              "accuracy", round(float((predicted == target).mean()), 3),
              "predicted_on", dict(zip(BUTTON_ORDER, predicted.sum(axis=0).tolist())),
              "recall", recall)


if __name__ == "__main__":
    main()
