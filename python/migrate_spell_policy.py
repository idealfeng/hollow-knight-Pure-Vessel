"""Initialize a new spell-enabled run from a frozen seven-button checkpoint.

The source file is read only. Actor and observation normalization tensors are
copied exactly; the new spell head starts with 91% no spell and 3% per spell.
The critic and optimizer are reset because the new objective and loadout differ.
"""

import argparse
import copy
import hashlib
import io
import math
from pathlib import Path

import numpy as np
import torch

from obs_norm import RunningNorm
from ppo_model import ActorCritic


BASE_ACTION_NAMES = ("left", "right", "up", "down", "attack", "jump", "dash")
SPELL_ACTION_NAMES = BASE_ACTION_NAMES + ("spell",)
SPELL_CHOICES = ("none", "shade_soul", "abyss_shriek", "descending_dark")
SPELL_ACTION_SIZES = [1] * 7 + [4]
SPELL_CHECKPOINT_FORMAT = 9
SPELL_ACTION_DISTRIBUTION = "exclusive_directions_spells_v1"


def migrate_checkpoint(source_path, destination_path, *, spell_probability=0.03, seed=42):
    """Write a fresh-run initializer while preserving the old actor exactly."""
    source_path, destination_path = Path(source_path), Path(destination_path)
    if not math.isfinite(spell_probability) or not 0 < spell_probability < 0.25:
        raise ValueError("each spell probability must be positive and less than 0.25")
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("choose a new destination; the source must remain unchanged")
    if destination_path.exists():
        raise FileExistsError(f"destination already exists: {destination_path}")

    payload = source_path.read_bytes()
    source = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if (source.get("format_version") != 8
            or source.get("action_distribution") != "exclusive_directions_v1"
            or source.get("action_names") != list(BASE_ACTION_NAMES)
            or source.get("action_sizes") != [1] * 7):
        raise ValueError("source must be a format-8 exclusive seven-button checkpoint")

    observation_size = int(source["observation_size"])
    normalizer = RunningNorm(observation_size)
    normalizer.load_state_dict(source["obs_norm"])
    if (normalizer.mean.shape != (observation_size,)
            or normalizer.var.shape != (observation_size,)
            or not np.isfinite(normalizer.mean).all()
            or not np.isfinite(normalizer.var).all()
            or np.any(normalizer.var < 0)
            or not math.isfinite(normalizer.count) or normalizer.count <= 0):
        raise ValueError("source contains an invalid observation normalizer")

    # Forking the RNG keeps this utility from changing a caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ActorCritic(observation_size, SPELL_ACTION_SIZES,
                            hidden=int(source["model_hidden"]), bernoulli=True,
                            exclusive_directions=True)
    old_actor = {name: value for name, value in source["model"].items()
                 if not name.startswith("critic.")}
    incompatible = model.load_state_dict(old_actor, strict=False)
    if (set(incompatible.missing_keys) != {
            "heads.7.weight", "heads.7.bias", "critic.weight", "critic.bias"}
            or incompatible.unexpected_keys):
        raise ValueError(f"source actor does not match the new model: {incompatible}")
    with torch.no_grad():
        model.heads[7].weight.zero_()
        model.heads[7].bias.copy_(torch.tensor([
            0.0, *([math.log(spell_probability / (1 - 3 * spell_probability))] * 3),
        ]))

    initialized = {
        "format_version": SPELL_CHECKPOINT_FORMAT,
        "observation_size": observation_size,
        "frame_stack": int(source["frame_stack"]),
        "action_names": list(SPELL_ACTION_NAMES),
        "action_sizes": list(SPELL_ACTION_SIZES),
        "action_distribution": SPELL_ACTION_DISTRIBUTION,
        "spell_choices": list(SPELL_CHOICES),
        "model_hidden": int(source["model_hidden"]),
        "control_hz": int(source.get("control_hz", 10)),
        "model": model.state_dict(),
        "obs_norm": copy.deepcopy(source["obs_norm"]),
        "freeze_obs_norm": True,
        "initialization_only": True,
        "update": -1, "global_step": 0, "episode": 0,
        "pretraining": {
            "kind": "seven_button_actor_transfer",
            "source_checkpoint": str(source_path.resolve()),
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "source_format_version": source["format_version"],
            "source_update": int(source.get("update", -1)),
            "source_global_step": int(source.get("global_step", 0)),
            "actor_copied_exactly": True,
            "observation_normalizer_copied_exactly": True,
            "critic_reset": True, "optimizer_reset": True,
            "spell_initial_probabilities": [1 - 3 * spell_probability] + [spell_probability] * 3,
            "seed": int(seed),
        },
        "inference_config": {
            "boss_level": 2, "episode_time_limit": 120.0,
            "observation_time_limit": 60.0, "game_speed": 1.0,
            "reward_profile": "radiant_spells_v1", "action_selection": "greedy",
            "nail_damage": 21, "combat_charms": [],
        },
    }
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects previous experiment state even under a race.
    with destination_path.open("xb") as stream:
        torch.save(initialized, stream)
    if source_path.read_bytes() != payload:
        raise RuntimeError("source checkpoint changed during migration")
    return initialized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--spell-probability", type=float, default=0.03,
                        help="initial probability for each of the three spells")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    initialized = migrate_checkpoint(args.source, args.destination,
                                     spell_probability=args.spell_probability, seed=args.seed)
    print(f"Saved spell-enabled initialization: {args.destination}")
    print(f"Copied actor from update {initialized['pretraining']['source_update']}; "
          "critic and optimizer reset; old checkpoint unchanged.")


if __name__ == "__main__":
    main()
