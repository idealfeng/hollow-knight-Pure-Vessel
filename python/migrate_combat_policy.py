"""Expand a full spell PPO checkpoint with zero-initialized combat inputs.

Each frame keeps its original 118 columns, followed by the new combat fields.
Actor, critic, Adam moments and fitted observation statistics retain their old
values. The new columns initially contribute nothing to the policy or value.
"""

import argparse
import copy
from dataclasses import asdict
import hashlib
import io
import math
from pathlib import Path

import numpy as np
import torch

from agent.action import SPELL_ACTION_NAMES
from agent.reward import get_reward_config
from observation_schema import (COMBAT_EXTRA_FEATURES, COMBAT_FRAME_WIDTH,
                                COMBAT_SCHEMA, LEGACY_FRAME_WIDTH, LEGACY_SCHEMA)
from ppo_model import ActorCritic


def old_column_indices(frame_stack):
    """Indices in the new stacked vector corresponding to old stacked columns."""
    return np.concatenate([np.arange(index * COMBAT_FRAME_WIDTH,
                                    index * COMBAT_FRAME_WIDTH + LEGACY_FRAME_WIDTH)
                           for index in range(frame_stack)])


def _expand_columns(tensor, indices, new_width):
    result = tensor.new_zeros((tensor.shape[0], new_width))
    result[:, torch.as_tensor(indices, dtype=torch.long)] = tensor
    return result


def migrate_checkpoint(source_path, destination_path):
    """Exclusively create a compatible larger checkpoint; never modify source."""
    source_path, destination_path = Path(source_path), Path(destination_path)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("choose a new destination; the source must remain unchanged")
    if destination_path.exists():
        raise FileExistsError(f"destination already exists: {destination_path}")
    payload = source_path.read_bytes()
    source = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    stack = source.get("frame_stack")
    if (source.get("format_version") != 9
            or source.get("action_distribution") != "exclusive_directions_spells_v1"
            or source.get("action_names") != list(SPELL_ACTION_NAMES)
            or source.get("action_sizes") != [1] * 7 + [4]
            or source.get("observation_schema", LEGACY_SCHEMA) != LEGACY_SCHEMA
            or source.get("training_config", {}).get("observation_schema", LEGACY_SCHEMA) != LEGACY_SCHEMA
            or not isinstance(stack, int) or stack < 1
            or source.get("observation_size") != stack * LEGACY_FRAME_WIDTH
            or source.get("initialization_only", False)
            or "optimizer" not in source):
        raise ValueError("source must be a full legacy-observation format-9 spell checkpoint")
    old_width, new_width = stack * LEGACY_FRAME_WIDTH, stack * COMBAT_FRAME_WIDTH
    indices = old_column_indices(stack)
    normalizer = source.get("obs_norm", {})
    mean, variance = np.asarray(normalizer.get("mean")), np.asarray(normalizer.get("var"))
    count = normalizer.get("count", 0)
    if (mean.shape != (old_width,) or variance.shape != (old_width,)
            or not np.isfinite(mean).all() or not np.isfinite(variance).all()
            or np.any(variance < 0) or not math.isfinite(count) or count <= 0):
        raise ValueError("source contains an invalid fitted observation normalizer")
    if any(not isinstance(source.get(key), int) or source[key] < 0
           for key in ("update", "global_step", "episode")):
        raise ValueError("source must retain completed PPO update/sample/episode counters")

    # Instantiate inside fork_rng so verification has no effect on the caller's
    # sampling stream. Strict loading also verifies every actor/critic shape.
    with torch.random.fork_rng(devices=[]):
        old_model = ActorCritic(old_width, [1] * 7 + [4], hidden=source["model_hidden"],
                                bernoulli=True, exclusive_directions=True)
    try:
        old_model.load_state_dict(source["model"], strict=True)
    except (RuntimeError, KeyError) as error:
        raise ValueError("source model does not match its declared architecture") from error
    names, parameters = zip(*old_model.named_parameters())
    optimizer = source["optimizer"]
    groups, states = optimizer.get("param_groups", []), optimizer.get("state", {})
    if (len(groups) != 1 or len(groups[0].get("params", [])) != len(parameters)
            or len(set(groups[0]["params"])) != len(parameters)
            or not set(states).issubset(set(groups[0]["params"]))):
        raise ValueError("source Adam parameter layout is not the full ordered model group")
    ids = groups[0]["params"]
    if groups[0].get("param_names") is not None and list(groups[0]["param_names"]) != list(names):
        raise ValueError("source Adam parameter names do not match the model order")
    for identifier, parameter in zip(ids, parameters):
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            moment = states.get(identifier, {}).get(key)
            if moment is not None and (not torch.is_tensor(moment) or moment.shape != parameter.shape):
                raise ValueError(f"source Adam moment {key} has an incompatible shape")

    migrated = copy.deepcopy(source)
    migrated["model"]["trunk.0.weight"] = _expand_columns(source["model"]["trunk.0.weight"], indices, new_width)
    first_id = ids[list(names).index("trunk.0.weight")]
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        moment = source["optimizer"]["state"].get(first_id, {}).get(key)
        if moment is not None:
            migrated["optimizer"]["state"][first_id][key] = _expand_columns(moment, indices, new_width)
    new_mean, new_variance = np.zeros(new_width, dtype=mean.dtype), np.ones(new_width, dtype=variance.dtype)
    new_mean[indices], new_variance[indices] = mean, variance
    migrated["obs_norm"]["mean"], migrated["obs_norm"]["var"] = new_mean, new_variance
    source_config = copy.deepcopy(source.get("training_config", {}))
    migrated.update(observation_schema=COMBAT_SCHEMA, observation_size=new_width,
                    observation_frame_size=COMBAT_FRAME_WIDTH,
                    combat_extra_features=list(COMBAT_EXTRA_FEATURES), freeze_obs_norm=True)
    migrated["training_config"] = {
        **source_config, "observation_schema": COMBAT_SCHEMA, "observation_size": new_width,
        "reward_profile": "radiant_potential_v1",
        "reward_config": asdict(get_reward_config("radiant_potential_v1")),
        "freeze_obs_norm": True, "init_source": str(source_path.resolve()),
        "start_update": source["update"] + 1,
    }
    migration = {
        "kind": "combat_observation_expansion", "source_checkpoint": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(payload).hexdigest(), "source_update": source["update"],
        "source_global_step": source["global_step"], "source_episode": source["episode"],
        "source_observation_schema": LEGACY_SCHEMA, "source_observation_size": old_width,
        "destination_observation_schema": COMBAT_SCHEMA, "destination_observation_size": new_width,
        "previous_training_config": source_config,
        "previous_evaluation": copy.deepcopy(source.get("evaluation")),
        "previous_pretraining": copy.deepcopy(source.get("pretraining")),
        "actor_and_critic_old_columns_preserved": True,
        "new_input_weights_zero": True, "optimizer_reset": False, "critic_reset": False,
        "observation_statistics_old_columns_preserved": True,
        "new_normalizer_mean": 0.0, "new_normalizer_variance": 1.0,
    }
    migrated["pretraining"] = copy.deepcopy(migration)
    migrated["combat_migration"] = migration
    # The source evaluation remains in provenance; the new reward/config must
    # accumulate its own assessed best rather than inherit a misleading score.
    migrated.pop("evaluation", None)
    if source_path.read_bytes() != payload:
        raise RuntimeError("source checkpoint changed during migration")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("xb") as stream:
        torch.save(migrated, stream)
    if source_path.read_bytes() != payload:
        raise RuntimeError("source checkpoint changed during migration")
    return migrated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = migrate_checkpoint(args.source, args.destination)
    print(f"Saved {result['observation_schema']} full policy: {args.destination}")
    print(f"Preserved update {result['update']}, critic, Adam and old observation columns; "
          f"next update {result['update'] + 1}, input width {result['observation_size']}.")


if __name__ == "__main__":
    main()
