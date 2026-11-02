"""Expand a complete four-frame spell policy to sixteen frames without reset.

The newest four frames retain the source policy's input columns. New history
starts with zero input weights and Adam moments, but uses fitted statistics
from the source's oldest frame so it can subsequently learn at a useful scale.
The default keeps legacy observations; combat telemetry is an explicit option.
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
                                COMBAT_SCHEMA, LEGACY_FRAME_WIDTH, LEGACY_SCHEMA,
                                observation_width)
from ppo_model import ActorCritic


SOURCE_FRAME_STACK = 4
TARGET_FRAME_STACK = 16
TARGET_REWARD_PROFILE = "radiant_spells_dodge_v1"
ACTION_SIZES = [1] * 7 + [4]


def source_column_indices(source_schema=LEGACY_SCHEMA, target_schema=LEGACY_SCHEMA):
    """Map oldest-to-newest source frames to the newest four target frames."""
    source_width, target_width = observation_width(source_schema), observation_width(target_schema)
    if source_width > target_width:
        raise ValueError("cannot discard source observation fields during a preserving migration")
    offset = TARGET_FRAME_STACK - SOURCE_FRAME_STACK
    return np.concatenate([np.arange((offset + frame) * target_width,
                                     (offset + frame) * target_width + source_width)
                           for frame in range(SOURCE_FRAME_STACK)])


def _expand_columns(tensor, indices, width):
    expanded = tensor.new_zeros((tensor.shape[0], width))
    expanded[:, torch.as_tensor(indices, dtype=torch.long)] = tensor
    return expanded


def _validate_source(source):
    """Require a fitted, fully trained 21-damage spell policy and full Adam."""
    if not isinstance(source, dict):
        raise ValueError("source must contain a checkpoint dictionary")
    config = source.get("training_config", {})
    if not isinstance(config, dict):
        raise ValueError("source must contain its complete training configuration")
    schema = source.get("observation_schema", LEGACY_SCHEMA)
    if schema not in (LEGACY_SCHEMA, COMBAT_SCHEMA):
        raise ValueError("source observation schema is unsupported")
    width = observation_width(schema) * SOURCE_FRAME_STACK
    profile = config.get("reward_profile")
    if profile not in ("radiant_spells_v1", TARGET_REWARD_PROFILE, "radiant_potential_v1"):
        raise ValueError("source must use a verified 21-damage spell reward profile")
    expected_reward = asdict(get_reward_config(profile))
    if (source.get("format_version") != 9
            or source.get("action_distribution") != "exclusive_directions_spells_v1"
            or source.get("action_names") != list(SPELL_ACTION_NAMES)
            or source.get("action_sizes") != ACTION_SIZES
            or source.get("model_hidden") != 256
            or source.get("frame_stack") != SOURCE_FRAME_STACK
            or source.get("observation_size") != width
            or source.get("control_hz") != 10
            or source.get("freeze_obs_norm") is not True
            or config.get("freeze_obs_norm", True) is not True
            or source.get("initialization_only", False)
            or config.get("observation_schema", schema) != schema
            or config.get("frame_stack", SOURCE_FRAME_STACK) != SOURCE_FRAME_STACK
            or config.get("observation_size", width) != width
            or config.get("reward_config") != expected_reward
            or expected_reward["nail_damage_reference"] != 21.0
            or config.get("boss_level") != 2
            or config.get("episode_time_limit") not in (120, 150)
            or config.get("observation_time_limit") != 60
            or config.get("game_speed") != 1
            or config.get("train_mode") != "all"
            or not isinstance(source.get("optimizer"), dict)):
        raise ValueError("source must be a full frozen 21-damage format-9 four-frame spell checkpoint")
    for metadata in (source, config, source.get("inference_config", {})):
        if not isinstance(metadata, dict):
            raise ValueError("source loadout metadata is invalid")
        if metadata.get("nail_damage", 21) != 21 or metadata.get("combat_charms", []) != []:
            raise ValueError("source loadout must use 21 nail damage without combat charms")
    if any(type(source.get(key)) is not int or source[key] < 0
           for key in ("update", "global_step", "episode")):
        raise ValueError("source must retain completed PPO update/sample/episode counters")
    normalizer = source.get("obs_norm", {})
    if not isinstance(normalizer, dict):
        raise ValueError("source contains an invalid fitted observation normalizer")
    mean = np.asarray(normalizer.get("mean"))
    variance = np.asarray(normalizer.get("var"))
    try:
        valid_norm = (mean.shape == (width,) and variance.shape == (width,)
                      and np.issubdtype(mean.dtype, np.floating)
                      and np.issubdtype(variance.dtype, np.floating)
                      and np.isfinite(mean).all() and np.isfinite(variance).all()
                      and np.all(variance >= 0)
                      and math.isfinite(normalizer.get("count", 0))
                      and normalizer.get("count", 0) > 0
                      and math.isfinite(normalizer.get("clip", 10))
                      and normalizer.get("clip", 10) > 0)
    except (TypeError, ValueError):
        valid_norm = False
    if not valid_norm:
        raise ValueError("source contains an invalid fitted observation normalizer")

    # Keep the caller's sampling RNG unchanged while checking the architecture.
    with torch.random.fork_rng(devices=[]):
        model = ActorCritic(width, ACTION_SIZES, hidden=256,
                            bernoulli=True, exclusive_directions=True)
    try:
        model.load_state_dict(source["model"], strict=True)
    except (RuntimeError, KeyError, TypeError) as error:
        raise ValueError("source model does not match its declared architecture") from error
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("source model contains nonfinite parameters")
    names, parameters = zip(*model.named_parameters())
    groups = source["optimizer"].get("param_groups", [])
    states = source["optimizer"].get("state", {})
    if (len(groups) != 1 or len(groups[0].get("params", [])) != len(parameters)
            or len(set(groups[0]["params"])) != len(parameters)
            or set(states) != set(groups[0]["params"])):
        raise ValueError("source Adam must retain the full ordered model parameter group and states")
    identifiers = groups[0]["params"]
    if groups[0].get("param_names") is not None and list(groups[0]["param_names"]) != list(names):
        raise ValueError("source Adam parameter names do not match the model order")
    for identifier, parameter in zip(identifiers, parameters):
        state = states[identifier]
        required = ("exp_avg", "exp_avg_sq") + (("max_exp_avg_sq",) if groups[0].get("amsgrad") else ())
        for key in set(required) | ({"max_exp_avg_sq"} & set(state)):
            value = state.get(key)
            if (not torch.is_tensor(value) or value.shape != parameter.shape
                    or not torch.isfinite(value).all()):
                raise ValueError(f"source Adam moment {key} has an invalid shape or value")
        try:
            step = float(state["step"])
            if not math.isfinite(step) or step < 1 or not step.is_integer():
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ValueError("source Adam must retain valid completed step counters") from None
    return schema, mean, variance, names, identifiers


def migrate_checkpoint(source_path, destination_path, *, target_schema=LEGACY_SCHEMA):
    """Exclusively create a 16-frame checkpoint; the source is never written."""
    if target_schema not in (LEGACY_SCHEMA, COMBAT_SCHEMA):
        raise ValueError("target schema must be legacy_v1 or combat_v2")
    source_path, destination_path = Path(source_path), Path(destination_path)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("choose a new destination; the source must remain unchanged")
    if destination_path.exists():
        raise FileExistsError(f"destination already exists: {destination_path}")
    payload = source_path.read_bytes()
    source = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    schema, mean, variance, names, identifiers = _validate_source(source)
    indices = source_column_indices(schema, target_schema)
    frame_width = observation_width(target_schema)
    new_width = frame_width * TARGET_FRAME_STACK
    migrated = copy.deepcopy(source)
    migrated["model"]["trunk.0.weight"] = _expand_columns(source["model"]["trunk.0.weight"], indices, new_width)
    first_id = identifiers[list(names).index("trunk.0.weight")]
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        if key in source["optimizer"]["state"][first_id]:
            migrated["optimizer"]["state"][first_id][key] = _expand_columns(
                source["optimizer"]["state"][first_id][key], indices, new_width)

    new_mean = np.zeros(new_width, dtype=mean.dtype)
    new_variance = np.ones(new_width, dtype=variance.dtype)
    for frame in range(TARGET_FRAME_STACK - SOURCE_FRAME_STACK):
        start = frame * frame_width
        new_mean[start:start + LEGACY_FRAME_WIDTH] = mean[:LEGACY_FRAME_WIDTH]
        new_variance[start:start + LEGACY_FRAME_WIDTH] = variance[:LEGACY_FRAME_WIDTH]
    new_mean[indices], new_variance[indices] = mean, variance
    migrated["obs_norm"]["mean"], migrated["obs_norm"]["var"] = new_mean, new_variance
    previous_config = copy.deepcopy(source["training_config"])
    migrated.update(frame_stack=TARGET_FRAME_STACK, observation_schema=target_schema,
                    observation_size=new_width, observation_frame_size=frame_width)
    if target_schema == COMBAT_SCHEMA:
        migrated["combat_extra_features"] = list(COMBAT_EXTRA_FEATURES)
    migrated["training_config"] = {
        **previous_config, "frame_stack": TARGET_FRAME_STACK,
        "observation_schema": target_schema, "observation_size": new_width,
        "observation_frame_size": frame_width, "reward_profile": TARGET_REWARD_PROFILE,
        "reward_config": asdict(get_reward_config(TARGET_REWARD_PROFILE)),
        "init_source": str(source_path.resolve()), "start_update": source["update"] + 1,
    }
    if "reward_profile" in migrated:
        migrated["reward_profile"] = TARGET_REWARD_PROFILE
    if "inference_config" in migrated:
        migrated["inference_config"].update(frame_stack=TARGET_FRAME_STACK,
                                            observation_schema=target_schema,
                                            observation_size=new_width,
                                            reward_profile=TARGET_REWARD_PROFILE)
    migration = {
        "kind": "frame_stack_expansion", "source_checkpoint": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_update": source["update"], "source_global_step": source["global_step"],
        "source_episode": source["episode"], "source_frame_stack": SOURCE_FRAME_STACK,
        "destination_frame_stack": TARGET_FRAME_STACK,
        "source_observation_schema": schema, "destination_observation_schema": target_schema,
        "source_observation_size": source["observation_size"], "destination_observation_size": new_width,
        "previous_training_config": previous_config,
        "previous_evaluation": copy.deepcopy(source.get("evaluation")),
        "previous_pretraining": copy.deepcopy(source.get("pretraining")),
        "actor_and_critic_recent_columns_preserved": True,
        "recent_frame_offset": TARGET_FRAME_STACK - SOURCE_FRAME_STACK,
        "new_input_weights_zero": True, "new_adam_moments_zero": True,
        "new_history_normalization": "source_oldest_frame_legacy_features",
        "new_combat_normalizer_mean": 0.0, "new_combat_normalizer_variance": 1.0,
        "optimizer_reset": False, "critic_reset": False,
        "source_evaluation_not_inherited": True,
    }
    migrated["pretraining"] = copy.deepcopy(migration)
    migrated["frame_stack_migration"] = migration
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
    parser.add_argument("--target-schema", choices=(LEGACY_SCHEMA, COMBAT_SCHEMA), default=LEGACY_SCHEMA)
    args = parser.parse_args()
    result = migrate_checkpoint(args.source, args.destination, target_schema=args.target_schema)
    print(f"Saved full {result['observation_schema']} policy: {args.destination}")
    print(f"Input {result['observation_size']}, sixteen frames; newest four frames, critic, "
          f"Adam and update {result['update']} preserved. Reward: {TARGET_REWARD_PROFILE}.")


if __name__ == "__main__":
    main()
