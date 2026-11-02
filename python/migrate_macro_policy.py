"""Initialize a 35-choice, sixteen-frame policy from a full spell checkpoint.

The backbone and critic are preserved. The new action probabilities equal the
old factorized policy conditioned on the 35 *canonical button vectors*. This
does not preserve its unrestricted policy or sum all equivalent cast-aim
requests. Adam is deliberately restarted for every parameter; learned weights,
normalization and completed environment/update counters are retained.
"""

import argparse
import copy
from dataclasses import asdict
import hashlib
import io
from pathlib import Path

import numpy as np
import torch

from agent.action import EXECUTION_PROFILES
from agent.reward import get_reward_config
from macro_actions import (MACRO_ACTIONS, MACRO_DISTRIBUTION,
                           MACRO_EXECUTION_PROFILE, NUM_MACRO_ACTIONS,
                           macro_metadata, validate_macro_checkpoint)
from migrate_frame_stack import (_expand_columns, _validate_source,
                                 source_column_indices)
from observation_schema import LEGACY_FRAME_WIDTH, LEGACY_SCHEMA
from ppo_model import ActorCritic


TARGET_FRAME_STACK = 16
TARGET_REWARD_PROFILE = "radiant_spells_dodge_v1"


def canonical_head_parameters(source_model):
    """Analytically restrict independent old logits to canonical macro IDs.

Old joint log probabilities share one state-dependent denominator across
all table rows: two direction categoricals, three Bernoullis and the spell
categorical. It cancels upon conditioning, so summing selected logits is exact.
    """
    weights, biases = [], []
    for macro in MACRO_ACTIONS:
        buttons = macro.buttons
        weight = torch.zeros_like(source_model["heads.0.weight"][0])
        bias = torch.zeros_like(source_model["heads.0.bias"][0])
        for index in range(7):
            if buttons[index]:
                weight = weight + source_model[f"heads.{index}.weight"][0]
                bias = bias + source_model[f"heads.{index}.bias"][0]
        weight = weight + source_model["heads.7.weight"][buttons[7]]
        bias = bias + source_model["heads.7.bias"][buttons[7]]
        weights.append(weight)
        biases.append(bias)
    return torch.stack(weights), torch.stack(biases)


def _fresh_adam_state(model, source_optimizer):
    """Keep Adam hyperparameters, reset every moment and per-parameter step."""
    optimizer = torch.optim.Adam(model.parameters())
    state = optimizer.state_dict()
    previous_group = copy.deepcopy(source_optimizer["param_groups"][0])
    previous_group.pop("params")
    had_names = previous_group.pop("param_names", None) is not None
    state["param_groups"][0].update(previous_group)
    if had_names:
        state["param_groups"][0]["param_names"] = [name for name, _ in model.named_parameters()]
    # Empty state is Adam's standard lazy, step-zero initialization.
    return state


def migrate_checkpoint(source_path, destination_path, *, execution_profile=MACRO_EXECUTION_PROFILE):
    """Exclusively create a new format-10 policy without changing the source."""
    if execution_profile not in EXECUTION_PROFILES:
        raise ValueError(f"unknown execution profile: {execution_profile!r}")
    source_path, destination_path = Path(source_path), Path(destination_path)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("choose a new destination; the source must remain unchanged")
    if destination_path.exists():
        raise FileExistsError(f"destination already exists: {destination_path}")
    payload = source_path.read_bytes()
    source = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    schema, mean, variance, _, _ = _validate_source(source)
    if schema != LEGACY_SCHEMA:
        raise ValueError("macro migration requires the legacy four-frame source, without new telemetry inputs")
    previous_config = copy.deepcopy(source["training_config"])
    expected_reward = asdict(get_reward_config(TARGET_REWARD_PROFILE))
    if (previous_config.get("reward_profile") != TARGET_REWARD_PROFILE
            or previous_config.get("reward_config") != expected_reward):
        raise ValueError("macro migration keeps radiant_spells_dodge_v1 rewards unchanged")
    frame_width, new_width = LEGACY_FRAME_WIDTH, LEGACY_FRAME_WIDTH * TARGET_FRAME_STACK
    indices = source_column_indices(LEGACY_SCHEMA, LEGACY_SCHEMA)
    with torch.random.fork_rng(devices=[]):
        model = ActorCritic(new_width, [NUM_MACRO_ACTIONS], hidden=256,
                            bernoulli=False, exclusive_directions=False)
    model_state = {name: value.clone() for name, value in source["model"].items()
                   if not name.startswith("heads.")}
    model_state["trunk.0.weight"] = _expand_columns(source["model"]["trunk.0.weight"], indices, new_width)
    model_state["heads.0.weight"], model_state["heads.0.bias"] = canonical_head_parameters(source["model"])
    model.load_state_dict(model_state, strict=True)

    migrated = copy.deepcopy(source)
    new_mean, new_variance = np.empty(new_width, dtype=mean.dtype), np.empty(new_width, dtype=variance.dtype)
    for frame in range(TARGET_FRAME_STACK - 4):
        start = frame * frame_width
        new_mean[start:start + frame_width] = mean[:frame_width]
        new_variance[start:start + frame_width] = variance[:frame_width]
    new_mean[indices], new_variance[indices] = mean, variance
    migrated["obs_norm"]["mean"], migrated["obs_norm"]["var"] = new_mean, new_variance
    action_metadata = macro_metadata()
    migrated.update(
        format_version=10, action_distribution=MACRO_DISTRIBUTION,
        action_names=["macro"], action_sizes=[NUM_MACRO_ACTIONS],
        model=model_state, optimizer=_fresh_adam_state(model, source["optimizer"]),
        frame_stack=TARGET_FRAME_STACK, observation_schema=LEGACY_SCHEMA,
        observation_size=new_width, observation_frame_size=frame_width,
        freeze_obs_norm=True, execution_profile=execution_profile,
        reward_profile=TARGET_REWARD_PROFILE, **action_metadata,
    )
    migrated["training_config"] = {
        **previous_config, "format_version": 10,
        "action_distribution": MACRO_DISTRIBUTION, "action_names": ["macro"],
        "action_sizes": [NUM_MACRO_ACTIONS], **copy.deepcopy(action_metadata),
        "frame_stack": TARGET_FRAME_STACK, "observation_schema": LEGACY_SCHEMA,
        "observation_size": new_width, "observation_frame_size": frame_width,
        "freeze_obs_norm": True, "execution_profile": execution_profile,
        "reward_profile": TARGET_REWARD_PROFILE, "reward_config": expected_reward,
        "init_source": str(source_path.resolve()), "start_update": source["update"] + 1,
        "train_mode": "all",
    }
    if "inference_config" in migrated:
        migrated["inference_config"].update(
            frame_stack=TARGET_FRAME_STACK, observation_schema=LEGACY_SCHEMA,
            observation_size=new_width, action_distribution=MACRO_DISTRIBUTION,
            action_names=["macro"], action_sizes=[NUM_MACRO_ACTIONS],
            execution_profile=execution_profile, reward_profile=TARGET_REWARD_PROFILE,
            **copy.deepcopy(action_metadata),
        )
    provenance = {
        "kind": "canonical_macro_policy_initialization",
        "source_checkpoint": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_update": source["update"], "source_global_step": source["global_step"],
        "source_episode": source["episode"],
        "previous_training_config": previous_config,
        "previous_evaluation": copy.deepcopy(source.get("evaluation")),
        "previous_pretraining": copy.deepcopy(source.get("pretraining")),
        "previous_inference_config": copy.deepcopy(source.get("inference_config")),
        "source_observation_size": source["observation_size"],
        "destination_observation_size": new_width,
        "source_frame_stack": 4, "destination_frame_stack": TARGET_FRAME_STACK,
        "recent_frame_offset": TARGET_FRAME_STACK - 4,
        "backbone_and_critic_recent_columns_preserved": True,
        "new_history_input_weights_zero": True,
        "new_history_normalization": "source_oldest_frame_legacy_features",
        "action_head_initialization": "old_joint_probability_conditioned_on_canonical_macro_vectors",
        "unrestricted_source_policy_preserved": False,
        "effective_aim_preimages_marginalized": False,
        "optimizer_reset": True, "optimizer_reset_scope": "all_parameters",
        "optimizer_reset_reason": "new action head and expanded input must begin Adam at step zero",
        "optimizer_hyperparameters_preserved": True,
        "critic_reset": False, "reward_changed": False,
        "source_evaluation_not_inherited": True,
        "source_execution_profile": source.get("execution_profile", previous_config.get("execution_profile", "legacy_v1")),
        "destination_execution_profile": execution_profile,
        **copy.deepcopy(action_metadata),
    }
    migrated["pretraining"] = copy.deepcopy(provenance)
    migrated["macro_migration"] = provenance
    # A score obtained by the old action space is not a score of the new one.
    for key in ("evaluation", "best_evaluation", "best_eval", "best_eval_score"):
        migrated.pop(key, None)
    validate_macro_checkpoint(migrated)
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
    parser.add_argument("--execution-profile", choices=EXECUTION_PROFILES,
                        default=MACRO_EXECUTION_PROFILE)
    args = parser.parse_args()
    result = migrate_checkpoint(args.source, args.destination, execution_profile=args.execution_profile)
    print(f"Saved optional {NUM_MACRO_ACTIONS}-choice macro policy: {args.destination}")
    print(f"Input {result['observation_size']}, sixteen frames; backbone/critic retained, "
          f"Adam restarted, next update {result['update'] + 1}. No training started.")


if __name__ == "__main__":
    main()
