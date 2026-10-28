"""Convert recorded human demonstrations into the NPZ format train_bc.py reads.

A human session records 60 FPS raw button bitmasks plus the same state protocol
the trainer reads (see HUMAN_DEMONSTRATIONS.md). ``train_bc.py`` needs 10 Hz
stacked observations paired with one of the 12 macro-action labels. This script
bridges the two:

  * it uses the *corrected* episode boundaries from ``episodes_verified.jsonl``
    (the raw ``frames.jsonl`` ``episode`` tag came from the old, buggy recorder);
  * it downsamples each fight from 60 FPS to the 10 Hz control rate;
  * it maps the held-button bitmask to the closest 12-way macro action;
  * it stacks four consecutive frames and recomputes per-step rewards;
  * it writes one NPZ in the same shape ``diagnose_checkpoint.py --save-demos``
    produces, so ``train_bc.py`` can consume it unchanged.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from agent.action import MACRO_ACTION_NAMES, MacroAction
from agent.reward import calculate_reward
from enviroment.state_to_vector import OBSERVATION_SIZE, state_to_vector
from game_state import _parse_frame
from human_demo import BUTTON_NAMES
from main_multi import CHECKPOINT_FORMAT

CONTROL_HZ = 10
FRAME_STACK = 4
EPISODE_TIME_LIMIT = 60.0
STEP_DT = 1.0 / CONTROL_HZ

# Bit positions in the recorded button mask (see human_demo.BUTTON_NAMES).
LEFT, RIGHT, UP, DOWN, ATTACK, JUMP, DASH = 0, 1, 2, 3, 4, 5, 6

POLICY = "human"


def held_mask_to_macro(held, state=None):
    """Map a held-button bitmask to the closest 12-way macro action.

    ``dash`` and ``jump`` already combine with movement in the macro set
    (``dash_left``, ``jump_right``, ...), so those chords survive exactly.
    Attack macros stop movement first (see ``agent/action.py``), so a moving
    attack keeps only its facing direction - the part the policy can act on.
    """
    if held & (1 << ATTACK):
        if held & (1 << UP):
            return MacroAction.ATTACK_UP
        if held & (1 << DOWN):
            return MacroAction.ATTACK_DOWN
        if held & (1 << LEFT):
            return MacroAction.ATTACK_LEFT
        if held & (1 << RIGHT):
            return MacroAction.ATTACK_RIGHT
        # A bare attack has no direction; aim by the knight's current facing.
        if state is not None and not state.get("knight_facing_right", True):
            return MacroAction.ATTACK_LEFT
        return MacroAction.ATTACK_RIGHT

    if held & (1 << DASH):
        if held & (1 << LEFT):
            return MacroAction.DASH_LEFT
        if held & (1 << RIGHT):
            return MacroAction.DASH_RIGHT
        if state is not None and not state.get("knight_facing_right", True):
            return MacroAction.DASH_LEFT
        return MacroAction.DASH_RIGHT

    if held & (1 << JUMP):
        if held & (1 << LEFT):
            return MacroAction.JUMP_LEFT
        if held & (1 << RIGHT):
            return MacroAction.JUMP_RIGHT
        return MacroAction.JUMP

    if held & (1 << LEFT) and held & (1 << RIGHT):
        return MacroAction.IDLE
    if held & (1 << LEFT):
        return MacroAction.MOVE_LEFT
    if held & (1 << RIGHT):
        return MacroAction.MOVE_RIGHT
    return MacroAction.IDLE


def frame_to_state_dict(frame, hit_baseline=0, damage_baseline=0):
    """Adapt a parsed :class:`Frame` to the dict ``state_to_vector`` expects.

    ``hit_count``/``damage_count`` are lifetime counters on the wire; subtract
    the episode's starting snapshot so the observation matches the per-episode
    delta the live trainer exposes.
    """
    return {
        "knight_box": frame.knight_box,
        "attacking": frame.attacking,
        "attack_box": frame.attack_box,
        "hit_count": frame.hit_count - hit_baseline,
        "damage_count": frame.damage_count - damage_baseline,
        "hit_streak": 0,
        "hazards": frame.hazards,
        "hazard_boxes": [hazard.box for hazard in frame.hazards],
        "knight_velocity": frame.knight_velocity,
        "knight_facing_right": frame.knight_facing_right,
        "boss_health": frame.boss_health,
        "boss_facing_left": frame.boss_facing_left,
        "boss_velocity": frame.boss_velocity,
        "boss_position": frame.boss_position,
        "boss_box": frame.boss_box,
        "boss_control_state": frame.boss_control_state,
        "boss_state_age": 0.0,
        "wall_distance_left": frame.wall_distance_left,
        "wall_distance_right": frame.wall_distance_right,
        "knight_health": frame.knight_health,
        "mana": frame.mana,
        "is_dashing": frame.is_dashing,
        "is_jumping": frame.is_jumping,
        "is_falling": frame.is_falling,
        "is_on_ground": frame.is_on_ground,
        "is_recoiling": frame.is_recoiling,
        "is_touching_wall_left": frame.is_touching_wall_left,
        "is_touching_wall_right": frame.is_touching_wall_right,
        "can_shade_cloak_now": frame.can_shade_cloak_now,
        "is_invuln": frame.is_invuln,
    }


def downsample_frames(records, start_game_time, step_dt=STEP_DT):
    """Pick one frame per control step, keyed to game time.

    ``records`` is a list of ``(record, Frame)`` pairs. The recorder captures
    60 FPS; the trainer acts at 10 Hz. Taking the first frame at or after each
    0.1 s grid point reproduces the live control cadence and stays correct
    through the brief game-speed dips seen around loads.
    """
    picked = []
    next_target = 0.0
    for record, state in records:
        rel = record["game_time"] - start_game_time
        if rel < 0:
            continue
        if rel >= next_target - 1e-6:
            picked.append((record, state))
            next_target += step_dt
    return picked


def stack_observations(observations, count=FRAME_STACK):
    """Replicate ``hollow_knight_env.FrameStacker`` for an offline sequence."""
    buffer = deque(maxlen=count)
    stacked = []
    for obs in observations:
        if len(buffer) < count:
            buffer.clear()
            for _ in range(count):
                buffer.append(obs.copy())
        else:
            buffer.append(obs)
        stacked.append(np.concatenate(tuple(buffer)).astype(np.float32))
    return stacked


def load_episodes(session_dir):
    """Return verified per-fight metadata, preferring the corrected file."""
    verified = session_dir / "episodes_verified.jsonl"
    fallback = session_dir / "episodes.jsonl"
    path = verified if verified.exists() else fallback
    episodes = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.endswith("\n"):
                episodes.append(json.loads(line))
    return episodes


def load_frames(session_dir):
    """Parse every raw frame into ``(record, Frame)`` pairs."""
    frames = []
    with (session_dir / "frames.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            record = json.loads(line)
            frames.append((record, _parse_frame(record["state"])))
    return frames


def build_session(session_dir):
    """Build observation/action/reward arrays for one recording session."""
    episodes = load_episodes(session_dir)
    frames = load_frames(session_dir)

    observations, actions, episode_ids, rewards, dones = [], [], [], [], []
    for episode in episodes:
        start_frame = episode["start_frame"]
        end_frame = episode["end_frame"]
        start_game_time = episode["start_game_time"]
        hit_baseline = episode.get("start_hits", 0)
        damage_baseline = episode.get("start_damage", 0)

        # Corrected boundaries, not the old per-frame episode tag.
        ep_records = [(r, s) for r, s in frames
                      if start_frame <= r["frame"] <= end_frame]
        ep_records = downsample_frames(ep_records, start_game_time)
        if len(ep_records) < 2:
            continue

        step_states = []
        step_observations = []
        step_actions = []
        for record, state in ep_records:
            state_dict = frame_to_state_dict(state, hit_baseline, damage_baseline)
            rel = record["game_time"] - start_game_time
            time_frac = min(rel / EPISODE_TIME_LIMIT, 1.0)
            obs = state_to_vector(state_dict, time_frac)
            step_states.append(state_dict)
            step_observations.append(np.asarray(obs, dtype=np.float32))
            step_actions.append(int(held_mask_to_macro(record["held"], state_dict)))

        stacked = stack_observations(step_observations)
        for index in range(len(step_states)):
            done = index == len(step_states) - 1
            reward = 0.0
            if index > 0:
                reward = calculate_reward(
                    step_states[index - 1], step_states[index], done,
                    step_dt=STEP_DT,
                )
            observations.append(stacked[index])
            actions.append(step_actions[index])
            episode_ids.append(episode["episode"])
            rewards.append(reward)
            dones.append(done)

    if not observations:
        raise ValueError(f"{session_dir}: no complete fights produced samples")
    return {
        "observations": np.stack(observations),
        "actions": np.asarray(actions, dtype=np.int64),
        "episode_ids": np.asarray(episode_ids, dtype=np.int32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=bool),
    }


def merge_sessions(sessions):
    """Concatenate several sessions, remapping episode ids to stay unique."""
    merged = {
        "observations": [], "actions": [], "episode_ids": [],
        "rewards": [], "dones": [],
    }
    episode_offset = 0
    for session in sessions:
        data = build_session(session)
        merged["observations"].append(data["observations"])
        merged["actions"].append(data["actions"])
        merged["episode_ids"].append(data["episode_ids"] + episode_offset)
        merged["rewards"].append(data["rewards"])
        merged["dones"].append(data["dones"])
        episode_offset += int(data["episode_ids"].max()) + 1
    return {
        "observations": np.concatenate(merged["observations"]),
        "actions": np.concatenate(merged["actions"]),
        "episode_ids": np.concatenate(merged["episode_ids"]),
        "rewards": np.concatenate(merged["rewards"]),
        "dones": np.concatenate(merged["dones"]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", required=True,
                        help="one or more human recording directories")
    parser.add_argument("--output", required=True,
                        help="output .npz path for train_bc.py")
    return parser.parse_args()


def main():
    args = parse_args()
    sessions = [Path(s).resolve() for s in args.sessions]
    data = merge_sessions(sessions)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
    np.savez_compressed(
        temporary,
        observations=data["observations"],
        actions=data["actions"],
        episode_ids=data["episode_ids"],
        rewards=data["rewards"],
        dones=data["dones"],
        action_names=np.asarray(MACRO_ACTION_NAMES),
        observation_size=np.int32(data["observations"].shape[1]),
        format_version=np.int32(CHECKPOINT_FORMAT),
        policy=np.str_(POLICY),
    )
    temporary.replace(output)
    print(
        f"wrote {output}: {len(data['actions'])} samples across "
        f"{int(np.unique(data['episode_ids']).size)} episodes, "
        f"observation width {data['observations'].shape[1]}",
        flush=True,
    )


if __name__ == "__main__":
    main()


