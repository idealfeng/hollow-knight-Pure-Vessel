"""Convert recorded human demonstrations into the NPZ format train_bc.py reads.

A human session records 60 FPS raw button bitmasks plus the same state protocol
the trainer reads (see HUMAN_DEMONSTRATIONS.md). ``train_bc.py`` needs 10 Hz
stacked observations paired with seven button labels. This script
bridges the two:

  * it uses the *corrected* episode boundaries from ``episodes_verified.jsonl``
    (the raw ``frames.jsonl`` ``episode`` tag came from the old, buggy recorder);
  * it downsamples each fight from 60 FPS to the 10 Hz control rate;
  * it preserves the seven combat buttons from the held-button bitmask;
  * it stacks four consecutive frames and aligns rewards to the action that
    precedes each state transition;
  * it writes an NPZ that ``train_bc.py`` can consume.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from agent.action import BUTTON_ORDER
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


def held_to_buttons(held):
    """Extract the multi-binary button vector from a recorded held-bitmask.

    The recorder stores the full 11-button mask; the policy only needs the
    seven combat buttons in ``BUTTON_ORDER``. Compatible chords remain intact;
    opposing directions are mapped to neutral because the policy excludes them.
    """
    buttons = [int(bool(held & (1 << BUTTON_NAMES.index(name))))
               for name in BUTTON_ORDER]
    # The policy represents each direction pair as a three-way choice. A
    # human can briefly hold both keys during a turn; that input has no clear
    # direction and is represented as neutral in the training data.
    for first, second in ((0, 1), (2, 3)):
        if buttons[first] and buttons[second]:
            buttons[first] = buttons[second] = 0
    return buttons


def frame_to_state_dict(frame, hit_baseline=0, damage_baseline=0,
                        boss_state_age=0.0):
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
        "boss_state_age": boss_state_age,
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


def build_session(session_dir, frame_stack=FRAME_STACK):
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
        state_ages = {}
        previous_fsm = None
        state_started_at = start_game_time
        for record, state in ep_records:
            if state.boss_control_state != previous_fsm:
                state_started_at = record["game_time"]
                previous_fsm = state.boss_control_state
            state_ages[record["frame"]] = max(record["game_time"] - state_started_at, 0.0)
        final_record = ep_records[-1] if ep_records else None
        ep_records = downsample_frames(ep_records, start_game_time)
        # A death/win can fall between two 10 Hz grid points. Keep its final
        # state for the preceding action's reward, but never train an action
        # from that terminal frame.
        if ep_records and final_record[0]["frame"] != ep_records[-1][0]["frame"]:
            ep_records.append(final_record)
        if len(ep_records) < 2:
            continue

        step_states = []
        step_observations = []
        step_actions = []
        for record, state in ep_records:
            state_dict = frame_to_state_dict(
                state, hit_baseline, damage_baseline,
                boss_state_age=state_ages[record["frame"]],
            )
            rel = record["game_time"] - start_game_time
            time_frac = min(rel / EPISODE_TIME_LIMIT, 1.0)
            obs = state_to_vector(state_dict, time_frac)
            step_states.append(state_dict)
            step_observations.append(None if obs is None else np.asarray(obs, dtype=np.float32))
            step_actions.append(held_to_buttons(record["held"]))

        if any(obs is None for obs in step_observations[:-1]):
            raise ValueError(f"{session_dir}: missing knight geometry during combat")
        stacked = stack_observations(step_observations[:-1], count=frame_stack)
        for index in range(len(step_states) - 1):
            done = index == len(step_states) - 2
            reward = calculate_reward(
                step_states[index], step_states[index + 1], done,
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


def merge_sessions(sessions, frame_stack=FRAME_STACK):
    """Concatenate several sessions, remapping episode ids to stay unique."""
    merged = {
        "observations": [], "actions": [], "episode_ids": [],
        "rewards": [], "dones": [],
    }
    episode_offset = 0
    for session in sessions:
        data = build_session(session, frame_stack=frame_stack)
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
    parser.add_argument("--frame-stack", type=int, default=FRAME_STACK,
                        help="number of 10 Hz observations to stack")
    args = parser.parse_args()
    if args.frame_stack < 1:
        parser.error("--frame-stack must be positive")
    return args


def main():
    args = parse_args()
    sessions = [Path(s).resolve() for s in args.sessions]
    data = merge_sessions(sessions, frame_stack=args.frame_stack)

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
        action_names=np.asarray(BUTTON_ORDER),
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


