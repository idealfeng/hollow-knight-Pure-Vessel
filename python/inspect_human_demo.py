"""Audit a human recording, including an ongoing session, without touching the game."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from game_state import _parse_frame
from human_demo import BUTTON_NAMES, BUTTON_MASK


def inspect_session(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    if (metadata.get("format") != "hollowrl_human_frames_v1"
            or metadata.get("button_names") != list(BUTTON_NAMES)):
        raise ValueError("unsupported recording schema")
    pressed, held, released, combinations = Counter(), Counter(), Counter(), Counter()
    combat_pressed = Counter()
    frames = combat_frames = zero_hero_frames = 0
    frame_gaps = 0
    last_frame = None
    last_wall = None
    intervals = []
    speeds = Counter()
    issues = []
    with (directory / "frames.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                issues.append("last frame is still being written or was interrupted")
                break
            record = json.loads(line)
            state = _parse_frame(record["state"])
            frame_id = record["frame"]
            if last_frame is not None:
                if frame_id <= last_frame:
                    raise ValueError("non-increasing frame IDs")
                frame_gaps += max(0, frame_id - last_frame - 1)
                dt = record["wall_time"] - last_wall
                if dt <= 0:
                    raise ValueError("non-increasing timestamps")
                if frame_id == last_frame + 1:
                    intervals.append(dt)
            last_frame, last_wall = frame_id, record["wall_time"]
            frames += 1
            combat = record["episode"] is not None
            combat_frames += int(combat)
            zero_hero_frames += int(state.knight_box is None)
            speeds[round(state.game_speed, 2)] += 1
            for key, target in (("held", held), ("pressed", pressed), ("released", released)):
                bits = record[key]
                if bits < 0 or bits & ~BUTTON_MASK:
                    raise ValueError("invalid button mask")
                for i, name in enumerate(BUTTON_NAMES):
                    if bits & (1 << i):
                        target[name] += 1
                        if key == "pressed" and combat:
                            combat_pressed[name] += 1
            chord = [name for i, name in enumerate(BUTTON_NAMES)
                     if record["held"] & (1 << i)]
            if combat and len(chord) > 1:
                combinations["+".join(chord)] += 1
    episodes = []
    with (directory / "episodes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.endswith("\n"):
                episodes.append(json.loads(line))
    if metadata.get("dropped_frames", 0):
        issues.append(f"Mod queue dropped {metadata['dropped_frames']} frames")
    if not frames:
        issues.append("no frames yet; load the training save")
    elif not pressed:
        issues.append("no button presses observed yet; input capture needs a human check")
    if not combat_frames:
        issues.append("no combat frames yet")
    if metadata.get("status") == "failed" or metadata.get("shutdown_error"):
        issues.append("recorder reported an error; inspect session.json")
    return {
        "session": str(directory.resolve()), "status": metadata["status"],
        "frames": frames, "combat_frames": combat_frames,
        "frames_without_hero": zero_hero_frames, "frame_id_gaps": frame_gaps,
        "dropped_frames": metadata.get("dropped_frames", 0),
        "median_fps": round(1 / statistics.median(intervals), 1) if intervals else None,
        "game_speed_frame_counts": dict(speeds),
        "button_presses": dict(pressed), "button_releases": dict(released),
        "button_held_frames": dict(held), "combat_button_presses": dict(combat_pressed),
        "combat_combinations": dict(combinations.most_common(20)),
        "episodes": episodes, "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_session(args.session)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
