"""Rebuild combat annotations from immutable raw input/state records.

The recording's raw frames and original episode file are preserved. This
also repairs a live-recording v1 boundary issue where respawn HP in the old
arena was mistaken for the beginning of a new fight.
"""

import argparse
import json
from pathlib import Path

from game_state import _parse_frame
from human_demo import EpisodeTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    tracker = EpisodeTracker()
    episodes = []
    with (args.session / "frames.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            record = json.loads(line)
            _, finished = tracker.observe(record, _parse_frame(record["state"]))
            if finished:
                episodes.append(finished)
    partial = tracker.finish_partial()
    if partial:
        episodes.append(partial)
    output = args.session / "episodes_verified.jsonl"
    temporary = output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode) + "\n")
    temporary.replace(output)
    print(json.dumps({"complete_fights": tracker.completed, "episodes": episodes,
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
