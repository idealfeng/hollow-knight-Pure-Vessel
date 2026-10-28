"""Summarize live PPO progress from a run's training_metrics.jsonl file."""

import argparse
import json
from pathlib import Path
from statistics import mean, median


def load_metrics(path):
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            # The trainer may still be writing the final line.
    return rows


def describe(label, episodes):
    if not episodes:
        return
    print(
        f"{label}: n={len(episodes)}, wins={sum(bool(e['won']) for e in episodes)}, "
        f"hits={mean(e['hits'] for e in episodes):.2f}, "
        f"boss_min_hp={mean(e['min_boss_health'] for e in episodes):.1f}, "
        f"reward={mean(e['reward'] for e in episodes):+.1f}, "
        f"steps={mean(e['steps'] for e in episodes):.0f}, "
        f"reset_median={median(e['reset_seconds'] for e in episodes):.1f}s"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")

    rows = load_metrics(args.run_dir / "training_metrics.jsonl")
    episodes = [row for row in rows if row.get("event") == "episode"]
    updates = [row for row in rows if row.get("event") == "update"]
    print(f"{len(episodes)} complete fights, {len(updates)} PPO updates")
    describe("first", episodes[:args.window])
    describe("latest", episodes[-args.window:])
    if updates:
        latest = updates[-1]
        print(
            f"latest update {latest['update']}: "
            f"explained_variance={latest['explained_variance']:+.2f}, "
            f"approx_kl={latest['approx_kl']:.4f}, "
            f"step/s={latest['n_samples'] / max(latest['elapsed_seconds'], 1e-6):.1f}"
        )


if __name__ == "__main__":
    main()
