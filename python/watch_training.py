"""Read-only live PPO dashboard; never connects to games or starts training."""
import argparse
import json
from pathlib import Path

from monitor import TrainingMonitor

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "python/runs/traitor_4env_iframes_fixed/active_experiment.json"


class JsonlTail:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.pending = b""

    def read(self):
        if not self.path.exists():
            return []
        with self.path.open("rb") as stream:
            if self.path.stat().st_size < self.offset:
                self.offset, self.pending = 0, b""
            stream.seek(self.offset)
            chunk = stream.read()
            self.offset += len(chunk)
        lines = (self.pending + chunk).split(b"\n")
        self.pending = lines.pop()
        records = []
        for line in lines:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except (ValueError, UnicodeDecodeError):
                    continue
        return records


def active_run():
    try:
        directory = Path(json.loads(ACTIVE.read_text(encoding="utf-8-sig"))["directory"])
        progress_path = directory / "progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8-sig")) if progress_path.exists() else {}
        metrics = list(directory.glob("*/training_metrics.jsonl"))
        run_dir = max(metrics, key=lambda item: item.stat().st_mtime).parent if metrics else directory
        return run_dir, f"{progress.get('phase', 'waiting')} {progress.get('active') or ''}".strip()
    except (OSError, ValueError, KeyError):
        return ROOT / "python/runs/traitor_4env_iframes_fixed", "waiting for active task"


class ReadOnlyMonitor(TrainingMonitor):
    def _load_history(self):
        pass

    def _save_history(self):
        pass

    def _save_history_throttled(self):
        pass

    def __init__(self, run_dir=None):
        super().__init__(action_labels=["left", "right", "up", "down", "attack", "jump", "dash"])
        self.fixed_run = Path(run_dir).resolve() if run_dir else None
        self.tail = None
        self.root.title("PPO Training Monitor — read only")
        self.poll()

    def poll(self):
        run_dir, phase = (self.fixed_run, "live") if self.fixed_run else active_run()
        path = run_dir / "training_metrics.jsonl"
        if self.tail is None or self.tail.path != path:
            self.tail = JsonlTail(path)
            # Each profile/branch has its own curves; reward scales may differ.
            for name, value in list(vars(self).items()):
                if isinstance(value, list):
                    if name not in ("action_sizes", "action_labels"):
                        value.clear()
        for record in self.tail.read():
            if record.get("event") == "episode":
                self.add_reward(record["reward"], steps=record.get("steps"),
                                hits=record.get("hits"), damage=record.get("damage"),
                                misses=record.get("misses"), min_boss_hp=record.get("min_boss_health"),
                                won=record.get("won"))
            elif record.get("event") == "update":
                self.add_update(record)
        self.run_info_label.config(text=f"READ ONLY | {run_dir.name} | {phase}")
        self.root.after(1000, self.poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="optional fixed run directory; default follows the active experiment")
    args = parser.parse_args()
    ReadOnlyMonitor(args.run_dir).start()


if __name__ == "__main__":
    main()
