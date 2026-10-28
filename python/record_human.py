"""Record human play from the mod's per-frame input/state queue.

Raw button combinations are preserved, not converted to the 12 PPO macros.
The player controls entry, retries and combat. No policy controls the game.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import signal
import socket
import time

from human_demo import BUTTON_NAMES, EpisodeTracker, parse_demo_frame


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--episodes", type=int, default=5,
                        help="complete fights to record; wins and losses both count")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="optional wall-clock limit; 0 waits for the requested fights")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or args.episodes < 1 or args.max_seconds < 0:
        parser.error("invalid port, episodes or max-seconds")
    return args


def main():
    args = parse_args()
    tracker = EpisodeTracker()
    metadata = {
        "format": "hollowrl_human_frames_v1", "button_names": list(BUTTON_NAMES),
        "port": args.port, "requested_episodes": args.episodes,
        "created_at": datetime.now().astimezone().isoformat(),
        "timing": "state from Unity Update, hardware input from LateUpdate of the same frame",
        "labels": "raw held/pressed/released bitmasks; seven combat buttons used for BC",
        "status": "starting", "frames": 0, "completed_episodes": 0, "dropped_frames": 0,
    }
    with socket.create_connection(("127.0.0.1", args.port), timeout=5.0) as sock:
        sock.settimeout(5.0)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with sock.makefile("rwb") as stream:
            def command(value):
                stream.write(value.encode("ascii") + b"\n")
                stream.flush()
                line = stream.readline().decode("utf-8-sig").strip()
                if not line or line.startswith("err "):
                    raise RuntimeError(f"{value}: {line or 'game disconnected'}")
                return line

            status = command("DEMO STATUS")
            if not status.startswith("demo_v1 "):
                raise RuntimeError("install the human-demonstration Mod before recording")
            print(f"port {args.port}: {status}", flush=True)
            if args.probe:
                return
            if status.endswith("recording"):
                raise RuntimeError("another recording is active; leave that session running")

            output = args.output or (Path(__file__).resolve().parent / "runs" / "human_demos"
                                     / f"human_{args.port}_{datetime.now():%Y%m%d_%H%M%S}")
            output.mkdir(parents=True, exist_ok=False)
            metadata_path = output / "session.json"

            def save_metadata():
                temporary = metadata_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                temporary.replace(metadata_path)

            command("SPEED 1")
            time.sleep(0.25)
            command("DEMO START")
            metadata["status"] = "recording"
            save_metadata()
            print(f"Recording to: {output}\n"
                  f"Play normally in the {args.port} window (keyboard or controller).\n"
                  "Load the training save and enter Traitor Lord manually.\n"
                  f"Menus/intro are excluded from combat; stops after {args.episodes} fights.\n"
                  "Ctrl+C, or creating a STOP file in the output folder, saves and stops.", flush=True)
            started = time.monotonic()

            with (output / "frames.jsonl").open("w", encoding="utf-8") as frames_file, \
                 (output / "episodes.jsonl").open("w", encoding="utf-8") as episodes_file:
                stop_requested = False

                def request_stop(signum, frame):
                    nonlocal stop_requested
                    stop_requested = True

                previous_sigint = signal.signal(signal.SIGINT, request_stop)

                def drain():
                    header = command("DEMO READ").split()
                    if len(header) != 4 or header[0] != "demo_v1":
                        raise RuntimeError(f"invalid demonstration batch header: {header}")
                    count, dropped, active = map(int, header[1:])
                    if not 0 <= count <= 3600:
                        raise RuntimeError("invalid demonstration batch size")
                    for _ in range(count):
                        payload = stream.readline()
                        if not payload:
                            raise RuntimeError("game disconnected in a demonstration batch")
                        record, state = parse_demo_frame(payload.decode("utf-8"))
                        began, finished = tracker.observe(record, state)
                        frames_file.write(json.dumps(record, separators=(",", ":")) + "\n")
                        metadata["frames"] += 1
                        if began:
                            print(f"Fight {record['episode']} started: boss HP={state.boss_health}, "
                                  f"knight HP={state.knight_health}", flush=True)
                        if finished:
                            episodes_file.write(json.dumps(finished) + "\n")
                            episodes_file.flush()
                            metadata["completed_episodes"] = tracker.completed
                            print(f"Fight {finished['episode']} saved: hits={finished['hits']} "
                                  f"boss HP={finished['min_boss_hp']} won={finished['won']} "
                                  f"duration={finished['game_seconds']:.1f}s", flush=True)
                    frames_file.flush()
                    metadata["dropped_frames"] = dropped
                    save_metadata()
                    if dropped:
                        raise RuntimeError(f"Mod queue dropped {dropped} frames; recording retained for inspection")
                    return bool(active)

                try:
                    while tracker.completed < args.episodes:
                        if (output / "STOP").exists():
                            stop_requested = True
                        if stop_requested:
                            break
                        if args.max_seconds and time.monotonic() - started >= args.max_seconds:
                            break
                        if not drain():
                            raise RuntimeError("the Mod stopped recording unexpectedly")
                        time.sleep(0.1)
                    metadata["status"] = ("complete" if tracker.completed >= args.episodes
                                          else "stopped_by_user" if stop_requested else "time_limit")
                except KeyboardInterrupt:
                    metadata["status"] = "stopped_by_user"
                except Exception as exc:
                    metadata["status"] = "failed"
                    metadata["error"] = str(exc)
                    raise
                finally:
                    try:
                        command("DEMO STOP")
                        drain()
                    except (OSError, ValueError, RuntimeError) as exc:
                        metadata["shutdown_error"] = str(exc)
                    partial = tracker.finish_partial()
                    if partial:
                        episodes_file.write(json.dumps(partial) + "\n")
                    metadata["completed_episodes"] = tracker.completed
                    save_metadata()
                    signal.signal(signal.SIGINT, previous_sigint)
            print(f"Saved {metadata['frames']} frames, {tracker.completed} complete fights: {output}", flush=True)


if __name__ == "__main__":
    main()
