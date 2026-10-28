"""Check real input combinations, episode boundaries and the recording protocol."""

import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest

from human_demo import EpisodeTracker, parse_demo_frame


def payload(frame, hp=1600, hero_hp=9, fsm="Idle", held=0, pressed=0, released=0):
    wire = (f"40,5,40.5,6.3;0;;3,4;0,0,1,{hp},0,0,0,{hero_hp},0,1,52,9,10,20;"
            f"0,0,0,0,0,0,1,0,0,1;{fsm};51,8,53,10")
    return f"{frame}\t{frame / 60}\t{frame / 60}\t{held}\t{pressed}\t{released}\t1\t{wire}"


class HumanDemoTests(unittest.TestCase):
    def test_chords_and_tap_edges_are_not_replaced_by_macro_labels(self):
        combo = (1 << 1) | (1 << 4) | (1 << 5)  # right + attack + jump
        record, _ = parse_demo_frame(payload(1, held=combo, pressed=1 << 4, released=1 << 6))
        self.assertEqual(record["held"], combo)
        self.assertEqual(record["pressed"], 1 << 4)
        self.assertEqual(record["released"], 1 << 6)

    def test_intro_missing_boss_and_death_are_distinguished(self):
        tracker = EpisodeTracker()
        self.assertEqual(tracker.observe(*parse_demo_frame(payload(1, fsm="Intro 4"))), (False, None))
        self.assertTrue(tracker.observe(*parse_demo_frame(payload(2)))[0])
        self.assertIsNone(tracker.observe(*parse_demo_frame(payload(3, hp=-1)))[1])
        finished = tracker.observe(*parse_demo_frame(payload(4, hp=1303, hero_hp=0)))[1]
        self.assertTrue(finished["complete"])
        self.assertFalse(finished["won"])
        self.assertEqual(finished["min_boss_hp"], 1303)
        self.assertEqual(tracker.completed, 1)

    def test_partial_episode_and_invalid_input_masks(self):
        tracker = EpisodeTracker()
        tracker.observe(*parse_demo_frame(payload(1)))
        self.assertFalse(tracker.finish_partial()["complete"])
        with self.assertRaises(ValueError):
            parse_demo_frame(payload(2, held=1 << 11))

    def test_respawn_hp_in_old_arena_is_not_a_new_fight(self):
        tracker = EpisodeTracker()
        tracker.observe(*parse_demo_frame(payload(1)))
        tracker.observe(*parse_demo_frame(payload(2, hp=16, hero_hp=0)))
        began, ended = tracker.observe(*parse_demo_frame(payload(3, hp=16, hero_hp=9)))
        self.assertFalse(began)
        self.assertIsNone(ended)
        self.assertIsNone(tracker.current)
        tracker.observe(*parse_demo_frame(payload(4, hp=-1, fsm="")))
        tracker.observe(*parse_demo_frame(payload(5, hp=1600, fsm="Intro 4")))
        self.assertTrue(tracker.observe(*parse_demo_frame(payload(6)))[0])
        self.assertEqual(tracker.current["initial_boss_hp"], 1600)


    def test_recorder_drains_batches_and_saves_a_complete_fight(self):
        errors = []
        with socket.socket() as server, tempfile.TemporaryDirectory() as temporary:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            server.settimeout(10)
            port = server.getsockname()[1]

            def serve():
                try:
                    with server.accept()[0] as peer, peer.makefile("rwb") as stream:
                        delivered = False
                        active = False
                        while True:
                            command = stream.readline().decode().strip()
                            if not command:
                                break
                            if command == "DEMO STATUS": reply = "demo_v1 idle\n"
                            elif command == "SPEED 1": reply = "ok\n"
                            elif command == "DEMO START":
                                active = True
                                reply = "ok demo_v1 started\n"
                            elif command == "DEMO STOP":
                                active = False
                                reply = "ok demo_v1 stopped\n"
                            elif command == "DEMO READ":
                                if delivered:
                                    reply = f"demo_v1 0 0 {int(active)}\n"
                                else:
                                    delivered = True
                                    reply = ("demo_v1 3 0 1\n" + payload(1, fsm="Intro 4") + "\n"
                                             + payload(2, held=50, pressed=16) + "\n"
                                             + payload(3, hp=0) + "\n")
                            else:
                                raise AssertionError(f"unexpected control command: {command}")
                            stream.write(reply.encode())
                            stream.flush()
                except Exception as exc:
                    errors.append(exc)

            worker = threading.Thread(target=serve, daemon=True)
            worker.start()
            output = Path(temporary) / "session"
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("record_human.py")),
                 "--port", str(port), "--episodes", "1", "--output", str(output)],
                capture_output=True, text=True, timeout=15,
            )
            worker.join(timeout=2)
            self.assertFalse(errors, errors)
            self.assertEqual(result.returncode, 0, result.stderr)
            meta = json.loads((output / "session.json").read_text())
            self.assertEqual((meta["status"], meta["frames"], meta["completed_episodes"]), ("complete", 3, 1))
            records = [json.loads(s) for s in (output / "frames.jsonl").read_text().splitlines()]
            self.assertEqual(records[1]["held"], 50)
            self.assertIsNone(records[0]["episode"])
            self.assertTrue(json.loads((output / "episodes.jsonl").read_text())["won"])


if __name__ == "__main__":
    unittest.main()
