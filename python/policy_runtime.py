"""Checkpoint loading and native boss staging shared by policy players."""

import hashlib
import io
from pathlib import Path
import socket
import time

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_checkpoint(path):
    payload = Path(path).read_bytes()
    return (torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False),
            hashlib.sha256(payload).hexdigest())


def stage_fresh_fight(endpoint, boss_level):
    """Stage a verified full-health native fight, including on a fresh launch."""
    if boss_level not in (0, 1, 2):
        raise ValueError("boss level must be 0, 1, or 2")
    with socket.create_connection(endpoint, timeout=5) as sock:
        sock.settimeout(5)
        with sock.makefile("rwb", buffering=0) as stream:
            def command(value):
                stream.write((value + "\n").encode("ascii"))
                answer = stream.readline(65536).decode().strip()
                if not answer or answer.startswith("err "):
                    raise RuntimeError(f"{endpoint} {value}: {answer!r}")
                return answer

            command("RESET_FIGHT ENABLE")
            if command("RESET_FIGHT STATUS").split(";", 4)[1] == "uncaptured":
                # A new game process must enter the statue once before the Mod
                # can capture the resources needed for subsequent scene resets.
                from main_multi import prepare_fight
                prepare_fight(*endpoint)
            command(f"RESET_FIGHT LEVEL {boss_level}")
            generation = int(command("RESET_FIGHT").split()[-1])
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                status = command("RESET_FIGHT STATUS").split(";", 4)
                if int(status[2]) == generation and status[1] == "failed":
                    raise RuntimeError(f"{endpoint}: {status}")
                if int(status[2]) == generation and status[1] == "ready":
                    command("SPEED 0")
                    scalars = command("pos").split(";")[4].split(",")
                    expected_hp = 800 if boss_level == 0 else 1300
                    target = command("RESET_FIGHT TARGET").split(";")
                    if (int(scalars[3]) == expected_hp and int(scalars[7]) == 9
                            and float(scalars[9]) == 0 and len(target) == 4
                            and int(target[2]) == boss_level
                            and int(target[3]) == boss_level):
                        return
                time.sleep(0.1)
            raise RuntimeError(f"Fresh native fight reset timed out: {endpoint}")
