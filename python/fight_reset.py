"""Optional scene reset protocol; old mods keep using statue navigation."""

import os
import time

import controls


def enabled_by_config():
    return os.environ.get("HOLLOWRL_FAST_RESET", "auto").lower() not in ("0", "off", "false")


def enable():
    if not enabled_by_config():
        return False
    reply = controls.request("RESET_FIGHT ENABLE")
    return bool(reply and reply.startswith("ok reset_v1"))


def disable():
    return controls.request("RESET_FIGHT DISABLE")


def status():
    reply = controls.request("RESET_FIGHT STATUS")
    if not reply or not reply.startswith("reset_v1;"):
        return None
    parts = reply.split(";", 4)
    if len(parts) != 5:
        raise ValueError(f"invalid reset status: {reply!r}")
    return {"state": parts[1], "generation": int(parts[2]),
            "scene": parts[3], "message": parts[4]}


def reset(combat_ready, timeout=35.0):
    """Return reset details only after both the protocol and state feed agree.

    None means no captured target, allowing the menu path on first launch.
    A failed supported request is returned explicitly so the caller can turn
    off the death guard before falling back to normal game behavior.
    """
    initial = status()
    if initial is None or initial["state"] == "uncaptured":
        return None
    started = time.perf_counter()
    reply = controls.request("RESET_FIGHT", retry=False)
    if not reply or not reply.startswith("ok reset_v1 "):
        return {"ok": False, "mode": "scene", "attempts": 1,
                "seconds": time.perf_counter() - started, "error": reply}
    generation = int(reply.split()[-1])
    last_status = initial
    while time.perf_counter() - started < timeout:
        last_status = status()
        if last_status and last_status["generation"] == generation:
            if last_status["state"] == "failed":
                break
            if last_status["state"] == "ready" and combat_ready():
                return {"ok": True, "mode": "scene", "attempts": 1,
                        "seconds": time.perf_counter() - started,
                        "generation": generation}
        time.sleep(0.05)
    return {"ok": False, "mode": "scene", "attempts": 1,
            "seconds": time.perf_counter() - started, "error": last_status}
