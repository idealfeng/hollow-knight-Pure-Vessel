"""Optional scene reset protocol; old mods keep using statue navigation."""

import os
import time

import controls


_OWNER_MISSING = "err fast reset is not enabled"
_OWNER_BUSY = "err another trainer owns fast reset"


def enabled_by_config():
    return os.environ.get("HOLLOWRL_FAST_RESET", "auto").lower() not in ("0", "off", "false")


def enable():
    if not enabled_by_config():
        return False
    reply = controls.request("RESET_FIGHT ENABLE")
    return bool(reply and reply.startswith("ok reset_v1"))


def disable():
    return controls.request("RESET_FIGHT DISABLE")


def _request_owned(command, *, retry=True):
    """Recover socket-bound reset ownership after an automatic reconnect.

    Only an explicit rejection is safe to replay. In particular a missing
    START reply may mean that Unity already started loading the fight, so it
    must remain a failed request rather than risk resetting it twice.
    """
    reply = controls.request(command) if retry else controls.request(command, retry=False)
    if reply != _OWNER_MISSING or not enabled_by_config():
        return reply

    # Closing the old socket and accepting the replacement run on separate
    # Unity threads. Give the old owner a short chance to release its lease;
    # ENABLE never displaces an existing owner, even if it is another trainer.
    for attempt in range(3):
        enabled = controls.request("RESET_FIGHT ENABLE")
        if enabled == "ok reset_v1 enabled":
            return (controls.request(command) if retry
                    else controls.request(command, retry=False))
        if enabled != _OWNER_BUSY or attempt == 2:
            return enabled
        time.sleep(0.1)
    return reply


def set_level(level):
    if level not in (0, 1, 2):
        raise ValueError("boss level must be 0, 1, or 2")
    reply = _request_owned(f"RESET_FIGHT LEVEL {level}")
    if reply != f"ok reset_v1 level {level}":
        raise RuntimeError(f"Cannot set boss difficulty (tier-switch mod required): {reply!r}")


def target():
    reply = controls.request("RESET_FIGHT TARGET")
    if not reply or not reply.startswith("target_v1;"):
        return None
    parts = reply.split(";")
    if len(parts) != 4:
        raise ValueError(f"invalid target reply: {reply!r}")
    return {"scene": parts[1], "target_level": int(parts[2]), "observed_level": int(parts[3])}


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
    reply = _request_owned("RESET_FIGHT", retry=False)
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
