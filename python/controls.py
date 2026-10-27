import os
import socket
import threading

HOST = os.environ.get("HOLLOWRL_HOST", "127.0.0.1")
PORT = int(os.environ.get("HOLLOWRL_PORT", "9000"))
SOCKET_TIMEOUT = 2.0

# How long a plain press holds the action down. pyautogui.press() was too fast
# for the game's per-frame input polling to see reliably, which is why
# reset.py grew its own tap() with an explicit hold - 0.06 s spans ~4 frames
# at 60 FPS. Same reasoning, now the default everywhere.
DEFAULT_TAP_MS = 60

# Every action the mod understands. Kept here so a typo fails in Python with a
# useful message instead of being silently ignored inside the game.
ACTIONS = frozenset({
    "left", "right", "up", "down",
    "jump", "attack", "dash", "cast", "quick_cast", "super_dash", "dream_nail",
    "submit", "cancel", "pause",
})

_sock = None
_lock = threading.Lock()   # the control loop and the monitor thread both send
_warned = False


def _connect():
    """(Re)establish the command socket. Caller must hold _lock."""
    global _sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(SOCKET_TIMEOUT)
    sock.connect((HOST, PORT))
    # Input is latency-critical and every command is tiny - Nagle would sit on
    # a 10-byte "KD left" waiting for company while the knight keeps running.
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    _sock = sock
    return sock


def _send(command):
    """Send one command and wait for the mod's acknowledgement.

    Synchronous on purpose: the reply is one round trip on loopback (well
    under a millisecond) and it keeps the socket in lockstep, so a dropped
    command shows up here instead of as an inexplicable policy regression.
    Reconnects once on failure - the game restarting mid-session should not
    take the trainer down with it.
    """
    global _sock, _warned

    with _lock:
        for attempt in (1, 2):
            try:
                sock = _sock or _connect()
                sock.sendall(command.encode() + b"\n")
                if not sock.recv(64):
                    raise ConnectionError("mod closed the connection")
                _warned = False
                return True
            except (ConnectionError, OSError):
                if _sock is not None:
                    _sock.close()
                    _sock = None
                if attempt == 2 and not _warned:
                    # Once per outage, not once per dropped keystroke at 15 Hz.
                    print("controls: no input channel to the game "
                          "(is it running with the HollowRL mod?)")
                    _warned = True
        return False


def _check(action):
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}; expected one of "
                         f"{', '.join(sorted(ACTIONS))}")
    return action


def key_down(action):
    """Hold an action until key_up/release_all. Idempotent."""
    _send(f"KD {_check(action)}")


def key_up(action):
    """Release a held action. Safe to call on one that is not held."""
    _send(f"KU {_check(action)}")


def tap(action, hold_ms=DEFAULT_TAP_MS):
    _send(f"TAP {_check(action)} {int(hold_ms)}")


def release_all():
    _send("CLEAR")


def kill():
    _send("KILL")


def set_speed(factor):
    factor = float(factor)
    if not 0.0 <= factor <= 4.0:
        raise ValueError("game speed must be in 0..4.0")
    return _send(f"SPEED {factor:g}")


def close():
    """Drop the command socket, releasing any held action."""
    global _sock
    with _lock:
        if _sock is not None:
            _sock.close()
            _sock = None
