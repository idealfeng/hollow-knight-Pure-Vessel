"""Connectivity probe for idle local and remote Hollow Knight mods.

Connects to each HOST:PORT, sends a single ``pos`` request and validates that
the reply looks like the mod's state frame. Run this only when no trainer or
policy player controls the endpoint: disconnecting a probe can clear inputs.

Usage:
    python check_endpoints.py 127.0.0.1:9001 192.168.1.50:9002
"""

import argparse
import socket
import sys

from endpoints import parse_endpoint


def check_endpoint(host, port, timeout=5.0):
    """Return the raw state-frame text, or raise on any failure.

    Only ever issues one ``pos`` command - read-only, no state change.
    """
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(b"pos\n")
        with sock.makefile("rb") as reader:
            line = reader.readline(65536)
    if not line:
        raise ConnectionError("mod closed the connection without a reply")
    text = line.decode("utf-8").rstrip("\r\n")
    if not text:
        raise ConnectionError("empty reply from mod")
    sections = text.split(";")
    if len(sections) < 8:
        raise ValueError(f"unexpected frame: {len(sections)} sections (expected >= 8)")
    scalars = sections[4].split(",")
    if len(scalars) < 14:
        raise ValueError(f"unexpected scalar section: {len(scalars)} fields (expected >= 14)")
    try:
        int(scalars[3])  # boss_health
        int(scalars[7])  # knight_health
    except ValueError as exc:
        raise ValueError(f"unparseable health fields: {exc}") from exc
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoints", nargs="+", help="HOST:PORT ...")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)

    failed = False
    for token in args.endpoints:
        try:
            host, port = parse_endpoint(token)
            text = check_endpoint(host, port, args.timeout)
            print(f"OK   {host}:{port}  ({len(text.split(';'))} sections)", flush=True)
        except Exception as exc:
            failed = True
            print(f"FAIL {token}  ({exc})", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
