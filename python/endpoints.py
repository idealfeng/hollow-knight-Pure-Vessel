"""Endpoint parsing shared by the trainer and the connectivity probe.

A remote environment is addressed as ``HOST:PORT``. The legacy ``--ports``
flag still maps every port to 127.0.0.1, so single-machine setups behave
exactly as before.
"""

DEFAULT_HOST = "127.0.0.1"
MIN_PORT = 1024
MAX_PORT = 65535


def parse_endpoint(token):
    """Parse one ``HOST:PORT`` string into ``(host, port)``.

    Only IPv4 / hostname endpoints are supported (exactly one colon). Raises
    ``ValueError`` for malformed input.
    """
    token = str(token).strip()
    if token.count(":") != 1:
        raise ValueError(f"endpoint must be HOST:PORT, got {token!r}")
    host, port_text = token.split(":", 1)
    host = host.strip()
    if not host:
        raise ValueError(f"endpoint host is empty: {token!r}")
    try:
        port = int(port_text)
    except ValueError:
        raise ValueError(f"endpoint port must be an integer, got {token!r}")
    if not MIN_PORT <= port <= MAX_PORT:
        raise ValueError(
            f"endpoint port must be in {MIN_PORT}..{MAX_PORT}, got {token!r}")
    return host, port


def resolve_endpoints(endpoint_tokens=None, ports=None):
    """Unify ``--endpoints`` and legacy ``--ports`` into ``[(host, port), ...]``.

    ``endpoint_tokens`` (a list of ``HOST:PORT`` strings) takes precedence.
    Otherwise ``ports`` (a list of ints) is mapped to ``127.0.0.1``. Raises
    ``ValueError`` for malformed input or duplicate endpoints.
    """
    if endpoint_tokens:
        endpoints = [parse_endpoint(token) for token in endpoint_tokens]
    elif ports:
        endpoints = []
        for port in ports:
            port = int(port)
            if not MIN_PORT <= port <= MAX_PORT:
                raise ValueError(
                    f"port must be in {MIN_PORT}..{MAX_PORT}, got {port}")
            endpoints.append((DEFAULT_HOST, port))
    else:
        raise ValueError("no endpoints or ports provided")

    seen = set()
    for endpoint in endpoints:
        if endpoint in seen:
            raise ValueError(f"duplicate endpoint: {endpoint[0]}:{endpoint[1]}")
        seen.add(endpoint)
    return endpoints
