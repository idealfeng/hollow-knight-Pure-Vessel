"""Unit tests for endpoint parsing and the read-only connectivity probe."""

import socket
import threading
import unittest

from endpoints import parse_endpoint, resolve_endpoints
from check_endpoints import check_endpoint


class ParseEndpointTests(unittest.TestCase):
    def test_local_and_remote_endpoints(self):
        self.assertEqual(parse_endpoint("127.0.0.1:9001"), ("127.0.0.1", 9001))
        self.assertEqual(parse_endpoint("192.168.1.50:9002"), ("192.168.1.50", 9002))

    def test_hostname_endpoint(self):
        self.assertEqual(parse_endpoint("game-host:9003"), ("game-host", 9003))

    def test_rejects_missing_colon(self):
        with self.assertRaises(ValueError):
            parse_endpoint("9001")

    def test_rejects_empty_host(self):
        with self.assertRaises(ValueError):
            parse_endpoint(":9001")

    def test_rejects_non_integer_port(self):
        with self.assertRaises(ValueError):
            parse_endpoint("127.0.0.1:notaport")

    def test_rejects_port_out_of_range(self):
        with self.assertRaises(ValueError):
            parse_endpoint("127.0.0.1:80")


class ResolveEndpointsTests(unittest.TestCase):
    def test_legacy_ports_map_to_loopback(self):
        self.assertEqual(
            resolve_endpoints(ports=[9001, 9002]),
            [("127.0.0.1", 9001), ("127.0.0.1", 9002)],
        )

    def test_local_endpoint(self):
        self.assertEqual(
            resolve_endpoints(endpoint_tokens=["127.0.0.1:9001"]),
            [("127.0.0.1", 9001)],
        )

    def test_remote_endpoint(self):
        self.assertEqual(
            resolve_endpoints(endpoint_tokens=["192.168.1.50:9002"]),
            [("192.168.1.50", 9002)],
        )

    def test_mixed_endpoints(self):
        self.assertEqual(
            resolve_endpoints(endpoint_tokens=["127.0.0.1:9001", "192.168.1.50:9002"]),
            [("127.0.0.1", 9001), ("192.168.1.50", 9002)],
        )

    def test_endpoints_take_precedence_over_ports(self):
        self.assertEqual(
            resolve_endpoints(endpoint_tokens=["192.168.1.50:9002"], ports=[9001, 9002]),
            [("192.168.1.50", 9002)],
        )

    def test_duplicate_endpoint_raises(self):
        with self.assertRaises(ValueError):
            resolve_endpoints(endpoint_tokens=["127.0.0.1:9001", "127.0.0.1:9001"])

    def test_same_port_different_host_allowed(self):
        self.assertEqual(
            resolve_endpoints(endpoint_tokens=["127.0.0.1:9001", "192.168.1.50:9001"]),
            [("127.0.0.1", 9001), ("192.168.1.50", 9001)],
        )

    def test_no_input_raises(self):
        with self.assertRaises(ValueError):
            resolve_endpoints()


class CheckEndpointTests(unittest.TestCase):
    def test_connection_refused_raises(self):
        with self.assertRaises(OSError):
            check_endpoint("127.0.0.1", 1, timeout=0.5)

    def test_server_closing_connection_raises(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def accept_then_close():
            conn, _ = server.accept()
            conn.close()

        thread = threading.Thread(target=accept_then_close, daemon=True)
        thread.start()
        try:
            with self.assertRaises((ConnectionError, OSError)):
                check_endpoint("127.0.0.1", port, timeout=2.0)
        finally:
            thread.join(timeout=2.0)
            server.close()


if __name__ == "__main__":
    unittest.main()
