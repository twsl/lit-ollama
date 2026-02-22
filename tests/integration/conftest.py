"""Integration test fixtures.

Starts a lit-ollama server in-process with the mock model on a free port
and provides an ``ollama.Client`` pointing at it.
"""

from __future__ import annotations

from collections.abc import Generator
import socket
import threading
import time

import httpx
import litserve as ls
from ollama import Client
import pytest

from lit_ollama.server.api import LitOllamaAPI


def _free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 30.0, interval: float = 0.5) -> None:
    """Block until *url* returns a 200 response or *timeout* is reached."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=3)
            if r.status_code == 200:
                return
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
            pass
        time.sleep(interval)
    raise TimeoutError(f"Server at {url} did not become ready within {timeout}s")


@pytest.fixture(scope="session")
def server_url() -> Generator[str, None, None]:
    """Start the mock server in-process and yield its base URL."""
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    api = LitOllamaAPI("mock")
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices="auto",
        callbacks=None,
        middlewares=None,
    )

    t = threading.Thread(
        target=server.run,
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "log_level": "warning",
            "generate_client_file": False,
        },
        daemon=True,
    )
    t.start()

    try:
        _wait_for_server(f"{url}/health")
        yield url
    finally:
        server._shutdown_event.set()
        t.join(timeout=10)


@pytest.fixture(scope="session")
def client(server_url: str) -> Client:
    """Return an ollama ``Client`` connected to the running test server."""
    return Client(host=server_url)


@pytest.fixture(scope="session")
def base_url(server_url: str) -> str:
    """Raw base URL for endpoints not covered by the ollama client."""
    return server_url
