"""Content-addressable blob storage for /api/blobs.

Ollama exposes a blobs API where the client can check (HEAD) if a blob exists
and upload (POST) a blob addressed by its sha256 digest.

This implementation stores files under ``blobs/sha256-<hex>``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class BlobStore:
    def __init__(self, root: str | Path = "blobs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _normalize(self, digest: str) -> str:
        d = digest.strip()
        if d.startswith("sha256-"):
            return d[len("sha256-") :]
        if d.startswith("sha256:"):
            return d[len("sha256:") :]
        return d

    def _path(self, digest: str) -> Path:
        return self.root / f"sha256-{self._normalize(digest)}"

    def exists(self, digest: str) -> bool:
        return self._path(digest).is_file()

    def save(self, digest: str, content: bytes) -> Path:
        expected = self._normalize(digest)
        actual = hashlib.sha256(content).hexdigest()
        if expected != actual:
            raise ValueError(f"Digest mismatch: expected {expected}, got {actual}")
        path = self._path(digest)
        path.write_bytes(content)
        return path

    def get_path(self, digest: str) -> Path | None:
        path = self._path(digest)
        return path if path.is_file() else None
