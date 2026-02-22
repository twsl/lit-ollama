from __future__ import annotations

from pathlib import Path

from lit_ollama.server.schema.base import TagModel
from lit_ollama.store.blobs import BlobStore
from lit_ollama.store.models import ModelStore


class LocalStore:
    """Local filesystem-backed storage used by the API.

    This is intentionally a thin façade that composes the independent blob/model
    stores so the API only has to wire up a single dependency.
    """

    def __init__(self, *, blobs_root: str | Path = "blobs") -> None:
        self._models = ModelStore()
        self._blobs = BlobStore(root=blobs_root)

    @property
    def models(self) -> dict[str, TagModel]:
        return self._models.models

    @property
    def blobs(self) -> BlobStore:
        return self._blobs

    # ModelStore passthroughs
    def load_models(self) -> None:
        self._models.load_models()

    def get_file_hash(self, file_path: Path) -> str:
        return self._models.get_file_hash(file_path)

    def get_folder_hash(self, folder_path: Path) -> str:
        return self._models.get_folder_hash(folder_path)

    def get_model(self, model_name: str) -> dict:
        return self._models.get_model(model_name)

    def copy_model(self, model_name: str, new_model_name: str) -> None:
        self._models.copy_model(model_name, new_model_name)

    def delete_model(self, model_name: str) -> None:
        self._models.delete_model(model_name)

    def pull_model(self, model_name: str) -> None:
        self._models.pull_model(model_name)
