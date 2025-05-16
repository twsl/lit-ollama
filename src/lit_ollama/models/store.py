from datetime import datetime
import hashlib
from pathlib import Path
import shutil
from typing import cast

from litgpt.utils import auto_download_checkpoint

from lit_ollama.api.schema.base import ModelDetails, TagModel


class ModelStore:
    def __init__(self) -> None:
        self.models: dict[str, TagModel] = {}

        self.load_models()

    def get_file_hash(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_folder_hash(self, folder_path: Path) -> str:
        sha256 = hashlib.sha256()
        for file_path in sorted(folder_path.rglob("*")):
            if file_path.is_file():
                sha256.update(file_path.name.encode())  # Include file name in hash
                sha256.update(self.get_file_hash(file_path).encode())
        return sha256.hexdigest()

    def load_models(self):
        for path in Path("checkpoints").rglob("**/*/*"):
            if path.is_dir():
                stats = path.stat()
                modification_time = datetime.fromtimestamp(stats.st_mtime)
                self.models[path.name] = TagModel(
                    name=path.name,
                    modified_at=modification_time,
                    size=stats.st_size,
                    digest="",
                    details=ModelDetails(
                        family=path.parent.name,
                        format="",
                        families=[],
                        parameter_size="",
                        quantization_level="",
                    ),
                )

    def get_model(self, model_name: str) -> dict:
        model = self.models.get(model_name)
        return cast(TagModel, model).encode() if model else {}

    def copy_model(self, model_name: str, new_model_name: str) -> None:
        self.copy_folder(Path("checkpoints") / model_name, Path("checkpoints") / new_model_name)
        model = self.models.get(model_name)
        if model:
            self.models[new_model_name] = model

    def delete_model(self, model_name: str) -> None:
        if model_name in self.models:
            del self.models[model_name]

    def copy_folder(self, src: Path, dst: Path) -> None:
        if not src.is_dir():
            raise ValueError(f"Source {src} is not a directory")
        if dst.exists():
            raise ValueError(f"Destination {dst} already exists")
        shutil.copytree(src, dst)

    def pull_model(self, model_name: str) -> None:
        auto_download_checkpoint(model_name=model_name)
