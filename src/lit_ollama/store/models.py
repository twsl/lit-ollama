from datetime import datetime
import hashlib
from pathlib import Path
import shutil
from typing import Any, cast

from litgpt.config import Config
from litgpt.utils import auto_download_checkpoint

from lit_ollama.server.schema.base import ModelDetails, TagModel
from lit_ollama.utils.metadata import build_model_details, estimate_parameter_count


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

    def load_models(self) -> None:
        checkpoints = Path("checkpoints")
        if not checkpoints.is_dir():
            return

        def folder_size(folder: Path) -> int:
            total = 0
            for fp in folder.rglob("*"):
                if fp.is_file():
                    total += fp.stat().st_size
            return total

        # Discover model directories. In practice checkpoints may be nested as <org>/<name>/...
        for path in checkpoints.rglob("*"):
            if not path.is_dir():
                continue
            # Heuristic: treat any directory containing files as a model directory.
            if not any(p.is_file() for p in path.iterdir()):
                continue

            stats = path.stat()
            modification_time = datetime.fromtimestamp(stats.st_mtime)
            model_name = path.name

            digest = ""
            try:
                digest = f"sha256:{self.get_folder_hash(path)}"
            except Exception:
                digest = ""

            # Best-effort: populate details using litgpt's built-in config registry.
            config: Any | None = None
            # Try a few likely keys.
            rel = None
            try:
                rel = path.relative_to(checkpoints)
            except Exception:
                rel = None

            candidates: list[str] = [model_name]
            if rel is not None:
                candidates.insert(0, str(rel))
                if rel.parent != Path():
                    candidates.insert(0, f"{rel.parent.as_posix()}/{model_name}")

            for key in candidates:
                try:
                    config = Config.from_name(key)
                    break
                except Exception:
                    continue

            if config is not None:
                parameter_count = estimate_parameter_count(config)
                details = build_model_details(config, parameter_count=parameter_count)
            else:
                details = ModelDetails(
                    family=path.parent.name,
                    format="pytorch",
                    families=[],
                    parameter_size="",
                    quantization_level="",
                    parent_model="",
                )

            self.models[model_name] = TagModel(
                name=model_name,
                model=model_name,
                modified_at=modification_time,
                size=folder_size(path),
                digest=digest,
                details=details,
            )

    def get_model(self, model_name: str) -> dict[str, Any]:
        model = self.models.get(model_name)
        return cast(TagModel, model).encode() if model else {}

    def copy_model(self, model_name: str, new_model_name: str) -> None:
        self.copy_folder(Path("checkpoints") / model_name, Path("checkpoints") / new_model_name)
        model = self.models.get(model_name)
        if model:
            self.models[new_model_name] = TagModel(
                name=new_model_name,
                model=new_model_name,
                modified_at=datetime.now(),
                size=model.size,
                digest=model.digest,
                details=model.details,
            )

    def delete_model(self, model_name: str) -> None:
        if model_name in self.models:
            del self.models[model_name]

        # Best-effort remove from disk too.
        checkpoints = Path("checkpoints").resolve()
        model_path = (Path("checkpoints") / model_name).resolve()
        try:
            model_path.relative_to(checkpoints)
        except Exception:
            return
        if model_path.is_dir():
            shutil.rmtree(model_path)

    def copy_folder(self, src: Path, dst: Path) -> None:
        if not src.is_dir():
            raise ValueError(f"Source {src} is not a directory")
        if dst.exists():
            raise ValueError(f"Destination {dst} already exists")
        shutil.copytree(src, dst)

    def pull_model(self, model_name: str) -> None:
        auto_download_checkpoint(model_name=model_name)
