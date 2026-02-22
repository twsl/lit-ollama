from typing import Any

from pydantic.dataclasses import dataclass

# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings


@dataclass
class EmbedRequest:
    model: str
    input: str | list[str]
    truncate: bool | None = True
    options: dict[str, Any] | None = None
    keep_alive: str | None = "5m"
    dimensions: int | None = None


@dataclass
class EmbedResponse:
    model: str
    embeddings: list[list[float]]
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    total_duration: int
    load_duration: int
    prompt_eval_count: int
