from pydantic.dataclasses import dataclass

# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings


@dataclass
class EmbedRequest:
    model: str
    input: str | list[str]
    truncate: bool | None = True
    options: dict | None = None
    leep_alive: str | None = "5m"


@dataclass
class EmbedResponse:
    model: str
    embeddings: list[list[float]]
    total_duration: int
    load_duration: int
    prompt_eval_count: int
