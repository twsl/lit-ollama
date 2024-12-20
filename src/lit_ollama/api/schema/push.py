from datetime import datetime

from pydantic.dataclasses import dataclass

# https://github.com/ollama/ollama/blob/main/docs/api.md#push-a-model


@dataclass
class PushRequest:
    model: str
    insecure: bool | None = None
    stream: bool | None = None


@dataclass
class PushResponse:
    status: str
    digest: str | None = None
    total: int | None = None
