from datetime import datetime
from typing import Any, Literal

from pydantic.dataclasses import dataclass

from lit_ollama.api.schema.base import GenerateOptions, StructuredOutputFormat

# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion


@dataclass
class GenerateRequest:
    model: str
    prompt: str | None = None
    stream: bool | None = True
    suffix: str | None = None
    options: GenerateOptions | dict[str, Any] | None = None
    format: StructuredOutputFormat | Literal["json"] | None = None
    images: list[str] | None = None
    raw: bool | None = None
    keep_alive: bool | None = None  # make 0/1
    think: bool | None = None


@dataclass
class GenerateResponse:
    model: str
    created_at: str | datetime
    response: str
    done: bool = True
    done_reason: str | None = None
    context: list[int] | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None

    def serialize(self):
        data = self.__dict__.copy()
        if isinstance(self.created_at, datetime):
            data["created_at"] = self.created_at.isoformat()
        # Remove None values to keep response clean
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def deserialize(cls, data):
        if "created_at" in data and isinstance(data["created_at"], str):
            try:  # noqa: SIM105
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            except ValueError:
                pass
        return cls(**data)
