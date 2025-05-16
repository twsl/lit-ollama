from datetime import datetime
from typing import Literal

from pydantic.dataclasses import dataclass

from lit_ollama.api.schema.base import GenerateOptions, StructuredOutputFormat

# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion


@dataclass
class GenerateRequest:
    model: str
    prompt: str | None = None
    stream: bool | None = True
    suffix: str | None = None
    options: GenerateOptions | dict | None = None
    format: StructuredOutputFormat | Literal["json"] | None = None
    images: list[str] | None = None
    raw: bool | None = None
    keep_alive: bool | None = None  # make 0/1


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
