from typing import Any

from pydantic import Field
from pydantic.dataclasses import dataclass

from lit_ollama.api.schema.base import QuantizationType
from lit_ollama.api.schema.chat import Message

# https://github.com/ollama/ollama/blob/main/docs/api.md#create-a-model


@dataclass
class CreateRequest:
    model: str
    from_: str | None = Field(default=None, alias="from")
    files: dict[str, str] | None = None
    adapters: dict[str, str] | None = None
    template: str | None = None
    license: str | list[str] | None = None
    system: str | None = None
    parameters: dict[str, Any] | None = None
    messages: list[Message] | None = None
    stream: bool | None = None
    quantize: QuantizationType | str | None = None


@dataclass
class CreateResponse:
    status: str
