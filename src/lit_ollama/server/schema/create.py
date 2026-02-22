from typing import Any

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from lit_ollama.server.schema.base import QuantizationType
from lit_ollama.server.schema.chat import Message

# https://github.com/ollama/ollama/blob/main/docs/api.md#create-a-model


@dataclass(config=ConfigDict(populate_by_name=True))
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
    modelfile: str | None = None  # raw Modelfile text (legacy / convenience)


@dataclass
class CreateResponse:
    status: str
