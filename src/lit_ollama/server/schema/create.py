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
    # https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx#valid-parameters-and-values
    parameters: dict[str, Any] | None = None
    messages: list[Message] | None = None
    stream: bool | None = None
    quantize: QuantizationType | str | None = None


@dataclass
class CreateResponse:
    status: str
    digest: str | None = None
    total: int | None = None
    completed: int | None = None

    def serialize(self) -> dict[str, Any]:
        d = {
            "status": self.status,
        }
        if self.digest is not None:
            d["digest"] = self.digest
        if self.total is not None:
            d["total"] = self.total
        if self.completed is not None:
            d["completed"] = self.completed
        return d
