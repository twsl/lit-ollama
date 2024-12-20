from enum import Enum
from pathlib import Path

from pydantic.dataclasses import dataclass

from lit_ollama.api.schema.base import QuantizationType

# https://github.com/ollama/ollama/blob/main/docs/api.md#create-a-model


@dataclass
class CreateRequest:
    model: str
    modelfile: str | None = None
    stream: bool | None = None
    path: str | Path | None = None
    quantize: QuantizationType | str | None = None


@dataclass
class CreateResponse:
    status: str
