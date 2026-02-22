from datetime import datetime

from pydantic.dataclasses import dataclass

from lit_ollama.server.schema.base import ModelDetails, TagModel

# https://github.com/ollama/ollama/blob/main/docs/api.md#version


@dataclass
class VersionResponse:
    version: str
