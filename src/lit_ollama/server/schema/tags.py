from datetime import datetime

from pydantic.dataclasses import dataclass

from lit_ollama.server.schema.base import ModelDetails, TagModel

# https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models


@dataclass
class TagsResponse:
    models: list[TagModel]
