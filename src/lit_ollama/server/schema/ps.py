from pydantic.dataclasses import dataclass

from lit_ollama.server.schema.base import RunningModel

# https://github.com/ollama/ollama/blob/main/docs/api.md#list-running-models


@dataclass
class PsResponse:
    models: list[RunningModel]
