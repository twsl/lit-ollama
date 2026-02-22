from pydantic.dataclasses import dataclass

# https://github.com/ollama/ollama/blob/main/docs/api.md#copy-a-model


@dataclass
class CopyRequest:
    source: str
    destination: str
