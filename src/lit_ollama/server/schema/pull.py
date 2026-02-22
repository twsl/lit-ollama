from pydantic.dataclasses import dataclass

from lit_ollama.server.schema.push import PushRequest, PushResponse

# https://github.com/ollama/ollama/blob/main/docs/api.md#pull-a-model


@dataclass
class PullRequest(PushRequest):
    pass


@dataclass
class PullResponse(PushResponse):
    completed: int | None = None
