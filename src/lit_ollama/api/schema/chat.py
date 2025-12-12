from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic.dataclasses import dataclass

from lit_ollama.api.schema.base import GenerateOptions, Role, StructuredOutputFormat, StructuredOutputObject

# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion


@dataclass
class Message:
    role: str | Role
    content: str
    images: list[str] | None = None
    tool_calls: list[str] | None = None

    def serialize(self):
        data = {
            "role": self.role if isinstance(self.role, str) else self.role.value,
            "content": self.content,
        }
        if self.images is not None:
            data["images"] = self.images
        if self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        return data


@dataclass
class ToolParameters(StructuredOutputObject):
    pass


@dataclass
class ToolFunction:
    name: str
    description: str
    parameters: ToolParameters


@dataclass
class Tool(StructuredOutputObject):
    function: ToolFunction


@dataclass
class ChatRequest:
    model: str
    messages: list[Message] | None = None
    tools: list[Tool] | None = None
    format: Literal["json"] | StructuredOutputFormat | None = None
    options: GenerateOptions | dict[str, Any] | None = None
    stream: bool | None = True
    keep_alive: str | None = "5m"


@dataclass
class ChatResponse:
    model: str
    created_at: str | datetime
    message: Message | None = None
    done_reason: str | None = None
    done: bool = True
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0

    def serialize(self):
        data = self.__dict__.copy()
        if isinstance(self.created_at, datetime):
            data["created_at"] = self.created_at.isoformat()
        if self.message is not None:
            data["message"] = self.message.serialize()
        # Remove None values and zero durations to keep response clean
        return {k: v for k, v in data.items() if v is not None and (not isinstance(v, int) or v > 0 or k == "done")}

    @classmethod
    def deserialize(cls, data):
        if "created_at" in data and isinstance(data["created_at"], str):
            try:  # noqa: SIM105
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            except ValueError:
                pass
        return cls(**data)
