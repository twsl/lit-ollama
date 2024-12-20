from datetime import datetime
from enum import Enum

from pydantic.dataclasses import dataclass


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# https://github.com/ollama/ollama/blob/main/docs/import.md#quantizing-a-model
class QuantizationType(str, Enum):
    q2_K = "q2_K"  # noqa: N815
    q3_K_L = "q3_K_L"  # noqa: N815
    q3_K_M = "q3_K_M"  # noqa: N815
    q3_K_S = "q3_K_S"  # noqa: N815
    q4_0 = "q4_0"
    q4_1 = "q4_1"
    # recommended
    q4_K_M = "q4_K_M"  # noqa: N815
    q4_K_S = "q4_K_S"  # noqa: N815
    q5_0 = "q5_0"
    q5_1 = "q5_1"
    q5_K_M = "q5_K_M"  # noqa: N815
    q5_K_S = "q5_K_S"  # noqa: N815
    q6_K = "q6_K"  # noqa: N815
    # recommended
    q8_0 = "q8_0"


@dataclass
class StructuredOutputObject:
    type: str  # = "object"


@dataclass
class StructuredOutputFormat(StructuredOutputObject):
    properties: dict[str, StructuredOutputObject | dict]
    required: list[str]


@dataclass
class ModelDetails:
    format: str
    family: str
    families: list[str]
    parameter_size: str
    quantization_level: str


@dataclass
class TagModel:
    name: str
    modified_at: str | datetime
    size: int
    digest: str
    details: ModelDetails


@dataclass
class ShowModelDetails(ModelDetails):
    parent_model: str


@dataclass
class RunningModel:
    name: str
    model: str
    size: int
    digest: str
    details: ShowModelDetails
    expires_at: str | datetime
    size_vram: int


@dataclass
class GenerateOptions:
    num_keep: int | None = None
    num_predict: int | None = None
    seed: int | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    typical_p: float | None = None
    repeat_last_n: int | None = None
    temperature: float | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    mirostat: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None
    penalize_newline: bool | None = None
    stop: list[str] | None = None
    numa: bool | None = None
    num_ctx: int | None = None
    num_batch: int | None = None
    num_gpu: int | None = None
    main_gpu: int | None = None
    low_vram: bool | None = None
    vocab_only: bool | None = None
    use_mmap: bool | None = None
    use_mlock: bool | None = None
    num_thread: int | None = None
