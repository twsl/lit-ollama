from pydantic.dataclasses import dataclass


@dataclass
class LsResponse:
    models: list[str]
