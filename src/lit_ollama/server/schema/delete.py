from pydantic.dataclasses import dataclass


@dataclass
class DeleteRequest:
    model: str
