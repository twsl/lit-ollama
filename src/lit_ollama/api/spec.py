from collections.abc import Callable, Iterator
import inspect
from typing import Any

import litserve as ls
from litserve.specs.base import LitSpec

from lit_ollama.api.lit import LitLLMAPI
from lit_ollama.api.ollama import ollamaLitApi
from lit_ollama.utils import logging

logger = logging.get_logger(__name__)


class ollamaSpec(ollamaLitApi, LitSpec):  # noqa: N801
    def __init__(self) -> None:
        super().__init__()
        self.add_endpoint("/api/generate", self.generate, ["POST"])
        self.add_endpoint("/api/chat", self.chat, ["POST"])
        self.add_endpoint("/api/create", self.create, ["POST"])
        self.add_endpoint("/api/blobs/{digest}", self.blobs, ["HEAD", "POST"])
        self.add_endpoint("/api/tags", self.tags, ["GET"])
        self.add_endpoint("/api/show", self.show, ["POST"])
        self.add_endpoint("/api/copy", self.copy, ["POST"])
        self.add_endpoint("/api/delete", self.delete, ["DELETE"])
        self.add_endpoint("/api/pull", self.pull, ["POST"])
        self.add_endpoint("/api/push", self.push, ["POST"])
        self.add_endpoint("/api/embed", self.embed, ["POST"])
        self.add_endpoint("/api/ps", self.ps, ["GET"])
        # self.add_endpoint("/api/embeddings", self.embeddings, ["POST"]) # superseded by embed
        self.add_endpoint("/api/ls", self.ls, ["GET"])

    def pre_setup(self, lit_api: ls.LitAPI) -> None:
        from litgpt import LLM

        if not isinstance(lit_api, LitLLMAPI):
            raise TypeError("LitAPI must be an instance of LitLLMApi.")

        if not hasattr(lit_api, "llm") or not isinstance(lit_api.llm, LLM):  # type: ignore  # noqa: PGH003
            raise ValueError("LitAPI must have an attribute 'llm' of type LLM")

        if not inspect.isgeneratorfunction(lit_api.predict):
            raise ValueError("predict is not a generator")

        is_encode_response_original = lit_api.encode_response.__code__ is ls.LitAPI.encode_response.__code__
        if not is_encode_response_original and not inspect.isgeneratorfunction(lit_api.encode_response):
            raise ValueError("encode_response is not a generator")
        pass

    def setup(self, server: ls.LitServer) -> None:
        super().setup(server)
        if not server.stream:
            raise ValueError("stream must be set to `True`")
        print("ollama Spec Setup complete.")

    def populate_context(self, context: dict[str, Any], data: Any) -> None:
        # data_dict = data.dict()
        # data.pop("messages")
        context["raw"] = data

    def add_endpoint(self, path: str, endpoint: Callable, methods: list[str]) -> None:
        """Register an endpoint in the spec."""
        self._endpoints.append((path, endpoint, methods))

    @property
    def endpoints(self) -> list[tuple[str, Callable, list[str]]]:
        return self._endpoints.copy()

    def decode_request(self, request: Any, **kwargs: Any) -> Any:  # pyright: ignore [reportIncompatibleMethodOverride]
        """Convert the request payload to your model input."""
        return request

    # def batch(self, inputs) -> list[Any]:
    #     return list(inputs)

    # def unbatch(self, output) -> Generator[Any, Any, None]:
    #     yield output

    def _encode_response(self, output: Any) -> dict[str, Any]:
        logger.debug(output)
        return output
        return {"role": "assistant", "content": output}
        # if isinstance(output, str):
        #     message = {"role": "assistant", "content": output}

        # elif self.validate_chat_message(output):
        #     message = output
        # elif isinstance(output, dict) and "content" in output:
        #     message = output.copy()
        #     message.update(role="assistant")
        # elif isinstance(output, list) and output and self.validate_chat_message(output[-1]):
        #     message = output[-1]
        # else:
        #     error = (
        #         "Malformed output from LitAPI.predict: expected "
        #         f"string or {{'role': '...', 'content': '...'}}, got '{output}'."
        #     )
        #     logger.error(error)
        #     raise HTTPException(500, error)
        # return {**message}

    def encode_response(self, output: Any, **kwargs: Any) -> Iterator:  # pyright: ignore [reportIncompatibleMethodOverride]
        """Convert the model output to a response payload.

        To enable streaming, it should yield the output.

        """
        if inspect.isgenerator(output):
            for out in output:
                logger.debug(f"spec out: {out}")
                yield self._encode_response(out)
        elif isinstance(output, list | tuple):
            output, bench = output
            for out in output:
                logger.debug(f"spec out: {out}")
                yield self._encode_response(out)
        else:
            logger.debug(f"spec out: {output}")
            yield self._encode_response(output)
