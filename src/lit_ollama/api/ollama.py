import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Generator
from datetime import datetime
import inspect
import json
from pathlib import Path
import time
import typing
from typing import Any, cast
import uuid

from fastapi import BackgroundTasks, Request, Response, status
from fastapi.responses import StreamingResponse
from litgpt import LLM
from litgpt.config import name_to_config
from litserve.utils import LitAPIStatus, azip

from lit_ollama.api.lit import LitLLMAPI
from lit_ollama.api.schema.base import ModelDetails, TagModel
from lit_ollama.api.schema.chat import ChatRequest, ChatResponse
from lit_ollama.api.schema.copy import CopyRequest
from lit_ollama.api.schema.create import CreateRequest, CreateResponse
from lit_ollama.api.schema.custom.ls import LsResponse
from lit_ollama.api.schema.delete import DeleteRequest
from lit_ollama.api.schema.embed import EmbedRequest, EmbedResponse
from lit_ollama.api.schema.generate import GenerateRequest, GenerateResponse
from lit_ollama.api.schema.ps import PsResponse
from lit_ollama.api.schema.pull import PullRequest, PullResponse
from lit_ollama.api.schema.push import PushRequest, PushResponse
from lit_ollama.api.schema.show import ShowRequest, ShowResponse
from lit_ollama.api.schema.tags import TagsResponse
from lit_ollama.models.store import ModelStore
from lit_ollama.utils import logging

logger = logging.get_logger(__name__)

import litserve as ls


class ollamaLitApi:  # noqa: N801
    """ollama Web API based on https://github.com/ollama/ollama/blob/main/docs/api.md."""

    def __init__(self) -> None:
        super().__init__()
        self.store = ModelStore()
        print("Starting ollama Web API...")

    def setup(self, server: ls.LitServer) -> None:
        print("Initializing ollama Web API...")
        # used for non-copied endpoint methods
        self._server = server  # overrides LitSpec base implementation
        print("ollama Web API successfully initialized.")

    @property
    def lit_api(self) -> ls.LitAPI:
        return self._server.lit_api

    @property
    def llm_api(self) -> LitLLMAPI:
        return cast(LitLLMAPI, self.lit_api)

    @property
    def llm(self) -> LLM:
        return self.llm_api.llm

    async def ls(self, request: Request, response: Response) -> LsResponse:
        return LsResponse(models=list(name_to_config.keys()))

    ############
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #

    ##############

    async def generate(
        self, request: Request, response: Response, data: GenerateRequest, background_tasks: BackgroundTasks
    ) -> GenerateResponse:
        response_queue_id = cast(ls.LitServer, self).response_queue_id
        logger.debug("Received chat completion request %s", request)
        uid = uuid.uuid4()
        q = deque()
        event = asyncio.Event()
        self._server.response_buffer[uid] = (q, event)
        self._server.request_queue.put((response_queue_id, uid, time.monotonic(), data.prompt))

        responses = self._server.data_streamer(q, event, send_status=True)

        async def streaming_completion(
            data: GenerateRequest, streaming_response: AsyncGenerator
        ) -> AsyncGenerator[str, Any]:
            model = data.model
            async for i, (response, resp_status) in enumerate(streaming_response):  # type: ignore
                if resp_status == LitAPIStatus.ERROR:
                    raise response
                encoded_response = json.loads(response)
                logger.debug(encoded_response)

                yield f"data: {encoded_response.__dict__()}\n\n"

            # yield f"data: {last_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

            #     yield GenerateResponse(
            #         model=model, created_at=datetime.now(), response="data: {encoded_response.__dict__()}\n\n"
            #     )

            # # yield f"data: {last_chunk.model_dump_json()}\n\n"
            # yield GenerateResponse(model=model, created_at=datetime.now(), response="data: [DONE]\n\n")

        async def non_streaming_completion(
            data: GenerateRequest, streaming_response: AsyncGenerator
        ) -> GenerateResponse:
            model = data.model
            msgs = []
            content = ""
            async for response, resp_status in streaming_response:
                if resp_status == LitAPIStatus.ERROR:
                    raise response
                # data from LitAPI.encode_response
                encoded_response = json.loads(response)
                logger.debug(encoded_response)

                msgs.append(encoded_response)

            # if string, different handling for dicts
            content = "".join(msgs)

            return GenerateResponse(model=model, created_at=datetime.now(), response=content)

        if data.stream:
            return StreamingResponse(
                streaming_completion(data, responses),
                media_type="text/event-stream",
                background=background_tasks,
            )  # type: ignore  # noqa: PGH003

        response_task = asyncio.create_task(non_streaming_completion(data, responses))
        return await response_task

        return GenerateResponse(model="", created_at=datetime.now(), response="")

    async def chat(
        self, request: Request, response: Response, data: ChatRequest, background_tasks: BackgroundTasks
    ) -> ChatResponse:
        return ChatResponse(model="", created_at=datetime.now(), message=None)

    async def create(
        self, request: Request, response: Response, data: CreateRequest, background_tasks: BackgroundTasks
    ) -> StreamingResponse:
        def create_stream(data: CreateRequest) -> Generator:
            if data.modelfile:
                yield CreateResponse(status="reading model metadata")
                # system prompts, override llm.prompt_style, use PromptStyle.from_config
                pass

            if data.quantize:
                yield CreateResponse(status=f"quantizing F16 model to {data.quantize}")

            self.llm_api.initialize_model("")
            # {"status":"creating system layer"}
            # {"status":"using already created layer sha256:22f7f8ef5f4c791c1b03d7eb414399294764d7cc82c7e94aa81a1feb80a983a2"}
            # {"status":"using already created layer sha256:8c17c2ebb0ea011be9981cc3922db8ca8fa61e828c5d3f44cb6ae342bf80460b"}
            # {"status":"using already created layer sha256:7c23fb36d80141c4ab8cdbb61ee4790102ebd2bf7aeff414453177d4f2110e5d"}
            # {"status":"using already created layer sha256:2e0493f67d0c8c9c68a8aeacdf6a38a2151cb3c4c1d42accf296e19810527988"}
            # {"status":"using already created layer sha256:2759286baa875dc22de5394b4a925701b1896a7e3f8e53275c36f75a877a82c9"}
            # {"status":"writing layer sha256:df30045fe90f0d750db82a058109cecd6d4de9c90a3d75b19c09e5f64580bb42"}
            # {"status":"writing layer sha256:f18a68eb09bf925bb1b669490407c1b1251c5db98dc4d3d81f3088498ea55690"}

            yield CreateResponse(status="writing manifest")
            yield CreateResponse(status="success")

        return StreamingResponse(
            create_stream(data),
            media_type="text/event-stream",
            background=background_tasks,
        )

    async def blobs(self, request: Request, response: Response, digest: str) -> Response:
        if request.method == "HEAD":
            # https://github.com/ollama/ollama/blob/main/docs/api.md#check-if-a-blob-exists
            if True:
                response.status_code = status.HTTP_200_OK
                return response
            else:
                response.status_code = status.HTTP_404_NOT_FOUND
                return response
        if request.method == "POST":
            # https://github.com/ollama/ollama/blob/main/docs/api.md#create-a-blob
            return Response(content="path", status_code=status.HTTP_201_CREATED)
        response.status_code = status.HTTP_400_BAD_REQUEST
        return response

    async def tags(self, request: Request, response: Response, data: Any) -> TagsResponse:
        models = []
        for path in Path("checkpoints").rglob("*"):
            if path.is_dir():
                stats = path.stat()
                modification_time = datetime.fromtimestamp(stats.st_mtime)
                models.append(
                    TagModel(
                        name=path.name,
                        modified_at=modification_time,
                        size=stats.st_size,
                        digest="",
                        details=ModelDetails(
                            format="huggingface",
                            family="",
                            families=[],
                            parameter_size="",
                            quantization_level="",
                        ),
                    )
                )
        return TagsResponse(models=models)

    async def show(self, request: Request, response: Response, data: ShowRequest) -> ShowResponse:
        return ShowResponse(modelfile="", parameters="", template="", details={}, model_info={})

    async def copy(self, request: Request, response: Response, data: CopyRequest) -> Response:
        response.status_code = status.HTTP_200_OK
        return response

    async def delete(self, request: Request, response: Response, data: DeleteRequest) -> Response:
        self.store.delete_model(data.model)
        response.status_code = status.HTTP_200_OK
        return response

    async def pull(
        self, request: Request, response: Response, data: PullRequest, background_tasks: BackgroundTasks
    ) -> PullResponse:
        if data.stream:

            def digest_status(digest: str, part: int, total: int) -> PullResponse:
                return PullResponse(status=f"downloading {digest}", digest=digest, total=total, completed=part)

            def create_stream(data: PullRequest) -> Generator:
                yield PullResponse(status="pulling manifest")
                for i in range(10, 100, 10):
                    yield digest_status("asd", i, 100)
                    time.sleep(0.1)
                yield PullResponse(status="verifying sha256 digest")
                yield PullResponse(status="writing manifest")
                yield PullResponse(status="removing any unused layers")
                yield PullResponse(status="success")

            return StreamingResponse(
                create_stream(data),
                media_type="text/event-stream",
                background=background_tasks,
            )  # type: ignore  # noqa: PGH003
        else:
            self.store.pull_model(data.model)
            return PullResponse(status="success")

    async def push(
        self, request: Request, response: Response, data: PushRequest, background_tasks: BackgroundTasks
    ) -> PushResponse:
        if data.stream:

            def digest_status(part: int, total: int) -> PushResponse:
                return PushResponse(status="starting upload", digest=str(part), total=total)

            def create_stream(data: PushRequest) -> Generator:
                yield PushResponse(status="retrieving manifest")
                for i in range(10, 100, 10):
                    yield digest_status(i, 100)
                    time.sleep(0.1)
                yield PushResponse(status="pushing manifest")
                yield PushResponse(status="success")

            return StreamingResponse(
                create_stream(data),
                media_type="text/event-stream",
                background=background_tasks,
            )  # type: ignore  # noqa: PGH003
        else:
            self.store.pull_model(data.model)
            return PushResponse(status="success")

    async def embed(self, request: Request, response: Response, data: EmbedRequest) -> EmbedResponse:
        embed = self.llm._text_to_token_ids(data.input)
        return EmbedResponse(model=data.model, embeddings=[], total_duration=0, load_duration=0, prompt_eval_count=0)

    async def ps(self, request: Request, response: Response) -> PsResponse:
        return PsResponse(models=[])
