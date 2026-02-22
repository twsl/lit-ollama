import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Generator
from dataclasses import asdict
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

from lit_ollama.__about__ import __version__
from lit_ollama.api.lit import LitLLMAPI
from lit_ollama.api.schema.base import ModelDetails, TagModel
from lit_ollama.api.schema.chat import ChatRequest, ChatResponse, Message
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
from lit_ollama.api.schema.version import VersionResponse
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

    async def version(self, request: Request, response: Response) -> VersionResponse:
        return VersionResponse(version=__version__)

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
        start_time = time.perf_counter_ns()
        load_start = time.perf_counter_ns()

        response_queue_id = cast(ls.LitServer, self).response_queue_id
        logger.debug("Received generate completion request %s", request)

        # Handle empty prompt for model loading/unloading
        if not data.prompt:
            if data.keep_alive == 0:
                # Unload model
                return GenerateResponse(
                    model=data.model, created_at=datetime.now(), response="", done=True, done_reason="unload"
                )
            else:
                # Load model
                load_duration = time.perf_counter_ns() - load_start
                return GenerateResponse(
                    model=data.model,
                    created_at=datetime.now(),
                    response="",
                    done=True,
                    done_reason="load",
                    load_duration=load_duration,
                )

        # Prepare prompt with suffix if provided
        prompt = data.prompt
        if data.suffix:
            # For code completion, suffix comes after the generated text
            pass  # Handle in generation

        # Extract options for model generation
        gen_kwargs = {}
        if data.options:
            opts = data.options if isinstance(data.options, dict) else data.options.__dict__
            if opts.get("temperature") is not None:
                gen_kwargs["temperature"] = opts["temperature"]
            if opts.get("top_k") is not None:
                gen_kwargs["top_k"] = opts["top_k"]
            if opts.get("top_p") is not None:
                gen_kwargs["top_p"] = opts["top_p"]
            if opts.get("seed") is not None:
                gen_kwargs["seed"] = opts["seed"]

        load_duration = time.perf_counter_ns() - load_start

        uid = uuid.uuid4()
        q = deque()
        event = asyncio.Event()
        self._server.response_buffer[uid] = (q, event)

        prompt_eval_start = time.perf_counter_ns()
        self._server.request_queue.put((response_queue_id, uid, time.monotonic(), prompt))

        responses = self._server.data_streamer(q, event, send_status=True)

        async def streaming_completion(
            data: GenerateRequest, streaming_response: AsyncGenerator
        ) -> AsyncGenerator[str, Any]:
            model = data.model
            token_count = 0
            eval_start = time.perf_counter_ns()
            context_tokens = []

            async for response, resp_status in streaming_response:
                if resp_status == LitAPIStatus.ERROR:
                    raise response

                token_text = json.loads(response) if isinstance(response, str) else response
                logger.debug("Streaming token: %s", token_text)

                token_count += 1

                # Stream each token as a separate response
                chunk = GenerateResponse(model=model, created_at=datetime.now(), response=str(token_text), done=False)

                yield f"{json.dumps(chunk.serialize())}\n"

            # Final response with metrics
            eval_duration = time.perf_counter_ns() - eval_start
            total_duration = time.perf_counter_ns() - start_time
            prompt_eval_duration = eval_start - prompt_eval_start

            final_chunk = GenerateResponse(
                model=model,
                created_at=datetime.now(),
                response="",
                done=True,
                done_reason="stop",
                context=context_tokens if context_tokens else None,
                total_duration=total_duration,
                load_duration=load_duration,
                prompt_eval_count=len(prompt.split()) if prompt else 0,
                prompt_eval_duration=prompt_eval_duration,
                eval_count=token_count,
                eval_duration=eval_duration,
            )

            yield f"{json.dumps(final_chunk.serialize())}\n"

        async def non_streaming_completion(
            data: GenerateRequest, streaming_response: AsyncGenerator
        ) -> GenerateResponse:
            model = data.model
            tokens = []
            token_count = 0
            eval_start = time.perf_counter_ns()
            context_tokens = []

            async for response, resp_status in streaming_response:
                if resp_status == LitAPIStatus.ERROR:
                    raise response

                token_text = json.loads(response) if isinstance(response, str) else response
                logger.debug("Token: %s", token_text)
                tokens.append(str(token_text))
                token_count += 1

            content = "".join(tokens)
            eval_duration = time.perf_counter_ns() - eval_start
            total_duration = time.perf_counter_ns() - start_time
            prompt_eval_duration = eval_start - prompt_eval_start

            return GenerateResponse(
                model=model,
                created_at=datetime.now(),
                response=content,
                done=True,
                done_reason="stop",
                context=context_tokens if context_tokens else None,
                total_duration=total_duration,
                load_duration=load_duration,
                prompt_eval_count=len(prompt.split()) if prompt else 0,
                prompt_eval_duration=prompt_eval_duration,
                eval_count=token_count,
                eval_duration=eval_duration,
            )

        if data.stream:
            return StreamingResponse(
                streaming_completion(data, responses),
                media_type="application/x-ndjson",
                background=background_tasks,
            )  # type: ignore  # noqa: PGH003

        response_task = asyncio.create_task(non_streaming_completion(data, responses))
        return await response_task

    async def chat(
        self, request: Request, response: Response, data: ChatRequest, background_tasks: BackgroundTasks
    ) -> ChatResponse:
        start_time = time.perf_counter_ns()
        load_start = time.perf_counter_ns()

        response_queue_id = cast(ls.LitServer, self).response_queue_id
        logger.debug("Received chat completion request %s", request)

        # Handle empty messages for model loading/unloading
        if not data.messages or len(data.messages) == 0:
            if data.keep_alive == "0" or data.keep_alive == 0:
                # Unload model
                return ChatResponse(
                    model=data.model,
                    created_at=datetime.now(),
                    message=Message(role="assistant", content=""),
                    done=True,
                    done_reason="unload",
                )
            else:
                # Load model
                load_duration = time.perf_counter_ns() - load_start
                return ChatResponse(
                    model=data.model,
                    created_at=datetime.now(),
                    message=Message(role="assistant", content=""),
                    done=True,
                    done_reason="load",
                    load_duration=load_duration,
                )

        # Build prompt from messages
        prompt_parts = []
        for msg in data.messages:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            prompt_parts.append(f"{role}: {msg.content}")
        prompt = "\n".join(prompt_parts) + "\nassistant:"

        # Extract options for model generation
        gen_kwargs = {}
        if data.options:
            opts = data.options if isinstance(data.options, dict) else data.options.__dict__
            if opts.get("temperature") is not None:
                gen_kwargs["temperature"] = opts["temperature"]
            if opts.get("top_k") is not None:
                gen_kwargs["top_k"] = opts["top_k"]
            if opts.get("top_p") is not None:
                gen_kwargs["top_p"] = opts["top_p"]
            if opts.get("seed") is not None:
                gen_kwargs["seed"] = opts["seed"]

        load_duration = time.perf_counter_ns() - load_start

        uid = uuid.uuid4()
        q = deque()
        event = asyncio.Event()
        self._server.response_buffer[uid] = (q, event)

        prompt_eval_start = time.perf_counter_ns()
        self._server.request_queue.put((response_queue_id, uid, time.monotonic(), prompt))

        responses = self._server.data_streamer(q, event, send_status=True)

        async def streaming_chat(data: ChatRequest, streaming_response: AsyncGenerator) -> AsyncGenerator[str, Any]:
            model = data.model
            token_count = 0
            eval_start = time.perf_counter_ns()
            content_parts = []

            async for response, resp_status in streaming_response:
                if resp_status == LitAPIStatus.ERROR:
                    raise response

                token_text = json.loads(response) if isinstance(response, str) else response
                logger.debug("Streaming token: %s", token_text)

                token_count += 1
                content_parts.append(str(token_text))

                # Stream each token as a separate response
                chunk = ChatResponse(
                    model=model,
                    created_at=datetime.now(),
                    message=Message(role="assistant", content=str(token_text)),
                    done=False,
                )

                yield f"{json.dumps(chunk.serialize())}\n"

            # Final response with metrics
            eval_duration = time.perf_counter_ns() - eval_start
            total_duration = time.perf_counter_ns() - start_time
            prompt_eval_duration = eval_start - prompt_eval_start

            final_chunk = ChatResponse(
                model=model,
                created_at=datetime.now(),
                message=Message(role="assistant", content=""),
                done=True,
                done_reason="stop",
                total_duration=total_duration,
                load_duration=load_duration,
                prompt_eval_count=len(prompt.split()),
                prompt_eval_duration=prompt_eval_duration,
                eval_count=token_count,
                eval_duration=eval_duration,
            )

            yield f"{json.dumps(final_chunk.serialize())}\n"

        async def non_streaming_chat(data: ChatRequest, streaming_response: AsyncGenerator) -> ChatResponse:
            model = data.model
            tokens = []
            token_count = 0
            eval_start = time.perf_counter_ns()

            async for response, resp_status in streaming_response:
                if resp_status == LitAPIStatus.ERROR:
                    raise response

                token_text = json.loads(response) if isinstance(response, str) else response
                logger.debug("Token: %s", token_text)
                tokens.append(str(token_text))
                token_count += 1

            content = "".join(tokens)
            eval_duration = time.perf_counter_ns() - eval_start
            total_duration = time.perf_counter_ns() - start_time
            prompt_eval_duration = eval_start - prompt_eval_start

            return ChatResponse(
                model=model,
                created_at=datetime.now(),
                message=Message(role="assistant", content=content),
                done=True,
                done_reason="stop",
                total_duration=total_duration,
                load_duration=load_duration,
                prompt_eval_count=len(prompt.split()),
                prompt_eval_duration=prompt_eval_duration,
                eval_count=token_count,
                eval_duration=eval_duration,
            )

        if data.stream:
            return StreamingResponse(
                streaming_chat(data, responses),
                media_type="application/x-ndjson",
                background=background_tasks,
            )  # type: ignore  # noqa: PGH003

        response_task = asyncio.create_task(non_streaming_chat(data, responses))
        return await response_task

    async def create(
        self, request: Request, response: Response, data: CreateRequest, background_tasks: BackgroundTasks
    ) -> StreamingResponse:
        # If a raw modelfile string was provided, parse it and fill in missing structured fields.
        if data.modelfile and not data.from_:
            from lit_ollama.models.file import ModelFile

            mf = ModelFile.from_string(data.modelfile)
            if mf.base and not data.from_:
                data.from_ = mf.base
            if mf.system and not data.system:
                data.system = mf.system
            if mf.template and not data.template:
                data.template = mf.template
            if mf.license and not data.license:
                data.license = mf.license
            if mf.parameters and not data.parameters:
                data.parameters = mf.parameters
            if mf.messages and not data.messages:
                from lit_ollama.api.schema.chat import Message

                data.messages = [Message(role=m["role"], content=m["content"]) for m in mf.messages]

        def create_stream(data: CreateRequest) -> Generator:
            if data.from_:
                yield CreateResponse(status=f"reading model metadata from {data.from_}")

            if data.system:
                yield CreateResponse(status="creating system layer")

            if data.files:
                for name, digest in data.files.items():
                    yield CreateResponse(status=f"using file {name} ({digest})")

            if data.adapters:
                for name, digest in data.adapters.items():
                    yield CreateResponse(status=f"using adapter {name} ({digest})")

            if data.quantize:
                yield CreateResponse(status=f"quantizing F16 model to {data.quantize}")

            self.llm_api.initialize_model("")

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

    async def tags(self, request: Request, response: Response) -> TagsResponse:
        models = []
        for path in Path("checkpoints").rglob("*"):
            if path.is_dir():
                stats = path.stat()
                modification_time = datetime.fromtimestamp(stats.st_mtime)
                model_name = path.name
                models.append(
                    TagModel(
                        name=model_name,
                        model=model_name,
                        modified_at=modification_time,
                        size=stats.st_size,
                        digest="",
                        details=ModelDetails(
                            format="huggingface",
                            family="",
                            families=[],
                            parameter_size="",
                            quantization_level="",
                            parent_model="",
                        ),
                    )
                )
        return TagsResponse(models=models)

    async def show(self, request: Request, response: Response, data: ShowRequest) -> ShowResponse:
        from lit_ollama.models.file import ModelFile

        mf = ModelFile()
        mf.set_base(data.model)

        # Build parameters string (one per line, matching ``ollama show`` output)
        param_lines = []
        for k, v in mf.parameters.items():
            if isinstance(v, list):
                for item in v:
                    param_lines.append(f"{k}    {item}")
            else:
                param_lines.append(f"{k}    {v}")

        return ShowResponse(
            modelfile=mf.render(),
            parameters="\n".join(param_lines),
            template=mf.template or "",
            details={},
            model_info={},
            capabilities=["completion"],
        )

    async def copy(self, request: Request, response: Response, data: CopyRequest) -> Response:
        # Return 404 if source model not found
        source_path = Path("checkpoints") / data.source
        if not source_path.exists():
            response.status_code = status.HTTP_404_NOT_FOUND
            return response
        response.status_code = status.HTTP_200_OK
        return response

    async def delete(self, request: Request, response: Response, data: DeleteRequest) -> Response:
        model_path = Path("checkpoints") / data.model
        if not model_path.exists():
            response.status_code = status.HTTP_404_NOT_FOUND
            return response
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
        prompt_eval_start = time.perf_counter_ns()
        if isinstance(data.input, str):
            embed = [self.llm._text_to_token_ids(data.input)]
        else:
            embed = [self.llm._text_to_token_ids(text) for text in data.input]

        embed_list = [e.tolist() for e in embed]
        duration = time.perf_counter_ns() - prompt_eval_start
        return EmbedResponse(
            model=data.model, embeddings=embed_list, total_duration=duration, load_duration=0, prompt_eval_count=0
        )

    async def ps(self, request: Request, response: Response) -> PsResponse:
        return PsResponse(models=[])
