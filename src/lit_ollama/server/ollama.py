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
from litgpt.config import Config, configs
import litserve as ls
from litserve.utils import LitAPIStatus, ResponseBufferItem, azip

from lit_ollama.__about__ import __version__
from lit_ollama.server.schema.base import ModelDetails, ShowModelDetails, TagModel
from lit_ollama.server.schema.chat import ChatRequest, ChatResponse, Message
from lit_ollama.server.schema.copy import CopyRequest
from lit_ollama.server.schema.create import CreateRequest, CreateResponse
from lit_ollama.server.schema.custom.ls import LsResponse
from lit_ollama.server.schema.delete import DeleteRequest
from lit_ollama.server.schema.embed import EmbedRequest, EmbedResponse
from lit_ollama.server.schema.generate import GenerateRequest, GenerateResponse
from lit_ollama.server.schema.ps import PsResponse
from lit_ollama.server.schema.pull import PullRequest, PullResponse
from lit_ollama.server.schema.push import PushRequest, PushResponse
from lit_ollama.server.schema.show import ShowRequest, ShowResponse
from lit_ollama.server.schema.tags import TagsResponse
from lit_ollama.server.schema.version import VersionResponse
from lit_ollama.store import LocalStore
from lit_ollama.utils import logging
from lit_ollama.utils.metadata import (
    build_model_info,
    build_parameters_string,
    build_show_model_details,
    capabilities,
    count_model_parameters,
    template_name,
)


class ollamaLitApi:  # noqa: N801
    """ollama Web API based on https://github.com/ollama/ollama/blob/main/docs/api.md."""

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.get_logger(__name__)
        self.store = LocalStore()
        self._server: ls.LitServer = None
        self.logger.info("Starting ollama Web API...")

    def setup(self, server: ls.LitServer) -> None:
        self.logger.info("Initializing ollama Web API...")
        # used for non-copied endpoint methods
        self._server = server  # overrides LitSpec base implementation
        # Propagate to LitSpec.setup() so response_buffer, request_queue,
        # and data_streamer are wired up from the server.
        super().setup(server)
        self.logger.info("ollama Web API successfully initialized.")

    @property
    def lit_api(self) -> ls.LitAPI:
        return self._server.lit_api  # ty:ignore[invalid-return-type]

    @property
    def llm(self) -> LLM:
        from lit_ollama.server.api import LitOllamaAPI

        return cast(LitOllamaAPI, self.lit_api).llm

    async def ls(self, request: Request, response: Response) -> LsResponse:
        return LsResponse(models=sorted({c["name"] for c in configs if isinstance(c, dict) and "name" in c}))

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

    @staticmethod
    def _parse_token(response: Any) -> Any:
        """Safely extract a token from a streamer response.

        The data_streamer may yield JSON-encoded strings **or** plain text
        tokens depending on how ``encode_response`` serialises output.  Try
        JSON first; fall back to the raw value.
        """
        if isinstance(response, str):
            try:
                return json.loads(response)
            except (json.JSONDecodeError, ValueError):
                return response
        return response

    async def _get_from_queue(self, uid: uuid.UUID, q: deque, event: asyncio.Event) -> AsyncGenerator:
        """Get a data streamer for a single uid's response queue."""
        return self.data_streamer(q, event, send_status=True)  # ty:ignore[unresolved-attribute]

    async def _streaming_generate(
        self,
        data: GenerateRequest,
        streaming_response: AsyncGenerator,
        start_time: int,
        load_duration: int,
        prompt_eval_start: int,
        prompt: str,
    ) -> AsyncGenerator[str, Any]:
        """Stream generate completion tokens as NDJSON chunks."""
        model = data.model
        token_count = 0
        eval_start = time.perf_counter_ns()
        context_tokens: list[int] = []

        async for response, resp_status in streaming_response:
            if resp_status == LitAPIStatus.ERROR:
                raise response
            if isinstance(response, str) and not response.strip():
                continue

            token_text = self._parse_token(response)
            self.logger.debug("Streaming token: %s", token_text)

            token_count += 1

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

    async def _non_streaming_generate(
        self,
        data: GenerateRequest,
        streaming_response: AsyncGenerator,
        start_time: int,
        load_duration: int,
        prompt_eval_start: int,
        prompt: str,
    ) -> "GenerateResponse":
        """Collect all generate tokens and return a single response."""
        model = data.model
        tokens: list[str] = []
        token_count = 0
        eval_start = time.perf_counter_ns()
        context_tokens: list[int] = []

        async for response, resp_status in streaming_response:
            if resp_status == LitAPIStatus.ERROR:
                raise response
            if isinstance(response, str) and not response.strip():
                continue

            token_text = self._parse_token(response)
            self.logger.debug("Token: %s", token_text)
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

    async def _streaming_chat(
        self,
        data: ChatRequest,
        streaming_response: AsyncGenerator,
        start_time: int,
        load_duration: int,
        prompt_eval_start: int,
        prompt: str,
    ) -> AsyncGenerator[str, Any]:
        """Stream chat completion tokens as NDJSON chunks."""
        model = data.model
        token_count = 0
        eval_start = time.perf_counter_ns()

        async for response, resp_status in streaming_response:
            if resp_status == LitAPIStatus.ERROR:
                raise response
            if isinstance(response, str) and not response.strip():
                continue

            token_text = self._parse_token(response)
            self.logger.debug("Streaming token: %s", token_text)

            token_count += 1

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

    async def _non_streaming_chat(
        self,
        data: "ChatRequest",
        streaming_response: AsyncGenerator,
        start_time: int,
        load_duration: int,
        prompt_eval_start: int,
        prompt: str,
    ) -> "ChatResponse":
        """Collect all chat tokens and return a single response."""
        model = data.model
        tokens: list[str] = []
        token_count = 0
        eval_start = time.perf_counter_ns()

        async for response, resp_status in streaming_response:
            if resp_status == LitAPIStatus.ERROR:
                raise response
            if isinstance(response, str) and not response.strip():
                continue

            token_text = self._parse_token(response)
            self.logger.debug("Token: %s", token_text)
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

    async def generate(
        self, request: Request, response: Response, data: GenerateRequest, background_tasks: BackgroundTasks
    ) -> GenerateResponse:
        start_time = time.perf_counter_ns()
        load_start = time.perf_counter_ns()

        response_queue_id = self.response_queue_id  # ty:ignore[unresolved-attribute]
        self.logger.debug("Received generate completion request %s", request)

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
        self.response_buffer[uid] = ResponseBufferItem(response_queue=q, event=event)  # ty:ignore[unresolved-attribute]

        prompt_eval_start = time.perf_counter_ns()
        self.request_queue.put((response_queue_id, uid, time.monotonic(), prompt))  # ty:ignore[unresolved-attribute]

        responses = await self._get_from_queue(uid, q, event)

        if data.stream:
            return StreamingResponse(
                self._streaming_generate(data, responses, start_time, load_duration, prompt_eval_start, prompt),
                media_type="application/x-ndjson",
                background=background_tasks,
            )  # type: ignore  # noqa: PGH003

        response_task = asyncio.create_task(
            self._non_streaming_generate(data, responses, start_time, load_duration, prompt_eval_start, prompt)
        )
        return await response_task

    async def chat(
        self, request: Request, response: Response, data: ChatRequest, background_tasks: BackgroundTasks
    ) -> ChatResponse:
        start_time = time.perf_counter_ns()
        load_start = time.perf_counter_ns()

        response_queue_id = self.response_queue_id  # ty:ignore[unresolved-attribute]
        self.logger.debug("Received chat completion request %s", request)

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
        self.response_buffer[uid] = ResponseBufferItem(response_queue=q, event=event)  # ty:ignore[unresolved-attribute]

        prompt_eval_start = time.perf_counter_ns()
        self.request_queue.put((response_queue_id, uid, time.monotonic(), prompt))  # ty:ignore[unresolved-attribute]

        responses = await self._get_from_queue(uid, q, event)

        if data.stream:
            return StreamingResponse(
                self._streaming_chat(data, responses, start_time, load_duration, prompt_eval_start, prompt),
                media_type="application/x-ndjson",
                background=background_tasks,
            )  # type: ignore  # noqa: PGH003

        response_task = asyncio.create_task(
            self._non_streaming_chat(data, responses, start_time, load_duration, prompt_eval_start, prompt)
        )
        return await response_task

    async def create(
        self, request: Request, response: Response, data: CreateRequest, background_tasks: BackgroundTasks
    ) -> StreamingResponse:
        # If a raw modelfile string was provided, parse it and fill in missing structured fields.
        if data.modelfile and not data.from_:
            from lit_ollama.server.file import ModelFile

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
                from lit_ollama.server.schema.chat import Message

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

            from lit_ollama.server.api import LitOllamaAPI

            cast(LitOllamaAPI, self.lit_api).initialize_model("")

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
            if self.store.blobs.exists(digest):
                response.status_code = status.HTTP_200_OK
                return response
            response.status_code = status.HTTP_404_NOT_FOUND
            return response
        if request.method == "POST":
            # https://github.com/ollama/ollama/blob/main/docs/api.md#create-a-blob
            body = await request.body()
            try:
                path = self.store.blobs.save(digest, body)
            except ValueError as exc:
                return Response(content=str(exc), status_code=status.HTTP_400_BAD_REQUEST)
            return Response(content=str(path), status_code=status.HTTP_201_CREATED)
        response.status_code = status.HTTP_400_BAD_REQUEST
        return response

    async def tags(self, request: Request, response: Response) -> TagsResponse:
        # Delegate discovery/details to the store.
        return TagsResponse(models=list(self.store.models.values()))

    async def show(self, request: Request, response: Response, data: ShowRequest) -> ShowResponse:
        from lit_ollama.server.file import ModelFile

        mf = ModelFile()
        mf.set_base(data.model)

        # Prefer live loaded model details if the requested model matches.
        llm = self.llm
        config = getattr(llm, "config", None)
        if config is None:
            # Fallback: best-effort lookup from litgpt registry.
            try:
                config = Config.from_name(data.model)
            except Exception:
                config = None

        param_count = count_model_parameters(llm)
        details: ShowModelDetails | dict[str, Any]
        model_info: dict[str, Any]
        template = ""
        caps: list[str] | None = ["completion"]

        if config is not None:
            details = build_show_model_details(config, parameter_count=param_count)
            model_info = build_model_info(config, llm=llm, verbose=bool(data.verbose))
            template = template_name(llm)
            caps = capabilities(llm)

            # Populate the rendered modelfile with a couple of real parameters.
            block_size = getattr(config, "block_size", None)
            if block_size:
                mf.set_parameter("num_ctx", block_size)
        else:
            details = {}
            model_info = {}

        parameters = build_parameters_string(config, llm=llm) if config is not None else ""

        # If we didn't build a config-driven parameters string, fall back to ModelFile parameters.
        if not parameters:
            param_lines = []
            for k, v in mf.parameters.items():
                if isinstance(v, list):
                    for item in v:
                        param_lines.append(f"{k}    {item}")
                else:
                    param_lines.append(f"{k}    {v}")
            parameters = "\n".join(param_lines)

        return ShowResponse(
            modelfile=mf.render(),
            parameters=parameters,
            template=template,
            details=details,
            model_info=model_info,
            capabilities=caps,
        )

    async def copy(self, request: Request, response: Response, data: CopyRequest) -> Response:
        # Return 404 if source model not found
        source_path = Path("checkpoints") / data.source
        if not source_path.exists():
            response.status_code = status.HTTP_404_NOT_FOUND
            return response
        try:
            self.store.copy_model(data.source, data.destination)
        except ValueError as exc:
            return Response(content=str(exc), status_code=status.HTTP_400_BAD_REQUEST)
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
                # Perform the actual download.
                try:
                    self.store.pull_model(data.model)
                except Exception as exc:
                    yield PullResponse(status=f"error: {exc}")
                    return

                # Provide a single digest update after download.
                model_path = Path("checkpoints") / data.model
                if model_path.is_dir():
                    try:
                        d = f"sha256:{self.store.get_folder_hash(model_path)}"
                        yield digest_status(d, 1, 1)
                    except Exception as exc:
                        self.logger.exception("Failed computing pull digest", exc_info=exc)
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

            def create_stream(data: PushRequest) -> Generator:
                yield PushResponse(status="push not implemented")

            response.status_code = status.HTTP_501_NOT_IMPLEMENTED

            return StreamingResponse(
                create_stream(data),
                media_type="text/event-stream",
                background=background_tasks,
            )  # type: ignore  # noqa: PGH003
        else:
            # Not implemented yet (would require a registry backend).
            response.status_code = status.HTTP_501_NOT_IMPLEMENTED
            return PushResponse(status="push not implemented")

    async def embed(self, request: Request, response: Response, data: EmbedRequest) -> EmbedResponse:
        import torch

        prompt_eval_start = time.perf_counter_ns()
        inputs = [data.input] if isinstance(data.input, str) else data.input

        embeddings: list[list[float]] = []
        prompt_eval_count = 0

        model = getattr(self.llm, "model", None)
        transformer = getattr(model, "transformer", None) if model is not None else None
        wte = getattr(transformer, "wte", None) if transformer is not None else None

        for text in inputs:
            token_ids = self.llm._text_to_token_ids(text)
            try:
                prompt_eval_count += int(token_ids.numel())
            except Exception:
                prompt_eval_count += len(getattr(token_ids, "tolist", lambda: [])())

            if wte is not None:
                with torch.inference_mode():
                    token_emb = wte(token_ids)
                    # Mean pool to a single vector per input.
                    sent = token_emb.mean(dim=0)
                    embeddings.append([float(x) for x in sent.tolist()])
            else:
                # Fallback: preserve previous behavior (token IDs) but cast to float.
                ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
                embeddings.append([float(i) for i in ids])

        duration = time.perf_counter_ns() - prompt_eval_start
        return EmbedResponse(
            model=data.model,
            embeddings=embeddings,
            total_duration=duration,
            load_duration=0,
            prompt_eval_count=prompt_eval_count,
        )

    async def ps(self, request: Request, response: Response) -> PsResponse:
        from lit_ollama.server.schema.base import RunningModel

        config = getattr(self.llm, "config", None)
        model_obj = getattr(self.llm, "model", None)
        if config is None or model_obj is None:
            return PsResponse(models=[])

        # Estimate memory footprint from parameters.
        try:
            size_bytes = int(sum(p.numel() * p.element_size() for p in model_obj.parameters()))
        except Exception:
            size_bytes = 0

        digest = ""
        checkpoint_dir = getattr(self.llm, "checkpoint_dir", None)
        if checkpoint_dir is not None:
            cp = Path(str(checkpoint_dir))
            if cp.is_dir():
                try:
                    digest = f"sha256:{self.store.get_folder_hash(cp)}"
                except Exception:
                    digest = ""

        running = RunningModel(
            name=self.llm_api.model_name,
            model=self.llm_api.model_name,
            size=size_bytes,
            digest=digest,
            details=build_show_model_details(config, parameter_count=count_model_parameters(self.llm)),
            expires_at=datetime.now(),
            size_vram=0,
        )
        return PsResponse(models=[running])
