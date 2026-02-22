from collections.abc import Generator
from typing import Any, cast

from litgpt import LLM
from litgpt.utils import auto_download_checkpoint
import litserve as ls

from lit_ollama.models.mock import MockLLM
from lit_ollama.server.spec import ollamaSpec
from lit_ollama.utils import logging


class LitOllamaAPI(ls.LitAPI):
    def __init__(
        self,
        model_name: str,
        distribute: bool = False,
        stream: bool = True,
        precision: str | None = None,
    ) -> None:
        super().__init__(stream=stream, spec=ollamaSpec())
        self.model_name = model_name
        self.distribute = distribute
        self.precision = precision
        self.llm: LLM = MockLLM()
        self.logger = logging.get_logger(__name__)

    def setup(self, device: str) -> None:
        self.initialize_model(device)

    def initialize_model(self, device: str) -> None:
        self.logger.debug("Initializing model...")
        if self.model_name != "mock":
            self.checkpoint_dir = auto_download_checkpoint(model_name=self.model_name)
            self.llm = LLM.load(model=self.checkpoint_dir.as_posix(), distribute=None if self.distribute else "auto")

        if self.distribute:
            pass
            # self.llm.distribute(
            #     devices=self.devices,
            #     accelerator=accelerator,
            #     quantize=self.quantize,
            #     precision=self.precision,
            #     generate_strategy="sequential" if self.devices is not None and self.devices > 1 else None,
            # )
        self.logger.debug("Model successfully initialized.")

    def get_config(self) -> Any:
        return getattr(self.llm, "config", None)

    def get_prompt_style_name(self) -> str:
        prompt_style = getattr(self.llm, "prompt_style", None)
        return type(prompt_style).__name__ if prompt_style is not None else ""

    def predict(self, prompt: str, context: dict[str, Any] = {}, **kwargs) -> Generator:  # type: ignore  # noqa: PGH003
        if kwargs.get("benchmark", False):
            response, benchmark_dict = cast(
                tuple[Generator, dict[str, Any]],
                self.llm.benchmark(
                    prompt=prompt,
                    max_new_tokens=int(kwargs.get("max_new_tokens", 50)),
                    temperature=float(kwargs.get("temperature", 1)),
                    top_k=kwargs.get("top_k"),
                    top_p=float(kwargs.get("top_p", 1)),
                    stream=True,
                ),
            )
        else:
            response = cast(
                Generator,
                self.llm.generate(
                    prompt,
                    max_new_tokens=int(kwargs.get("max_new_tokens", 50)),
                    temperature=float(kwargs.get("temperature", 1)),
                    top_k=kwargs.get("top_k"),
                    top_p=float(kwargs.get("top_p", 1)),
                    stream=True,
                ),
            )
        if kwargs.get("benchmark", False):
            for token in response:
                yield token, benchmark_dict
        else:
            yield from response
