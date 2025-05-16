from collections.abc import Generator
from typing import cast

from litgpt import LLM
from litgpt.utils import auto_download_checkpoint
import litserve as ls

from lit_ollama.models.mock import MockLLM
from lit_ollama.utils import logging

logger = logging.get_logger(__name__)


class LitLLMAPI(ls.LitAPI):
    llm: LLM

    def __init__(
        self,
        model_name: str,
        distribute: bool = False,
        precision: str | None = None,
        max_new_tokens: int = 50,
    ) -> None:
        self.model_name = model_name
        self.distribute = distribute
        self.precision = precision
        self.max_new_tokens = max_new_tokens
        self.checkpoint_dir = auto_download_checkpoint(model_name=model_name)

    def setup(self, device: str) -> None:
        self.initialize_model(device)

    def initialize_model(self, device: str) -> None:
        print("Initializing model...")
        # self.llm = LLM.load(model=self.checkpoint_dir.as_posix(), distribute=None if self.distribute else "auto")
        self.llm = MockLLM()

        if self.distribute:
            pass
            # self.llm.distribute(
            #     devices=self.devices,
            #     accelerator=accelerator,
            #     quantize=self.quantize,
            #     precision=self.precision,
            #     generate_strategy="sequential" if self.devices is not None and self.devices > 1 else None,
            # )
        print("Model successfully initialized.")

    def predict(self, prompt: str, context: dict = {}, **kwargs) -> Generator:  # type: ignore  # noqa: PGH003
        if kwargs.get("benchmark", False):
            response, benchmark_dict = cast(
                tuple[Generator, dict],
                self.llm.benchmark(
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
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
                    max_new_tokens=self.max_new_tokens,
                    temperature=float(kwargs.get("temperature", 1)),
                    top_k=kwargs.get("top_k"),
                    top_p=float(kwargs.get("top_p", 1)),
                    stream=True,
                ),
            )
        return response
