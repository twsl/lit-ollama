from collections.abc import Generator
import time
from typing import Any

from litgpt import GPT, LLM, Config
import torch
from torch import nn


class MockLLM(LLM):
    def __init__(self) -> None:
        # super().__init__(nn.Identity())  # pyright: ignore[reportArgumentType]
        pass

    @torch.inference_mode()
    def generate(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        prompt: str,
        sys_prompt: str | None = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        return_as_token_ids: bool = False,
        stream: bool = False,
    ) -> str | torch.Tensor | Generator[str, None, None]:
        result = ["Hello, ", "World!"]
        print("MockLLM.generate called")
        if stream:
            # Don't use yield directly in the method, which made the entire function a generator, even when stream=False
            def generator() -> Generator[str, Any, None]:
                for r in result:
                    time.sleep(2)
                    yield r

            return generator()
        else:
            time.sleep(4)
            return "".join(result)

    def benchmark(
        self, num_iterations: int = 1, **kwargs
    ) -> tuple[str | torch.Tensor | Generator[str, None, None], dict[str, Any]]:
        benchmark_dict: dict[str, Any] = {}
        t0 = time.perf_counter()
        outputs = self.generate(**kwargs)
        benchmark_dict.setdefault("Seconds total", []).append(time.perf_counter() - t0)

        benchmark_dict.setdefault("Tokens generated", []).append(2)
        return outputs, benchmark_dict
