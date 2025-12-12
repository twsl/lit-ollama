from collections.abc import Generator
import time

from litgpt import GPT, LLM, Config
import torch
from torch import nn


class MockLLM(LLM):
    def __init__(self) -> None:
        # super().__init__(nn.Identity())  # pyright: ignore[reportArgumentType]
        pass

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        sys_prompt: str | None = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        return_as_token_ids: bool = False,
        stream: bool = False,
    ) -> str | torch.Tensor:  # | Generator[str, None, None]: # pyright: ignore[reportInvalidTypeForm]
        result = ["Hello, ", "World!"]
        print("MockLLM.generate called")
        if stream:
            for r in result:
                time.sleep(2)
                yield r  # pyright: ignore[reportReturnType]
        else:
            time.sleep(4)
            return "".join(result)

    def benchmark(
        self, num_iterations: int = 1, **kwargs
    ) -> tuple[str | torch.Tensor, dict]:  # Generator[str, None, None]
        benchmark_dict = {}
        t0 = time.perf_counter()
        outputs = self.generate(**kwargs)
        benchmark_dict.setdefault("Seconds total", []).append(time.perf_counter() - t0)

        benchmark_dict.setdefault("Tokens generated", []).append(2)
        return outputs, benchmark_dict
