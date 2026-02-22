from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from litgpt import LLM
import torch
from torch import nn


@dataclass
class MockConfig:
    name: str = "mock"
    hf_config: dict[str, Any] = None  # type: ignore[assignment]
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 2
    n_head: int = 4
    n_query_groups: int | None = None
    n_embd: int = 64
    intermediate_size: int | None = None
    norm_eps: float = 1e-5
    rope_base: int = 10000
    rope_n_elem: int | None = None

    def __post_init__(self) -> None:
        if self.hf_config is None:
            self.hf_config = {"org": "mock", "name": "mock"}
        if self.n_query_groups is None:
            self.n_query_groups = self.n_head
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.n_embd
        if self.rope_n_elem is None:
            # Common default: a fraction of head size; keep simple for mock.
            head_size = self.n_embd // self.n_head if self.n_head else self.n_embd
            self.rope_n_elem = min(128, head_size)


class MockTokenizer:
    vocab_size: int = 32000
    bos_id: int = 1
    eos_id: int = 2
    backend: str = "mock"

    def encode(self, text: str) -> torch.Tensor:
        tokens = [abs(hash(w)) % self.vocab_size for w in text.split()]
        if not tokens:
            tokens = [0]
        return torch.tensor(tokens, dtype=torch.long)


class _MockEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, n_embd) * 0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class _MockTransformer(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int) -> None:
        super().__init__()
        self.wte = _MockEmbedding(vocab_size, n_embd)


class _MockModel(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int) -> None:
        super().__init__()
        self.transformer = _MockTransformer(vocab_size, n_embd)


class MockLLM(LLM):
    def __init__(self) -> None:
        # Bypass LLM.__init__ (requires a real GPT model) but call
        # nn.Module.__init__ so PyTorch bookkeeping (_modules, _parameters, etc.) is set up.
        nn.Module.__init__(self)
        self.config = MockConfig()
        self._tokenizer = MockTokenizer()
        self.model = _MockModel(vocab_size=self.config.vocab_size, n_embd=self.config.n_embd)
        self.checkpoint_dir = Path("checkpoints") / "mock"

    @property
    def tokenizer(self) -> MockTokenizer:
        return self._tokenizer

    def _text_to_token_ids(self, text: str, **_: Any) -> torch.Tensor:  # ty:ignore[invalid-method-override]
        return self._tokenizer.encode(text)

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
    ) -> str | torch.Tensor | Generator[str, None, None]:  # ty:ignore[invalid-method-override]
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

    @torch.inference_mode()
    def benchmark(
        self, num_iterations: int = 1, **kwargs
    ) -> tuple[str | torch.Tensor | Generator[str, None, None], dict[str, Any]]:
        benchmark_dict: dict[str, Any] = {}
        t0 = time.perf_counter()
        outputs = self.generate(**kwargs)
        benchmark_dict.setdefault("Seconds total", []).append(time.perf_counter() - t0)

        benchmark_dict.setdefault("Tokens generated", []).append(2)
        return outputs, benchmark_dict
