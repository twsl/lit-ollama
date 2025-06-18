import pytest

from lit_ollama.models.mock import MockLLM


@pytest.fixture(scope="module")
def llm() -> MockLLM:
    return MockLLM()


def test_mockllm_generate(llm: MockLLM) -> None:
    out = llm.generate("hi", stream=False)
    assert isinstance(out, str)
    out_stream = list(llm.generate("say hi", stream=True))
    assert all(isinstance(x, str) for x in out_stream)


def test_mockllm_benchmark(llm: MockLLM) -> None:
    out, bench = llm.benchmark(prompt="hi")
    if hasattr(out, "__iter__"):
        list(out)
    assert "Seconds total" in bench
