import pytest

from lit_ollama.models.mock import MockLLM


@pytest.fixture(scope="module")
def llm():
    return MockLLM()


def test_mockllm_generate(llm: MockLLM):
    out = llm.generate("hi", stream=False)
    assert isinstance(out, str)
    out_stream = list(llm.generate("hi", stream=True))
    assert all(isinstance(x, str) for x in out_stream)


def test_mockllm_benchmark(llm: MockLLM):
    out, bench = llm.benchmark(prompt="hi")
    if hasattr(out, "__iter__"):
        list(out)
    assert "Seconds total" in bench
