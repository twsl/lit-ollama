import pytest

from lit_ollama.models.mock import MockLLM


@pytest.fixture(scope="module")
def mock_llm() -> MockLLM:
    return MockLLM()


def test_mockllm_generate(mock_llm: MockLLM) -> None:
    out = mock_llm.generate("hi", stream=False)
    assert isinstance(out, str)
    out_stream = list(mock_llm.generate("say hi", stream=True))
    assert all(isinstance(x, str) for x in out_stream)


def test_mockllm_benchmark(mock_llm: MockLLM) -> None:
    out, bench = mock_llm.benchmark(prompt="hi")
    if hasattr(out, "__iter__"):
        list(out)
    assert "Seconds total" in bench
