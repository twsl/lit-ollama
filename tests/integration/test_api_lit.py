import pytest

from lit_ollama.api.lit import LitLLMAPI


@pytest.fixture(scope="module")
def api():
    api = LitLLMAPI("meta-llama/Llama-3.2-1B-Instruct")
    return api


def test_litllmapi_init(api: LitLLMAPI):
    assert api.model_name == "llama"
    api.setup("cpu")
    assert hasattr(api, "llm")


def test_litllmapi_predict(api: LitLLMAPI):
    api.setup("cpu")
    out = api.predict("hi")
    assert hasattr(out, "__iter__") or out is not None
