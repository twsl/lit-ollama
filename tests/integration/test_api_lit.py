import pytest

from lit_ollama.api.lit import LitLLMAPI


@pytest.fixture(scope="module")
def mode_name() -> str:
    return "meta-llama/Llama-3.2-1B-Instruct"


@pytest.fixture(scope="module")
def api(mode_name: str) -> LitLLMAPI:
    api = LitLLMAPI(mode_name)
    return api


@pytest.mark.skip("Ignore for now")
def test_litllmapi_init(api: LitLLMAPI, mode_name: str) -> None:
    assert api.model_name == mode_name
    api.setup("cpu")
    assert hasattr(api, "llm")


@pytest.mark.skip("Ignore for now")
def test_litllmapi_predict(api: LitLLMAPI) -> None:
    api.setup("cpu")
    out = api.predict("hi")
    assert hasattr(out, "__iter__") or out is not None
