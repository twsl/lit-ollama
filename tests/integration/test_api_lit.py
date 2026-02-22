import pytest

from lit_ollama.server.api import LitOllamaAPI


@pytest.fixture(scope="module")
def mode_name() -> str:
    return "meta-llama/Llama-3.2-1B-Instruct"


@pytest.fixture(scope="module")
def api(mode_name: str) -> LitOllamaAPI:
    api = LitOllamaAPI(mode_name)
    return api


@pytest.mark.skip("Ignore for now")
def test_litllmapi_init(api: LitOllamaAPI, mode_name: str) -> None:
    assert api.model_name == mode_name
    api.setup("cpu")
    assert hasattr(api, "llm")


@pytest.mark.skip("Ignore for now")
def test_litllmapi_predict(api: LitOllamaAPI) -> None:
    api.setup("cpu")
    out = api.predict("hi")
    assert hasattr(out, "__iter__") or out is not None
