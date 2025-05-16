from lit_ollama.api.ollama import ollamaLitApi


def test_ollama_lit_api_init():
    api = ollamaLitApi()
    assert api is not None
