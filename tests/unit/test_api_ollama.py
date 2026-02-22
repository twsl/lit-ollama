from lit_ollama.server.ollama import ollamaLitApi


def test_ollama_lit_api_init() -> None:
    api = ollamaLitApi()
    assert api is not None
