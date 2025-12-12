import os

from dotenv import load_dotenv
import litserve as ls

from lit_ollama.api.lit import LitLLMAPI
from lit_ollama.api.spec import ollamaSpec
from lit_ollama.models.mock import MockLLM

if __name__ == "__main__":
    load_dotenv()
    m = MockLLM()
    print(type(m))
    print(m.__class__.__mro__)
    asd = m.generate("hi")
    api = LitLLMAPI("meta-llama/Llama-3.2-1B-Instruct")
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices="auto",
        spec=ollamaSpec(),
        # spec=ls.OpenAISpec(),
        max_batch_size=None,
        stream=True,
        callbacks=None,
        middlewares=None,
    )
    server.run(
        host=os.environ.get("HOST", "0.0.0.0"),  # noqa: S104
        port=os.environ.get("PORT", 11434),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
