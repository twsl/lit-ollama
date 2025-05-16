import os

from dotenv import load_dotenv
import litserve as ls

from lit_ollama.api.lit import LitLLMAPI
from lit_ollama.api.spec import ollamaSpec

if __name__ == "__main__":
    load_dotenv()
    api = LitLLMAPI("meta-llama/Llama-3.2-1B-Instruct")
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices="auto",
        spec=ollamaSpec(),
        # spec=ls.OpenAISpec(),
        callbacks=None,
        middlewares=None,
        stream=True,
    )
    server.run(
        host=os.environ.get("HOST", "0.0.0.0"),  # noqa: S104
        port=os.environ.get("PORT", 11434),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
