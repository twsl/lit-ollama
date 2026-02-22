import argparse
import os

from dotenv import load_dotenv
import litserve as ls

from lit_ollama.api.lit import LitLLMAPI
from lit_ollama.api.spec import ollamaSpec

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="LitLLM Ollama-compatible server")
    parser.add_argument(
        "--model",
        type=str,
        default="mock",
        help="LitLLM model name to load (default: mock)",
    )
    args = parser.parse_args()

    api = LitLLMAPI(args.model)
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
