import argparse
import os

from dotenv import load_dotenv
import litserve as ls

from lit_ollama.server.api import LitOllamaAPI

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

    api = LitOllamaAPI(args.model)
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices="auto",
        callbacks=None,
        middlewares=None,
    )
    server.run(
        host=os.environ.get("HOST", "0.0.0.0"),  # noqa: S104
        port=os.environ.get("PORT", 11434),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
