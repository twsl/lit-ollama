# lit-ollama

[![Build](https://github.com/twsl/lit-ollama/actions/workflows/build.yaml/badge.svg)](https://github.com/twsl/lit-ollama/actions/workflows/build.yaml)
[![Documentation](https://github.com/twsl/lit-ollama/actions/workflows/docs.yaml/badge.svg)](https://github.com/twsl/lit-ollama/actions/workflows/docs.yaml)
[![PyPI - Package Version](https://img.shields.io/pypi/v/lit-ollama?logo=pypi&style=flat&color=orange)](https://pypi.org/project/lit-ollama/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lit-ollama?logo=pypi&style=flat&color=blue)](https://pypi.org/project/lit-ollama/)
[![Docs with MkDocs](https://img.shields.io/badge/MkDocs-docs?style=flat&logo=materialformkdocs&logoColor=white&color=%23526CFE)](https://squidfunk.github.io/mkdocs-material/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/twsl/lit-ollama/releases)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-border.json)](https://github.com/copier-org/copier)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Replace ollama with LitServe

## Features

- **LitGPT model support**: Load and serve any LitGPT-compatible model using the standard Ollama interface
- **Ollama-compatible API**: Full compatibility with the Ollama API specification, allowing you to use any Ollama client without modifications
- **LitServe powered**: Built on LitServe for high-performance model serving with auto-batching and GPU acceleration

## Installation

With `pip`:

```bash
python -m pip install lit-ollama
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv add lit-ollama
```

## How to use it

Run like any other `litserve` server:
```python
import litserve as ls

from lit_ollama.server.api import LitOllamaAPI

api = LitOllamaAPI("mock")
server = ls.LitServer(
    api,
    accelerator="auto",
    devices="auto",
    callbacks=None,
    middlewares=None,
)
server.run()
```

Start the server with a specific model:

```bash
python server.py --model "meta-llama/Llama-3.2-1B-Instruct"
```

You can test the server by using the client to interact with it:

```bash
python client.py
```

## Docs

```bash
uv run mkdocs build -f ./mkdocs.yml -d ./_build/
```

## Update template

```bash
copier update --trust -A --vcs-ref=HEAD
```

## Credits

This project was generated with [![🚀 python project template.](https://img.shields.io/badge/python--project--template-%F0%9F%9A%80-brightgreen)](https://github.com/twsl/python-project-template)
