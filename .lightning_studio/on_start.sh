#!/bin/bash
set -euo pipefail

echo "Starting lit-ollama setup..."

# Install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies
uv sync --all-extras

# Start the server in the background
# Lightning Studio exposes ports publicly via the Ports plugin
# server.py binds to 0.0.0.0:11434 (configurable via HOST/PORT env vars)
echo "Starting lit-ollama server..."
nohup uv run python server.py > /var/log/lit-ollama.log 2>&1 &
echo $! > /tmp/lit-ollama.pid

echo "lit-ollama server started on port ${PORT:-11434} (PID: $(cat /tmp/lit-ollama.pid))"
