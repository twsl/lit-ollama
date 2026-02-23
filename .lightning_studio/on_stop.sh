#!/bin/bash
set -euo pipefail

echo "Stopping lit-ollama server..."

# Gracefully stop the server
if [ -f /tmp/lit-ollama.pid ]; then
    PID=$(cat /tmp/lit-ollama.pid)
    if kill -0 "$PID" 2>/dev/null; then
        kill -SIGTERM "$PID"
        echo "Sent SIGTERM to lit-ollama server (PID: $PID)"
        # Wait for graceful shutdown (max 10 seconds)
        for i in $(seq 1 10); do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo "lit-ollama server stopped gracefully"
                break
            fi
            sleep 1
        done
        # Force kill if still running
        if kill -0 "$PID" 2>/dev/null; then
            kill -SIGKILL "$PID"
            echo "Force killed lit-ollama server (PID: $PID)"
        fi
    else
        echo "lit-ollama server is not running"
    fi
    rm -f /tmp/lit-ollama.pid
else
    echo "No PID file found, lit-ollama server may not be running"
fi
