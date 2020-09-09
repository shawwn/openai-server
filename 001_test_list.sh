#!/bin/bash
set -ex
export OPENAI_API_KEY="${OPENAI_API_KEY:-stub}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:8000}"
export OPENAI_LOG="${OPENAI_LOG:-debug}"
exec openai api engines.list "$@"
