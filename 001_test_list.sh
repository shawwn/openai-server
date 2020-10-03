#!/bin/bash
export OPENAI_API_KEY="${OPENAI_API_KEY:-stub}"
set -ex
export PORT="${PORT:-9000}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:${PORT}}"
#export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com}"
#export OPENAI_LOG="${OPENAI_LOG:-debug}"

set +ex
#export OPENAI_LOG="${OPENAI_LOG:-debug}"
set -x
exec openai api engines.list "$@"
