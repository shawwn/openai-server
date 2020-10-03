#!/bin/bash
export OPENAI_API_KEY="${OPENAI_API_KEY:-stub}"
set -ex
export PORT="${PORT:-9000}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:${PORT}}"
#export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com}"
#export OPENAI_LOG="${OPENAI_LOG:-debug}"

prompt="${1:-Hello, my name is}"

if [ -e "$prompt" ]
then
  prompt="$(cat "$prompt")"
fi

ENGINE="${ENGINE:-${E:-davinci}}"
TEMPERATURE="${TEMPERATURE:-${T:-0.9}}"
MAX_TOKENS="${MAX_TOKENS:-${M:-12}}"
N="${N:-1}"
set +ex
shift 1

set -x
exec openai api completions.create -e "${ENGINE}" -t "${TEMPERATURE}" -M "${MAX_TOKENS}" -n "${N}" -p "${prompt}" "$@"
