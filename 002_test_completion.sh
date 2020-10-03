#!/bin/bash
export OPENAI_API_KEY="${OPENAI_API_KEY:-stub}"
set -ex
export PORT="${PORT:-9000}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:${PORT}}"
#export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com}"
#export OPENAI_LOG="${OPENAI_LOG:-debug}"

engine="${1:-davinci}"
prompt="${2:-Test}"
temp="${3:-0.6}"
max_tokens="${4:-12}"
n="${5:-4}"
set +ex
shift 1
shift 1
shift 1
shift 1
shift 1

set -x
exec openai api completions.create -e "${engine}" -p "${prompt}" -t "${temp}" -M "${max_tokens}" -n "${n}" "$@"
