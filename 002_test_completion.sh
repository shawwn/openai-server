#!/bin/bash
set -ex
engine="${1:-davinci}"
prompt="${2:-Test}"
temp="${3:-0.6}"
max_tokens="${4:-12}"
n="${5:-1}"
set +e
shift 1
shift 1
shift 1
shift 1
shift 1
export OPENAI_API_KEY="${OPENAI_API_KEY:-stub}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:8000}"
export OPENAI_LOG="${OPENAI_LOG:-debug}"
exec openai api completions.create -e "${engine}" -p "${prompt}" -t "${temp}" -M "${max_tokens}" -n "${n}" "$@"
