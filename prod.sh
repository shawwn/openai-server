set -ex
export MAX_TOKENS="${MAX_TOKENS:-500}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PORT="${PORT:-9000}"
exec bash start.sh
