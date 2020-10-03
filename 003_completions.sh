export OPENAI_API_KEY="${OPENAI_API_KEY:-stub}"
set -x
export PORT="${PORT:-9000}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:${PORT}}"
#export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com}"
#export OPENAI_LOG="${OPENAI_LOG:-debug}"
set +x

prompt="${1:-Hello, my name is}"
while true
do
  prompt="$(openai api completions.create -e davinci -t 0.6 -M 32 -n 1 -p "$prompt")"
  printf "\033c"
  #echo '----------'
  echo "$prompt"
done
