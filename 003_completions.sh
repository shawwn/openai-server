prompt="${1:-Hello, my name is}"
while true
do
  prompt="$(OPENAI_API_BASE=http://test.tensorfork.com:9000 openai api completions.create -e 1558M -t 0.6 -M 32 -n 1 -p "$prompt" -F 0.85)"
  echo '----------'
  echo "$prompt"
done
