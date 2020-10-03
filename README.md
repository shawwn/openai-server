# openai-server

A clone of OpenAI's GPT-3 API.

## quickstart

```sh
# setup
pip3 install -r requirements.txt

# download some gpt-2 models. (If you're out of drive space, just download 117M and skip the others.)
python3 download_model.py 117M
python3 download_model.py 345M
python3 download_model.py 774M
python3 download_model.py 1558M

# then, do ONE of the following. Either...

# ...serve one specific model,
MODELS=117M bash prod.sh
# ...or serve multiple models,
MODELS=117M,345M bash prod.sh
# ...or serve all models you've downloaded (the default)
bash prod.sh

# now you can open a different terminal and run:
curl 'http://localhost:9000/v1/engines/117M/completions?prompt=Hello,%20my%20name%20is&max_tokens=32&n=4&temperature=0.9&echo=true'

# for 1558M, the best results seem to come from temperature=0.4 and frequency_penalty=0.9:
curl 'http://localhost:9000/v1/engines/1558M/completions?prompt=Hello,%20my%20name%20is&max_tokens=32&n=4&temperature=0.4&frequency_penalty=0.9&echo=true'

# Warning: you shouldn't use frequency_penalty unless your model is
# the largest (1.5B). For some reason, frequency_penalty causes the
# output to be scrambled when used with any smaller model.
```
