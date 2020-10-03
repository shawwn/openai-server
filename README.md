# openai-server

`openai-server` is an implementation of the [OpenAI API](https://openai.com/blog/openai-api/).

Specifically, we implement `/v1/engines/list` and `/v1/engines/{model_name}/completions` endpoints.

Both endpoints are mostly feature-complete, with a few differences. The JSON response is identical; any library that works with the OpenAI API will probably work with this.

To get started, see the [quickstart](#Quickstart) or the [examples](#Examples) or the [JavaScript API](https://github.com/shawwn/tensorfork-openai-api).

## Contact

- Twitter: [@theshawwn](https://twitter.com/theshawwn)
- HN: [sillysaurusx](https://news.ycombinator.com/item?id=23346972)
- ML discord: [https://discordapp.com/invite/x52Xz3y](https://discordapp.com/invite/x52Xz3y)
- Support me on patreon: [patreon.com/shawwn](https://patreon.com/shawwn)

## Quickstart

```sh
# setup.
pip3 install -r requirements.txt
# grab a gpt-2 model.
python3 download_model.py 117M # or 345M, 774M, 1558M
# start the server.
MODELS=117M bash prod.sh
```
Your server is now serving the OpenAI API at localhost:9000. (You can change the port via `export PORT=8000`)

Then, in a separate terminal, grab some completions using the official `openai` command-line tool:
```sh
$ OPENAI_API_BASE=http://localhost:9000 openai api completions.create -e davinci -p 'Hello, world' -t 0.8 -M 16 -n 4
===== Completion 0 =====
Hello, world. It seems like a good idea to make a living. The fact that it
===== Completion 1 =====
Hello, world. This is not the first time you're seeing the same thing at any given
===== Completion 2 =====
Hello, world, please do my best to continue the development of Monad and its conforming
===== Completion 3 =====
Hello, world controlled enemy.

"Be careful. We have come across a near total
```

continuously dump completions to the terminal:
```sh
$ bash 003_completions.sh 'Yo dawg, we implemented OpenAI API'
Yo dawg, we implemented OpenAI API. Now, we have the ability to connect to Signal, a cryptographic data store.

We can now make this secure by using new kid on the block chain, OpenAI.

OpenAI is the new block chain protocol for the internet. This is a major milestone. As the internet becomes more open and open for everybody, it is important for us to have a robust, high-quality blockchain. It is also important that we never create an untraceable chain. The blockchain is the only way to guarantee that everyone has the same access to the network.

We are an open consortium and we believe that the blockchain is the bridge between the internet and the rest of the world. We're committed to this project. We believe that the blockchain is a bridge between the internet and
^C
```

fetch the JSON endpoint manually:
```sh
$ curl 'http://localhost:9000/v1/engines/117M/completions?prompt=Hello,%20my%20name%20is&max_tokens=32&n=4&temperature=0.9&echo=true'
{
  "choices": [
    {
      "finish-reason": "length",
      "index": 0,
      "logprobs": null,
      "text": "Hello, my name is Loium Chazz, and I have been far from satisfied with your departure. But I will, at least by some chance, give you permission to decide for"
    },
    {
      "finish-reason": "length",
      "index": 1,
      "logprobs": null,
      "text": "Hello, my name is Tim and my name is Jodie. Yours, Tom.\n\nTim: Oh hello, my name is Tim.\n\nJB: Where?'"
    },
    {
      "finish-reason": "length",
      "index": 2,
      "logprobs": null,
      "text": "Hello, my name is Rosen Sylvan. That's right, Buck Paoli, who was a member of the Board of Governors for George W. Bush in the 2009 Democratic primary\u2014"
    },
    {
      "finish-reason": "length",
      "index": 3,
      "logprobs": null,
      "text": "Hello, my name is Nick Martens, I am an English-speaking Canadian, University of Toronto, Mississauga, Canada. I work in a computer software company located in Canada."
    }
  ],
  "created": 1601701785.777768,
  "id": "cmpl-3qN8kwW1Ya7_qxWz4h8wuIzN",
  "model": "117M",
  "object": "text_completion"
}
```

Or just [open the JSON endpoint in your browser](http://localhost:9000/v1/engines/117M/completions?prompt=Hello,%20my%20name%20is&max_tokens=32&n=4&temperature=0.9&echo=true) and start playing around with the query params.

## Examples

### A simple bash script for dumping completions

```sh
$ T=0.8 M=32 bash 002_test_completion.sh 'Hello, my name is'
Hello, my name is Plato and, like many of you, I am very happy with the pre-release.

The primary goal of the pre-release was to provide
```

The first argument to `002_test_completion.sh` is the prompt:
```sh
bash 002_test_completion.sh 'Hello there. My name is'
```

You can set the temperature using `T=0.8` and the token count using `M=32`:
```sh
T=0.8 M=32 bash 002_test_completion.sh 'Hello there. My name is'
```

To read a prompt from a file, simply pass in the filename. If the first argument is a valid filename, the file becomes the prompt:
```sh
T=0.8 M=32 bash 002_test_completion.sh README.md
```

If the prompt is too long, the last `1023 - M` tokens of the prompt are used. **Note**: This means if you request 500 tokens, it will only use `1023 minus 500` tokens from the prompt. Therefore, to let GPT see as many tokens as possible, request a small number of tokens (e.g. 16).

### Setting up everything from scratch

```sh
# grab the code.
git clone https://github.com/shawwn/openai-server
cd openai-server

# install dependencies.
pip3 install -r requirements.txt

# grab all models (requires ~8GB of disk space; if low, just download 117M, which only requires 550MB)
python3 download_model.py 117M
python3 download_model.py 345M
python3 download_model.py 774M
python3 download_model.py 1558M

# then, do *one* of the following:

# ...serve one specific model:
MODELS=117M bash prod.sh

# ...or serve multiple models:
MODELS=1558M,117M bash prod.sh

# ...or serve all models you've downloaded (the default):
bash prod.sh
```

The server listens on port 9000 by default. You can change it via PORT:
```sh
PORT=9000 bash prod.sh
```

Now that the server is running, you can start making API requests via `002_test_completion.sh`:
```sh
bash 002_test_completion.sh 'Hello there. My name is'
```

Or fetch completions from the JSON endpoint:
```sh
curl -s 'http://localhost:9000/v1/engines/117M/completions?echo=true&prompt=Hello,%20world' | jq .choices[].text
```

Or [open the endpoint in your browser](http://localhost:9000/v1/engines/117M/completions?prompt=Hello,%20my%20name%20is&max_tokens=32&n=4&temperature=0.9&echo=true) and mess with query params.

## Notes

### A warning about frequency_penalty

for 1558M, the best results seem to come from `temperature=0.6` and `frequency_penalty=0.9`:
```sh
curl 'http://localhost:9000/v1/engines/1558M/completions?prompt=Hello,%20my%20name%20is&max_tokens=32&n=4&temperature=0.4&frequency_penalty=0.9&echo=true'
```

But beware: you shouldn't use `frequency_penalty` unless your model is the largest (1558M, commonly known as "1.5B"). For some reason, `frequency_penalty` causes the output to be scrambled when the model is smaller than 1558M.

### Running in production

For production usage, consider running it via the following command:

```sh
while true; do MODELS=117M bash prod.sh ; sleep 20 ; done
```

That way, if the server terminates for any reason, it will automatically restart.

For endpoint monitoring, I recommend [updown.io](https://updown.io/).

## Community

### Join the ML Discord

If you're an ML enthusiast, join the [ML Discord](https://discordapp.com/invite/x52Xz3y).
There are ~800 members, with ~120 online at any given time:

![image](https://user-images.githubusercontent.com/59632/84269906-bc7d2080-aade-11ea-8b4e-f78412855d43.png)

There are a variety of interesting channels:

- `#papers` for pointing out interesting research papers
- `#research` for discussing ML research
- `#show` and `#samples` for showing off your work
- `#hardware` for hardware enthusiasts
- `#ideas` for brainstorming
- `#tensorflow` and `#pytorch`
- `#cats`, `#doggos`, and of course `#memes`
- Quite a few more.

## Support me

*If you found this library helpful, consider [joining my patreon](https://patreon.com/shawwn).*

