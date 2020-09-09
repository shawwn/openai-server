import os
import sys
from json import loads
from sanic import Sanic
from sanic.response import json, text
from pprint import pprint as pp

app = Sanic()

def log_request(request):
  #import pdb; pdb.set_trace()
  headers = dict(list(request.headers.items()))
  del headers['authorization']
  headers['x-openai-client-user-agent'] = loads(headers.get('x-openai-client-user-agent', '{}'))
  props = {}
  props['url'] = request.url
  props['method'] = request.method
  props['headers'] = headers
  props['request'] = request.json
  pp(props)
  #print(request.json)


@app.route('/v1/engines')
async def v1_engines_list(request):
  log_request(request)
  return json({
    "data": [
      {
        "id": "ada",
        "object": "engine",
        "owner": "openai",
        "ready": True
      },
      {
        "id": "ada-beta",
        "object": "engine",
        "owner": "openai",
        "ready": True
      },
      {
        "id": "babbage",
        "object": "engine",
        "owner": "openai",
        "ready": True
      },
      {
        "id": "babbage-beta",
        "object": "engine",
        "owner": "openai",
        "ready": True
      },
      {
        "id": "curie",
        "object": "engine",
        "owner": "openai",
        "ready": True
      },
      {
        "id": "curie-beta",
        "object": "engine",
        "owner": "openai",
        "ready": True
      },
      {
        "id": "davinci",
        "object": "engine",
        "owner": "openai",
        "ready": True
      },
      {
        "id": "davinci-beta",
        "object": "engine",
        "owner": "openai",
        "ready": True
      }
    ],
    "object": "list"
  })

@app.route('/v1/engines/davinci/completions', methods=['POST'])
async def v1_engines_davinci_completions(request):
  log_request(request)
  return json({"id": "cmpl-Wt5z1RZglyDHHl0SnSvKWVzA", "object": "text_completion", "created": 1599616871, "model": "davinci:2020-05-03", "choices": [{"text": "Test.SetLayerPropertiesWithNonContainedInvisible (", "index": 0, "logprobs": None, "finish_reason": "length"}]})
    
if __name__ == '__main__':
  args = sys.argv[1:]
  port = int(args[0] if len(args) > 0 else os.environ.get('PORT', '8000'))
  app.run(host='0.0.0.0', port=port)

