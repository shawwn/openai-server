import numpy as np
import os
import posixpath
import collections

import jax
import jax.numpy as jnp


if jax.default_backend() == 'tpu' and bool(int(os.environ.get('BF16', '1'))):
  default_dtype = jnp.bfloat16
  print('Using bfloat16')
else:
  default_dtype = jnp.float32


def load_tensor(reader, name: str) -> np.array:
  value = reader.get_tensor(name)
  value = np.squeeze(value)
  key = posixpath.normpath('///' + name)
  return key, value


def load_state(name: str):
  from tensorflow.python.training import py_checkpoint_reader
  reader = py_checkpoint_reader.NewCheckpointReader(f'models/{name}/model.ckpt')
  state_dict = dict([load_tensor(reader, k) for k in list(reader.get_variable_to_shape_map().keys())])
  return state_dict

def load_layers(name: str):
  state = load_state(name)
  n_blocks = 0
  for k in state.keys():
    if '/h' in k:
      idx, postfix = k.split('/h', 1)[1].split('/', 1)
      idx = int(idx)
      n_blocks = max(n_blocks, idx + 1)
  buckets = collections.defaultdict(lambda: [None] * n_blocks)
  for k in list(state.keys()):
    if '/h' in k:
      idx, postfix = k.split('/h', 1)[1].split('/', 1)
      idx = int(idx)
      buckets[postfix][idx] = state.pop(k)
  blocks = {k: jnp.stack(buckets.pop(k)) for k in list(buckets.keys())}
  state['/model/transformer'] = blocks
  return state


if __name__ == '__main__':
  from pprint import pprint as pp
  np.random.seed(0)
  model_name = os.environ.get('MODEL_NAME', '117M')
  state = load_layers(model_name)
  pp(jax.tree_util.tree_map(lambda x: x.shape, state))

