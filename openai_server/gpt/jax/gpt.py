import numpy as np
import os
import posixpath
import collections
import json
import random

import jax
import jax.numpy as jnp
from jax.experimental import stax


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


def load_layers(name: str, dtype=None):
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
  blocks = {k: jnp.array(jnp.stack(buckets.pop(k)), dtype=dtype) for k in list(buckets.keys())}
  state = {k: jnp.array(v, dtype=dtype) for k, v in state.items()}
  state['/model/transformer'] = blocks
  return state


@jax.tree_util.register_pytree_node_class
class VariableContext(object):
    def __init__(self, name2val, *, prefix, allow_new=True, **static_kwargs):
        self.static_kwargs = static_kwargs
        for k, v in static_kwargs.items():
          setattr(self, k, v)
        self.name2val = name2val
        self.prefix = prefix
        self.allow_new = allow_new

    def tree_flatten(self):
        return ((self.name2val,), {**self.static_kwargs, 'prefix': self.prefix, 'allow_new': self.allow_new})

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
          return False
        return (self.name2val, self.prefix, self.static_kwargs) == (other.name2val, other.prefix, other.static_kwargs)

    def __repr__(self):
        kws = {'prefix': self.prefix, 'allow_new': self.allow_new, **self.static_kwargs}
        kws = {k: v for k, v in kws.items() if v is not False}
        kws = ', '.join(['='.join([k, repr(v)]) for k, v in kws.items()])
        return "VariableContext({})".format(kws)

    def scope(self, name='', **static_kwargs):
        return VariableContext(self.name2val, 
            prefix=self._join(self.prefix, name),
            allow_new=self.allow_new,
            **{**self.static_kwargs, **static_kwargs})

    def __getitem__(self, name):
        return self.get_variable(name)

    def get_variable(self, name, initializer=None):
        return self.get_variable_absolute(
            name=self._join(self.prefix, name), 
            initializer=initializer)

    def get_variable_absolute(self, name, initializer=None):
        if name not in self.name2val:
            if initializer is None:
                raise KeyError(name)
            assert self.allow_new
            val = initializer()
            # val = jnp.array(val)
            self.name2val[name] = val
        return self.name2val[name]

    def _join(self, *xs):
        return posixpath.normpath(posixpath.join(*xs))


def shape_map(x):
  return jax.tree_util.tree_map(lambda x: x.shape, x)


def normax(shape, axis):
    out = np.random.randn(*shape).astype(np.float32)
    out /= np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
    return out

def normc(*shape):
    return normax(shape, axis=0)

def randn(shape, stddev):
    return np.random.randn(*shape).astype(np.float32) * stddev

# def gelu(x):
#     return 0.5*x*(1+np.tanh(0.79788*(x+0.044715*x**3)))

from jax.nn import gelu

@jax.named_call
def _norm(x, g, b, *, eps=1e-5, axis=-1):
    u = jnp.mean(x, axis=axis, keepdims=True)
    s = jnp.mean(jnp.square(x-u), axis=axis, keepdims=True)
    x = (x - u) / jnp.sqrt(s + eps)
    assert g is not None and b is not None
    x = x * g + b
    return x

@jax.named_call
def norm(cx, x, *, eps=1e-5, axis=-1):
    n_state = x.shape[axis]
    g = cx.get_variable("g", initializer=lambda : np.ones(n_state, 'f'))
    b = cx.get_variable("b", initializer=lambda : np.zeros(n_state, 'f'))
    return _norm(x, g, b, eps=eps, axis=axis)

@jax.named_call
def attention_mask(nd, ns, *, dtype):
    i = jnp.arange(nd)[:,None]
    j = jnp.arange(ns)
    m = i >= j - ns + nd
    return m.astype(dtype)

@jax.named_call
def mask_attn_weights(w):
    *nb, nd, ns = w.shape
    if nd <= 1:
      return w
    b = attention_mask(nd, ns, dtype=w.dtype)
    b = jnp.reshape(b, tuple([1 for _ in range(len(nb))]) + (nd, ns))
    w = w * b - jnp.array(1e9, dtype=w.dtype) * (1 - b)
    return w

@jax.named_call
def _dense(X_tk, W_kf, b_f, F):
    *B, T, K = X_tk.shape
    X_t_k = jnp.reshape(X_tk, (-1, K))
    Y_t_f = jnp.matmul(X_t_k, W_kf) + b_f
    return jnp.reshape(Y_t_f, (*B, T, F))

@jax.named_call
def dense(cx, X_tk, F):
    *B, T, K = X_tk.shape
    W_kf = cx.get_variable("w", initializer=lambda: normc(K, F))
    b_f = cx.get_variable("b", initializer=lambda: np.zeros(F,'f'))
    return _dense(X_tk, W_kf, b_f, F)

@jax.named_call
def attn(cx, X_tk, past):
    n_head = cx.n_head
    *B, T, n_state = X_tk.shape
    assert n_state % n_head==0
    QKV_t_3s = dense(cx.scope('c_attn'), X_tk, n_state * 3)
    QKV_t_3h_r = jnp.reshape(QKV_t_3s, (*B, T, 3 * n_head, n_state // n_head))
    Q_thr, K_thr, V_thr = jnp.split(QKV_t_3h_r, 3, axis=-2)
    if past is not None:
        pk, pv = past
        K_thr = jnp.concatenate([pk, K_thr], axis=-3)
        V_thr = jnp.concatenate([pv, V_thr], axis=-3)
    present = [K_thr, V_thr]
    R = Q_thr.shape[-1]
    W_htt = jnp.einsum("thr,Thr->htT", Q_thr, K_thr) / jnp.sqrt(R).astype(default_dtype)
    W_htt = mask_attn_weights(W_htt)
    W_htt = stax.softmax(W_htt, axis=-1)
    A_thr = jnp.einsum("htT,Thr->thr", W_htt, V_thr)
    A_ts = jnp.reshape(A_thr, (*B, T, n_state))
    P_ts = dense(cx.scope('c_proj'), A_ts, n_state)
    return P_ts, present

@jax.named_call
def mlp(cx, X_ts):
    S = X_ts.shape[-1]
    n_hid = S * 4
    H_th = gelu(dense(cx.scope('c_fc'), X_ts, n_hid))
    Y_ts = dense(cx.scope('c_proj'), H_th, S)
    return Y_ts

@jax.named_call
def block(cx, x, past):
    a, present = attn(cx.scope('attn'), norm(cx.scope('ln_1'), x), past)
    x = x + a
    m = mlp(cx.scope('mlp'), norm(cx.scope('ln_2'), x))
    x = x + m
    return x, present

@jax.named_call
def initial_embed(cx, tok_t, past_len=0):
    pos_t = jax.lax.broadcasted_iota(jnp.int32, tok_t.shape, len(tok_t.shape)-1)
    pos_t = pos_t + past_len
    tokembs_qe = cx.get_variable('wte', initializer=lambda: normc(cx.n_vocab, cx.n_embd) * 0.1)
    posembs_pe = cx.get_variable('wpe', initializer=lambda: normc(cx.n_ctx, cx.n_embd) * 0.1)
    tokemb_te = tokembs_qe[tuple([tok_t])]
    posemb_te = posembs_pe[tuple([pos_t])]
    last_ts = tokemb_te + posemb_te
    return last_ts

@jax.named_call
def final_embed(cx, last_ts):
    tokembs_qe = cx.get_variable('wte')
    last_ts = norm(cx.scope('ln_f'), last_ts)
    logits_tq = jnp.matmul(last_ts, tokembs_qe.T)
    return logits_tq

@jax.named_call
def transformer_embed(cx, last_ts, past=None):
  def apply_scan_fn(x, layer_state):
    x, = x
    past = layer_state.pop('past')
    cx2 = VariableContext(layer_state, prefix='', **cx.static_kwargs)
    x, present = block(cx2, x, past)
    return [x], present
  xs = {**cx['transformer']}
  xs['past'] = past
  [last_ts,], presents = jax.lax.scan(apply_scan_fn, [last_ts], xs=xs)
  return last_ts, presents

@jax.named_call
@jax.jit
def transformer(cx, tok_t, past=None, past_len=None):
    if past_len is None:
      past_len = past_length(past)
    last_ts = initial_embed(cx, tok_t, past_len)
    last_ts, presents = transformer_embed(cx, last_ts, past=past)
    logits_tq = final_embed(cx, last_ts)
    return logits_tq, presents

def past_length(past):
  if past is None:
    return 0
  elif isinstance(past, (list, tuple)):
    K_bthr, V_bthr = past
    return V_bthr.shape[-3]
  else:
    KV_bthr = past
    return KV_bthr.shape[-3]


# takes in a logit distribution, softmax and then sample
@jax.named_call
def softmax_sample(key, logits, temp=0.75):
    return jax.random.categorical(key, logits / temp, -1).astype(jnp.uint32)


@jax.named_call
def generate_token(logprobs_btq, sample_key=None, *, sampler, **sampler_options):
  if sample_key is None:
    sample_key = jax.random.PRNGKey(random.randint(0, 2 ** 60))
  sample_key, new_key = jax.random.split(sample_key)
  logits = logprobs_btq[..., -1:, :]
  next_token = sampler(sample_key, logits, **sampler_options)
  return next_token, new_key



if __name__ == '__main__':
  from pprint import pprint as pp
  np.random.seed(0)
  model_name = os.environ.get('MODEL_NAME', '117M')
  max_tokens = int(os.environ.get('MAX_TOKENS', '16'))
  prompt = os.environ.get('PROMPT', 'Hello, my name is')
  from src import encoder
  tokenizer = encoder.get_encoder(model_name)
  def encode(prompt):
    return jnp.array(tokenizer.encode(prompt), dtype='i')
  state = load_layers(model_name, dtype=default_dtype)
  hparams = json.loads(open(f'models/{model_name}/hparams.json').read())
  pp(jax.tree_util.tree_map(lambda x: (x.shape, x.dtype), state))
  cx = VariableContext(state, prefix='/model/', allow_new=False, **hparams)
  #tokens = jnp.zeros((1, cx.n_ctx), dtype='i')
  #tokens = jnp.ones((1, 1), dtype='i') * 50256
  tokens = encode(prompt)
  network = transformer
  # logits_tq = initial_embed(cx, tokens)
  # logits_tq, presents = transformer_embed(cx, logits_tq)
  # logits_tq = final_embed(cx, logits_tq)
  sample_key = None
  presents = None
  sampler = softmax_sample
  from jax.util import partial
  gen_token = jax.jit(partial(generate_token, sampler=softmax_sample))
  print('')
  print(prompt, end='', flush=True)
  logits_btq, presents = network(cx, encode(prompt))
  for i in range(max_tokens):
    token, sample_key = gen_token(logits_btq, sample_key)
    print(tokenizer.decode(token), end='', flush=True)
    logits_btq, presents = network(cx, token, presents)

