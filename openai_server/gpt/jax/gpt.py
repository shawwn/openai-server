import numpy as np
import os
import posixpath
import collections
import json

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
    *_, nd, ns = w.shape
    if nd <= 1:
      return w
    b = attention_mask(nd, ns, dtype=w.dtype)
    b = jnp.reshape(b, (1, 1, nd, ns))
    w = w * b - jnp.array(1e9, dtype=w.dtype) * (1 - b)
    return w

@jax.named_call
def _dense(X_btk, W_kf, b_f, F):
    B, T, K = X_btk.shape
    X_bt_k = jnp.reshape(X_btk, (-1, K))
    Y_bt_f = jnp.matmul(X_bt_k, W_kf) + b_f
    return jnp.reshape(Y_bt_f, (B, T, F))

@jax.named_call
def dense(cx, X_btk, F):
    B, T, K = X_btk.shape
    W_kf = cx.get_variable("w", initializer=lambda: normc(K, F))
    b_f = cx.get_variable("b", initializer=lambda: np.zeros(F,'f'))
    return _dense(X_btk, W_kf, b_f, F)

@jax.named_call
def attn(cx, X_btk, past):
    n_head = cx.n_head
    B, T, n_state = X_btk.shape
    assert n_state % n_head==0
    QKV_b_t_3s = dense(cx.scope('c_attn'), X_btk, n_state * 3)
    QKV_b_t_3h_r = jnp.reshape(QKV_b_t_3s, (B, T, 3 * n_head, n_state // n_head))
    Q_bthr, K_bthr, V_bthr = jnp.split(QKV_b_t_3h_r, 3, axis=-2)
    if past is not None:
        pk, pv = past
        K_bthr = jnp.concatenate([pk, K_bthr], axis=-3)
        V_bthr = jnp.concatenate([pv, V_bthr], axis=-3)
    present = [K_bthr, V_bthr]
    R = Q_bthr.shape[-1]
    W_bhtt = jnp.einsum("bthr,bThr->bhtT", Q_bthr, K_bthr) / jnp.sqrt(R).astype(default_dtype)
    W_bhtt = mask_attn_weights(W_bhtt)
    W_bhtt = stax.softmax(W_bhtt, axis=-1)
    A_bthr = jnp.einsum("bhtT,bThr->bthr", W_bhtt, V_bthr)
    A_bts = jnp.reshape(A_bthr, (B, T, n_state))
    P_bts = dense(cx.scope('c_proj'), A_bts, n_state)
    return P_bts, present

@jax.named_call
def mlp(cx, X_bts):
    S = X_bts.shape[-1]
    n_hid = S * 4
    H_bth = gelu(dense(cx.scope('c_fc'), X_bts, n_hid))
    Y_bts = dense(cx.scope('c_proj'), H_bth, S)
    return Y_bts

@jax.named_call
def block(cx, x, past):
    a, present = attn(cx.scope('attn'), norm(cx.scope('ln_1'), x), past)
    x = x + a
    m = mlp(cx.scope('mlp'), norm(cx.scope('ln_2'), x))
    x = x + m
    return x, present

@jax.named_call
def initial_embed(cx, tok_bt, past_len=0):
    B, T = tok_bt.shape
    pos_bt = jax.lax.broadcasted_iota(jnp.int32, (B, T), 1)
    pos_bt = pos_bt + past_len
    tokembs_qe = cx.get_variable('wte', initializer=lambda: normc(cx.n_vocab, cx.n_embd) * 0.1)
    posembs_pe = cx.get_variable('wpe', initializer=lambda: normc(cx.n_ctx, cx.n_embd) * 0.1)
    tokemb_bte = tokembs_qe[tuple([tok_bt])]
    posemb_bte = posembs_pe[tuple([pos_bt])]
    last_bts = tokemb_bte + posemb_bte
    if len(last_bts.shape) < 3:
        last_bts = last_bts[jnp.newaxis, ...]
    return last_bts

@jax.named_call
def final_embed(cx, last_bts):
    tokembs_qe = cx.get_variable('wte')
    last_bts = norm(cx.scope('ln_f'), last_bts)
    logits_btq = jnp.matmul(last_bts, tokembs_qe.T)
    return logits_btq

@jax.named_call
def transformer_embed(cx, last_bts, past=None):
  def apply_scan_fn(x, layer_state):
    x, past = x
    cx2 = VariableContext(layer_state, prefix='', **cx.static_kwargs)
    x, present = block(cx2, x, past)
    return [x, past], present
  [last_bts, past], presents = jax.lax.scan(apply_scan_fn, [last_bts, past], xs=cx['transformer'])
  return last_bts, presents

@jax.named_call
def transformer(cx, tok_bt, past=None, past_len=None):
    if past_len is None:
      past_len = past_length(past)
    last_bts = initial_embed(cx, tok_bt, past_len)
    last_bts, presents = transformer_embed(cx, last_bts, past=past)
    logits_btq = final_embed(cx, last_bts)
    return logits_btq, presents

def past_length(past):
  if past is None:
    return 0
  elif isinstance(past, (list, tuple)):
    K_bthr, V_bthr = past[0]
    return V_bthr.shape[-3]
  else:
    KV_bthr = past
    return KV_bthr.shape[-3]


if __name__ == '__main__':
  from pprint import pprint as pp
  np.random.seed(0)
  model_name = os.environ.get('MODEL_NAME', '117M')
  state = load_layers(model_name)
  hparams = json.loads(open(f'models/{model_name}/hparams.json').read())
  pp(shape_map(state))
  cx = VariableContext(state, prefix='/model/', allow_new=False, **hparams)
  toks = jnp.zeros((1, cx.n_ctx), dtype='i')
  # last_bts = initial_embed(cx, toks)
  # curr_bts, presents = transformer_embed(cx, last_bts)
  # logits_btq = final_embed(cx, curr_bts)
  network = jax.jit(transformer)
  logits_btq, presents = network(cx, toks)

