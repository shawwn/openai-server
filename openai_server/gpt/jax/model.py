import jax
import jax.numpy as jnp
from jax.experimental import maps, stax
import numpy as np
import os
from pprint import pprint as pp

from src import model as tf_model
import json
import functools
import time

def load_tensor(reader, name: str) -> np.array:
  #name = '.'.join(name.split('/')[1:])
  value = reader.get_tensor(name)
  #key = '.'.join(name.split('/'))
  key = name
  # if value.shape and value.shape[0] == 1:
  #   value = np.squeeze(value, axis=0)
  return key, value

def load_state(name: str):
  from tensorflow.python.training import py_checkpoint_reader
  reader = py_checkpoint_reader.NewCheckpointReader( 'models/{name}/model.ckpt'.format(name=name) )
  state_dict = dict([load_tensor(reader, k) for k in list(reader.get_variable_to_shape_map().keys())])
  return state_dict

import tensorflow as tf2
import tensorflow.compat.v1 as tf

_variables = {}

def get_variable(name, *args, **kws):
  global network
  scope = tf.get_variable_scope().name or ''
  if scope:
    name = scope + '/' + name
  print(f'get_variable {name}')
  if name in _variables:
    return _variables[name]
  _variables[name] = tf.Variable(network.params[name])
  return _variables[name]

def load(config):
  name = config['model_name']
  state_dict = load_state(name)
  pp({k: v.shape for k, v in state_dict.items()})
  return TransformerV3(config, state_dict)

# ================================================================
# Tf-like framework for Jax
# ================================================================

def create_root_context(prefix='model'):
    return VariableContext({}, prefix=prefix)

class VariableContext(object):
    def __init__(self, name2val, prefix, allow_new=True):
        self.name2val = name2val
        self.prefix = prefix
        self.allow_new = allow_new
    def scope(self, name):
        return VariableContext(self.name2val, 
            self._join(self.prefix, name), self.allow_new)
    def get_variable(self, name, initializer):
        return self.get_variable_absolute(
            name=self._join(self.prefix, name), 
            initializer=initializer)
    def get_variable_absolute(self, name, initializer):
        if name not in self.name2val:
            assert self.allow_new
            val = initializer()
            assert type(val) == np.ndarray and val.dtype == np.float32
            self.name2val[name] = val

        return self.name2val[name]
    def _join(self, *xs):
        return '/'.join(xs)
    def variables_list(self):
        return list(self.name2val.values())
    def replace_with_list(self, newlist):
        assert len(newlist) == len(self.name2val)
        name2val = {k : v for (k, v) in zip(self.name2val.keys(), newlist)}
        return VariableContext(name2val, self.prefix, self.allow_new)

def print_variables(cx):
    for (name, val) in sorted(cx.name2val.items()):
        print(f'{name:20s} {str(val.shape):20s} {str(val.dtype):20s}')

# End framework 
# ----------------------------------------



def normax(shape, axis):
    out = np.random.randn(*shape).astype(np.float32)
    out /= np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
    return out

def normc(*shape):
    return normax(shape, axis=0)

def randn(shape, stddev):
    return np.random.randn(*shape).astype(np.float32) * stddev

def gelu(x):
    return 0.5*x*(1+np.tanh(0.79788*(x+0.044715*x**3)))

def _norm(x, *, axis, g=None, b=None, e=1e-5):
    u = np.mean(x, axis=axis, keepdims=True)
    s = np.mean(np.square(x-u), axis=axis, keepdims=True)
    x = (x - u) / np.sqrt(s + e)
    assert g is not None and b is not None
    if g is not None and b is not None:
        x = x * g + b
    return x

def norm(cx, x, axis=-1):
    n_state = x.shape[axis]
    g = cx.get_variable("g", initializer=lambda : np.ones(n_state, 'f'))
    b = cx.get_variable("b", initializer=lambda : np.zeros(n_state, 'f'))
    return _norm(x, g=g, b=b, axis=axis)

def mask_attn_weights(w):
    n = w.shape[-1]
    b = np.tril(np.ones((n,n)))
    b = np.reshape(b, (1, 1, n, n))
    w = w * b - 1e9 * (1 - b)
    return w

def _attn(Q_bhtr, K_bhrt, V_bhtr):
    R = Q_bhtr.shape[-1]
    W_bhtt = np.matmul(Q_bhtr, K_bhrt) / np.sqrt(R)
    W_bhtt = mask_attn_weights(W_bhtt)
    W_bhtt = stax.softmax(W_bhtt, axis=-1)
    A_bhtr = np.matmul(W_bhtt, V_bhtr)
    return A_bhtr

def dense(cx, X_btk, F):
    B, T, K = X_btk.shape
    X_bt_k = np.reshape(X_btk, (-1, K))
    W_kf = cx.get_variable("w", initializer=lambda : normc(K, F))
    b_f = cx.get_variable("b", initializer=lambda : np.zeros(F,'f'))
    Y_bt_f = np.matmul(X_bt_k, W_kf) + b_f
    return np.reshape(Y_bt_f, (B, T, F))

def attn(cx, X_btk, n_state, n_head):
    B, T, _K = X_btk.shape
    assert n_state % n_head==0
    QKV_b_t_3s = dense(cx.scope('c_attn'), X_btk, n_state * 3)
    QKV_b_t_3h_r = np.reshape(QKV_b_t_3s, (B, T, 3 * n_head, n_state // n_head))
    Q_bthr, K_bthr, V_bthr = np.split(QKV_b_t_3h_r, 3, axis=2)
    Q_bhtr = np.transpose(Q_bthr, (0, 2, 1, 3))
    V_bhtr = np.transpose(V_bthr, (0, 2, 1, 3))
    K_bhrt = np.transpose(K_bthr, (0, 2, 3, 1))
    A_bhtr = _attn(Q_bhtr, K_bhrt, V_bhtr)
    A_bthr = np.transpose(A_bhtr, (0, 2, 1, 3))
    A_bts = np.reshape(A_bthr, (B, T, n_state))
    P_bts = dense(cx.scope('c_proj'), A_bts, n_state)
    return P_bts

def mlp(cx, X_bts, *, n_hid):
    S = X_bts.shape[-1]
    H_bth = stax.gelu(dense(cx.scope('c_fc'), X_bts, n_hid))
    Y_bts = dense(cx.scope('c_proj'), H_bth, S)
    return Y_bts

def block(cx, x, *, n_head):
    S = x.shape[-1]
    a = attn(cx.scope('attn'), norm(cx.scope('ln_1'), x), S, n_head)
    x = x + a
    m = mlp(cx.scope('mlp'), norm(cx.scope('ln_2'), x), n_hid=S * 4)
    x = x + m
    return x

def transformer(cx, tok_bt, *, n_vocab, n_head, n_layer, n_ctx, n_embd):
    B, T = tok_bt.shape
    pos_bt = jax.lax.broadcasted_iota(np.int32, (B, T), 1)
    tokenembs_qe = cx.get_variable('wte', initializer=lambda : normc(n_vocab, n_embd) * 0.1)
    posembs_pe = cx.get_variable('wpe', initializer=lambda : normc(n_ctx, n_embd) * 0.1)
    tokenemb_bte = tokenembs_qe[tok_bt]
    # assert isinstance(tok_bt, np.ndarray)
    posemb_bte = posembs_pe[pos_bt]
    last_bts = tokenemb_bte + posemb_bte
    if len(last_bts.shape) < 3:
        last_bts = last_bts[np.newaxis, ...]
    for layer in range(n_layer):
        #last_bts = block(cx.scope(f'h{layer:02d}'), last_bts, n_head=n_head)
        last_bts = block(cx.scope(f'h{layer:d}'), last_bts, n_head=n_head)
    last_bts = norm(cx.scope('ln_f'), last_bts, axis=-1)
    logits_btq = np.matmul(last_bts, tokenembs_qe.T)
    logprobs_btq = stax.logsoftmax(logits_btq)    
    return logprobs_btq


class TransformerV3:
  def __init__(self, config, params):
    self.config = config

    model_name = config['model_name']
    hparams = tf_model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    self.hparams = hparams

    n_ctx = config['seq']
    n_head = config['n_heads']
    n_layer = config['layers']
    n_embd = config['d_model']
    n_vocab = config['n_vocab']
    self.model = functools.partial(transformer, n_vocab=n_vocab,
        n_head=n_head, n_layer=n_layer, n_ctx=n_ctx, n_embd=n_embd)

    self.params = params
    self.cx = create_root_context()

    self.loss(jnp.zeros((1, n_ctx+1), dtype=jnp.int32)) # Just create variables
    self.cx.allow_new = False
    print_variables(self.cx)
    self.state = self.cx.variables_list()


  def loss(self, XY_bt):
    X_bt = XY_bt[:, :-1]
    B, T = X_bt.shape
    Y_bt = XY_bt[:, 1:]
    logprobs_btq = self.model(self.cx, X_bt)
    loglosses_bt = - logprobs_btq.reshape((B*T, -1))[ np.arange(B*T), Y_bt.reshape((-1,))]
    return loglosses_bt.mean()


  def tf_loss(self, context):
    X_bt = context[:, :-1]
    B, T = X_bt.shape
    Y_bt = context[:, 1:]
    output = tf_model.model(self.hparams, tf.convert_to_tensor(X_bt))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=context[:, 1:],
        logits=output['logits'])
    return tf.reduce_mean(loss)
    #loglosses_bt = - logprobs_btq.reshape((B*T, -1))[ np.arange(B*T), Y_bt.reshape((-1,))]
    #return loglosses_bt.mean()

  def eval_xmap(self, state, obs, target, ctx_length, use_tf=False):
    XY_bt = jnp.concatenate([obs, target[:, -1:]], axis=-1)
    return (self.tf_loss if use_tf else self.loss)(XY_bt)

  def eval(self, sample, use_tf=False):
    print("eval sample", sample["obs"].shape)
    print("eval target", sample["target"].shape)

    start = time.time()

    if "ctx_length" in sample:
        ctx_length = sample["ctx_length"]
    else:
        ctx_length = np.array([len(sample["obs"][0])] * len(sample["obs"]))

    out = self.eval_xmap(self.state, sample["obs"], sample["target"], ctx_length, use_tf=use_tf)
    print(f"eval dispatched in {time.time() - start:.06}s")

    # np.array(out["loss"])
    print(f"eval done in {time.time() - start:.06}s")
    return out


if __name__ == '__main__':
  np.random.seed(0)
  config = {
      'model_name': '117M',
      'cores_per_replica': 1,
      'seq': 1024,
      'n_vocab': 50257,
      'n_heads': 12,
      'layers': 12,
      'd_model': 768,
      'pe': 'fixed',
      # 'sampler': sampling.nucleaus_sample,
      }

  cores_per_replica = config['cores_per_replica']
  mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
  devices = np.array(jax.devices()).reshape(mesh_shape)
  #maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')), ())

  network = load(config)

  network.cx.name2val = network.params
  tf_model.tf.get_variable = get_variable
  X_bt = np.zeros((1, 1024), dtype=jnp.int32)
  Y_bt = np.zeros((1, 1024), dtype=jnp.int32)
  X = tf.convert_to_tensor(np.zeros((1, 1025), dtype=jnp.int32))
  jx_loss = network.eval({ 'obs': X_bt, 'target': Y_bt, })
  tf_loss = network.eval({ 'obs': X_bt, 'target': Y_bt, }, use_tf=True)
  print(jx_loss, tf_loss.numpy())
  self = network
  

