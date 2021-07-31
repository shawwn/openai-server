import jax
from jax import mask, jit, pmap, tree_util
from jax.experimental import PartitionSpec as P
from jax.experimental.pjit import pjit, pjit_p, with_sharding_constraint
import jax.numpy as jnp
from jax.experimental import maps, stax
import haiku as hk
import numpy as np
import os
from pprint import pprint as pp

from src import model as tf_model
from src import encoder

import json
import functools
from functools import partial
import time
import random

gBreak = False
gBreak2 = False

def load_tensor(reader, name: str) -> np.array:
  #name = '.'.join(name.split('/')[1:])
  value = reader.get_tensor(name)
  # value = jnp.array(value)
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

@tree_util.register_pytree_node_class
class VariableContext(object):
    def __init__(self, name2val, prefix, allow_new=True):
        self.name2val = name2val
        self.prefix = prefix
        self.allow_new = allow_new
    def tree_flatten(self):
        return ((self.name2val,), {'prefix': self.prefix, 'allow_new': self.allow_new})
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    def __eq__(self, other):
        return type(self) is type(other) and (self.name2val, self.prefix) == (other.name2val, other.prefix)
    def __repr__(self):
        if self.allow_new:
            return "VariableContext(prefix={})".format(self.prefix)
        else:
            return "VariableContext(prefix={}, allow_new=False)".format(self.prefix)
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
            # val = jnp.array(val)
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

# def gelu(x):
#     return 0.5*x*(1+np.tanh(0.79788*(x+0.044715*x**3)))

# from jax.nn import gelu
from jax.experimental.stax import gelu

@jax.jit
def _norm(x, g, b, e=1e-5):
    axis = -1
    u = jnp.mean(x, axis=axis, keepdims=True)
    s = jnp.mean(jnp.square(x-u), axis=axis, keepdims=True)
    x = (x - u) / jnp.sqrt(s + e)
    assert g is not None and b is not None
    x = x * g + b
    return x

# @partial(jax.jit, static_argnames=['cx'])
@jax.jit
def norm(cx, x):
    axis = -1
    n_state = x.shape[axis]
    g = cx.get_variable("g", initializer=lambda : np.ones(n_state, 'f'))
    b = cx.get_variable("b", initializer=lambda : np.zeros(n_state, 'f'))
    return _norm(x, g, b)

@partial(jax.jit, static_argnames=['nd', 'ns', 'dtype'])
def attention_mask(nd, ns, *, dtype):
    i = jnp.arange(nd)[:,None]
    j = jnp.arange(ns)
    m = i >= j - ns + nd
    return m.astype(dtype)

@jax.jit
def mask_attn_weights(w):
    *_, nd, ns = w.shape
    b = attention_mask(nd, ns, dtype=w.dtype)
    b = jnp.reshape(b, (1, 1, nd, ns))
    w = w * b - jnp.array(1e9, dtype=w.dtype) * (1 - b)
    return w

@jax.jit
def _attn(Q_bhtr, K_bhrt, V_bhtr):
    R = Q_bhtr.shape[-1]
    W_bhtt = jnp.matmul(Q_bhtr, K_bhrt) / jnp.sqrt(R)
    W_bhtt = mask_attn_weights(W_bhtt)
    W_bhtt = stax.softmax(W_bhtt, axis=-1)
    A_bhtr = jnp.matmul(W_bhtt, V_bhtr)
    return A_bhtr

@partial(jax.jit, static_argnames=['F'])
def _dense(X_btk, W_kf, b_f, F):
    B, T, K = X_btk.shape
    X_bt_k = jnp.reshape(X_btk, (-1, K))
    Y_bt_f = jnp.matmul(X_bt_k, W_kf) + b_f
    return jnp.reshape(Y_bt_f, (B, T, F))

@partial(jax.jit, static_argnames=['F'])
def dense(cx, X_btk, F):
    B, T, K = X_btk.shape
    W_kf = cx.get_variable("w", initializer=lambda: normc(K, F))
    b_f = cx.get_variable("b", initializer=lambda: np.zeros(F,'f'))
    return _dense(X_btk, W_kf, b_f, F)

@partial(jax.jit, static_argnames=['axis'])
def unstack(a, axis=0):
    return [jnp.squeeze(e, axis) for e in jnp.split(a, a.shape[axis], axis = axis)]

def past_length(past):
  if past is None:
    return 0
  elif isinstance(past, (list, tuple)):
    K_bhrt, V_bhtr = past[0]
    return K_bhrt.shape[-1]
  else:
    KV_bthr = past
    return KV_bthr.shape[-3]

# @partial(jax.jit, static_argnames=['n_state', 'n_head'])
def attn(cx, X_btk, n_state, n_head, past=None):
    B, T, _K = X_btk.shape
    assert n_state % n_head==0
    QKV_b_t_3s = dense(cx.scope('c_attn'), X_btk, n_state * 3)
    QKV_b_t_3h_r = jnp.reshape(QKV_b_t_3s, (B, T, 3 * n_head, n_state // n_head))
    Q_bthr, K_bthr, V_bthr = jnp.split(QKV_b_t_3h_r, 3, axis=2)
    # present = jnp.stack([K_bthr, V_bthr], axis=1)
    # if past is not None:
    #     pk, pv = unstack(past, axis=1)
    #     K_bthr = jnp.concatenate([pk, K_bthr], axis=-3)
    #     V_bthr = jnp.concatenate([pv, V_bthr], axis=-3)
    # (Pdb) present.shape
    # (1, 2, 1024, 12, 64)
    # (Pdb) K_bthr.shape
    # (1, 1024, 12, 64)
    # (Pdb) V_bthr.shape
    # (1, 1024, 12, 64)
    Q_bhtr = jnp.transpose(Q_bthr, (0, 2, 1, 3))
    K_bhrt = jnp.transpose(K_bthr, (0, 2, 3, 1))
    V_bhtr = jnp.transpose(V_bthr, (0, 2, 1, 3))
    #present = jnp.stack([K_bhrt, V_bhtr], axis=1)
    # present = [K_bhrt, V_bhtr]
    if past is not None:
        # pk, pv = unstack(past, axis=1)
        pk, pv = past
        K_bhrt = jnp.concatenate([pk, K_bhrt], axis=-1)
        V_bhtr = jnp.concatenate([pv, V_bhtr], axis=-2)
    present = [K_bhrt, V_bhtr]
    global gBreak
    if gBreak:
      gBreak = False
      breakpoint()
    global gBreak2
    if gBreak2 and past is not None:
      gBreak2 = False
      breakpoint()
    A_bhtr = _attn(Q_bhtr, K_bhrt, V_bhtr)
    A_bthr = jnp.transpose(A_bhtr, (0, 2, 1, 3))
    A_bts = jnp.reshape(A_bthr, (B, T, n_state))
    P_bts = dense(cx.scope('c_proj'), A_bts, n_state)
    return P_bts, present

@partial(jax.jit, static_argnames=['n_hid'])
def mlp(cx, X_bts, *, n_hid):
    S = X_bts.shape[-1]
    H_bth = gelu(dense(cx.scope('c_fc'), X_bts, n_hid))
    Y_bts = dense(cx.scope('c_proj'), H_bth, S)
    return Y_bts

def block(cx, x, *, n_head, past=None):
    S = x.shape[-1]
    a, present = attn(cx.scope('attn'), norm(cx.scope('ln_1'), x), S, n_head, past=past)
    x = x + a
    m = mlp(cx.scope('mlp'), norm(cx.scope('ln_2'), x), n_hid=S * 4)
    x = x + m
    return x, present

def transformer(cx, tok_bt, *, n_vocab, n_head, n_layer, n_ctx, n_embd, past=None):
    B, T = tok_bt.shape
    pos_bt = jax.lax.broadcasted_iota(jnp.int32, (B, T), 1)
    pos_bt = pos_bt + past_length(past)
    tokenembs_qe = cx.get_variable('wte', initializer=lambda: normc(n_vocab, n_embd) * 0.1)
    posembs_pe = cx.get_variable('wpe', initializer=lambda: normc(n_ctx, n_embd) * 0.1)
    #tokenemb_bte = tokenembs_qe[tuple(tok_bt)]
    tokenemb_bte = tokenembs_qe[tok_bt]
    posemb_bte = posembs_pe[pos_bt]
    last_bts = tokenemb_bte + posemb_bte
    if len(last_bts.shape) < 3:
        last_bts = last_bts[jnp.newaxis, ...]
    presents = []
    #pasts = unstack(past, axis=1) if past is not None else [None] * n_layer
    pasts = past if past is not None else [None] * n_layer
    prev_bts = None
    for layer in range(n_layer):
        prev_bts = last_bts
        last_bts, present = block(cx.scope(f'h{layer:d}'), last_bts, n_head=n_head, past=pasts[layer])
        presents.append(present)
    last_bts = norm(cx.scope('ln_f'), last_bts)
    logits_btq = np.matmul(last_bts, tokenembs_qe.T)
    # logits_btq = jnp.matmul(last_bts, tokenembs_qe.T)
    # breakpoint()
    #presents = np.stack(presents, axis=1)
    return logits_btq, presents

def scan(f, init, xs, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, np.stack(ys)


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


  def loss(self, XY_bt, past=None):
    X_bt = XY_bt[:, :-1]
    B, T = X_bt.shape
    Y_bt = XY_bt[:, 1:]
    logits_btq, presents = self.model(self.cx, X_bt, past=past)
    logprobs_btq = stax.logsoftmax(logits_btq)
    loglosses_bt = - logprobs_btq.reshape((B*T, -1))[ np.arange(B*T), Y_bt.reshape((-1,))]
    return loglosses_bt.mean(), presents


  def tf_loss(self, context, past=None):
    X_bt = context[:, :-1]
    B, T = X_bt.shape
    Y_bt = context[:, 1:]
    output = tf_model.model(self.hparams, tf.convert_to_tensor(X_bt), past=past)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=context[:, 1:],
        logits=output['logits'])
    return tf.reduce_mean(loss), output['present']
    #loglosses_bt = - logprobs_btq.reshape((B*T, -1))[ np.arange(B*T), Y_bt.reshape((-1,))]
    #return loglosses_bt.mean()

  def generate_one_token(self, context, sample_key=None, *, past=None, **sampler_options):
    logprobs_btq, presents = self.model(self.cx, context, past=past)
    sampler_input = None
    if sample_key is None:
      sample_key = jax.random.PRNGKey(random.randint(0, 2 ** 60))
    sample_key, new_key = jax.random.split(sample_key)
    sampler = self.config['sampler']
    logits = logprobs_btq[..., -1:, :]
    next_token, sample_info = sampler(sample_key, logits, sampler_input, **sampler_options)
    return next_token, new_key, presents

  def generate_tokens(self, tokenizer, prompt, length, sample_key=None, *, temp=0.8, **sampler_options):
    for i in range(length):
      context = np.array(tokenizer.encode(prompt))[None, :]
      token, sample_key, past = self.generate_one_token(context, sample_key=sample_key, temp=temp, **sampler_options)
      prompt += tokenizer.decode(token[0])
    return prompt

  def generate_tokens2(self, tokenizer, prompt, length, sample_key=None, *, temp=0.8, **sampler_options):
    context = np.array(tokenizer.encode(prompt))[None, :]
    past = None
    for i in range(length):
      context, sample_key, presents = self.generate_one_token(context, sample_key=sample_key, past=past, temp=temp, **sampler_options)
      if past is None:
        past = presents
      else:
        past = np.concatenate([past, presents], axis=-3)
      prompt += tokenizer.decode(context[0])
    return prompt

  def eval_xmap(self, state, obs, target, ctx_length, use_tf=False, past=None):
    XY_bt = jnp.concatenate([obs, target[:, -1:]], axis=-1)
    return (self.tf_loss if use_tf else self.loss)(XY_bt, past=past)

  def eval(self, sample, use_tf=False):
    print("eval sample", sample["obs"].shape)
    print("eval target", sample["target"].shape)

    start = time.time()

    if "ctx_length" in sample:
        ctx_length = sample["ctx_length"]
    else:
        ctx_length = np.array([len(sample["obs"][0])] * len(sample["obs"]))

    out, presents = self.eval_xmap(self.state, sample["obs"], sample["target"], ctx_length, use_tf=use_tf)
    print(f"eval dispatched in {time.time() - start:.06}s")

    # np.array(out["loss"])
    print(f"eval done in {time.time() - start:.06}s")
    return out

  def generate_initial(self, context, ctx_length, key):
    count = ctx_length[-1]
    assert (ctx_length == count).all()
    initial_context = context[..., context.shape[-1] - count:-1]
    last = context[..., -1:]
    logits, presents = self.model(self.cx, initial_context)
    # return logits, presents
    initial_logits = logits[..., -1, :]
    initial_presents = presents
    return initial_logits, (last, initial_presents, key)

  def generate_once(self, next_token, decode_state):
    logits, presents = self.model(self.cx, next_token, past=decode_state)
    #assert logits.shape[-2] == 1
    #next_logits = logits[..., -1, :]
    next_logits = logits
    # next_presents = np.concatenate([decode_state, presents], axis=-3)
    next_presents = presents
    return next_logits, next_presents

  def generate_xmap(self, state, key, ctx, ctx_length, aux, sampler_options):
    sampler = config["sampler"]
    gen_length = self.gen_length

    def generate_sample(context, ctx_length, aux):
      # initial_logits, initial_presents = self.generate_initial(context, ctx_length)
      # initial_state = [context[:, -1], initial_presents, key[0]]
      _, initial_state = self.generate_initial(context, ctx_length, key=key[0])

      def generate_scan_fn(carry, sampler_input):
        next_token, decode_state, sample_key = carry
        sample_key, new_key = jax.random.split(sample_key)

        output, new_state = self.generate_once(next_token, decode_state)
        next_token, sample_info = sampler(sample_key, output, sampler_input, **sampler_options)

        #output = (next_token, sample_info)
        output = next_token
        new_carry = (next_token, new_state, new_key)
        return new_carry, output

      # final_state, outputs = jax.lax.scan(generate_scan_fn, initial_state, xs=aux[0], length=gen_length)
      final_state, outputs = scan(generate_scan_fn, initial_state, xs=None, length=gen_length)
      return final_state, outputs[None, ...]

    # generate_fn = hk.transform(generate_sample).apply
    # return generate_fn(state["params"], key, ctx, ctx_length, aux)
    return generate_sample(ctx, ctx_length, aux)


  def generate(self, ctx, ctx_length, gen_length, sampler_options):
    key = hk.PRNGSequence(random.randint(0, 2 ** 60))

    batch_size = ctx.shape[0]
    aux = jnp.zeros((batch_size, gen_length), dtype=jnp.uint32)
    self.gen_length = gen_length

    return self.generate_xmap(self.state,
                              jnp.array(key.take(batch_size)),
                              ctx,
                              np.array(ctx_length, dtype=np.uint32),
                              aux,
                              sampler_options)
  

# takes in a logit distribution, softmax and then sample
def softmax_sample(key, logits, _, temp=1):
    return jax.random.categorical(key, logits/temp, -1).astype(jnp.uint32), None


def bucket_jit(f):
    compiled_f = jit(mask(f, ['n'], ''))
    def wrapped(x):
        amount = 128 - x.shape[0] % 128
        padded_x = jnp.pad(x, (0, amount))
        return compiled_f([padded_x], dict(n=x.shape[0]))
    return wrapped


@bucket_jit
def foo(x):
    print("recompiling!", x)  # actually retracing, but effectively correct
    return jnp.sum(x)


# foo(np.arange(4))  # recompiling!
# foo(np.arange(5))
# foo(np.arange(6))
# foo(np.arange(300))  # recompiling!


# https://jax.readthedocs.io/en/latest/jax.html#jax.jit

@partial(pmap, axis_name='rows')
@partial(pmap, axis_name='cols')
def normalize(x):
    row_normed = x / jax.lax.psum(x, 'rows')
    col_normed = x / jax.lax.psum(x, 'cols')
    doubly_normed = x / jax.lax.psum(x, ('rows', 'cols'))
    return row_normed, col_normed, doubly_normed

# >>> x = jnp.arange(8.).reshape((4, 2))
# >>> row_normed, col_normed, doubly_normed = normalize(x)
# >>> print(row_normed.sum(0))
# [ 1.  1.]
# >>> print(col_normed.sum(1))
# [ 1.  1.  1.  1.]
# >>> print(doubly_normed.sum((0, 1)))

def test_pjit():
  f = partial(pjit, in_axis_resources=(P('dp'), P('dp')), out_axis_resources=None)(lambda x, y: x + y)
  shape = (8, 8)
  x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
  actual = f(x, x + 1)
  expected = x + (x + 1)
  assert len(actual.device_buffers) == 8
  # self.assertAllClose(actual, expected, check_dtypes=False)
  # self.assertIsInstance(actual, pxla.ShardedDeviceArray)
  # self.assertLen(actual.device_buffers, 2)
  # self.assertAllClose(actual.device_buffers[0].to_py(), expected,
  #                     check_dtypes=False)


def test_basic2d():
    # @partial(pjit,
    #          in_axis_resources=(P(None, 'x', 'y'), P('y')),
    #          out_axis_resources=P('x'))
    # def f(x, y):
    #   return x @ y
    @partial(pjit,
             in_axis_resources=(P(None, 'dp', 'mp'), P('mp')),
             out_axis_resources=P('dp'))
    def f(x, y):
      return x @ y

    # x_shape = (8, 6, 4)
    # y_shape = (4, 2)
    dp_size = 1
    mp_size = 8
    x_shape = (8, dp_size, mp_size)
    y_shape = (mp_size, 2)
    x = jnp.arange(np.prod(x_shape)).reshape(x_shape)
    y = jnp.arange(np.prod(y_shape)).reshape(y_shape)
    actual = f(x, y)
    expected = x @ y
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertIsInstance(actual, pxla.ShardedDeviceArray)
    self.assertLen(actual.device_buffers, 4)



def test_sharding_constraint():
  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testShardingConstraintPyTree(self):
    @partial(pjit, in_axis_resources=None, out_axis_resources=None)
    def f(x):
      x = with_sharding_constraint(x, [P('x', 'y'), P('y', 'x')])
      x = x.copy()
      x[0]["a"] *= 2
      return x

    shape = (8, 8)
    v = np.arange(prod(shape)).reshape(shape)
    x = [{"a": v, "b": v * 2}, v * 3]
    actual = f(x)

    expected = x.copy()
    expected[0]["a"] *= 2
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertLen(actual[0]["a"].device_buffers, 2)

    hlo = jax.xla_computation(f)(x)
    # Annotations from with_sharding_constraint
    self.assertIn("sharding={devices=[2,1]0,1}", hlo.as_hlo_text())
    self.assertIn("sharding={devices=[1,2]0,1}", hlo.as_hlo_text())
    # Annotation from pjit
    self.assertIn("sharding={replicated}", hlo.as_hlo_text())

  @partial(pjit, in_axis_resources=None, out_axis_resources=None)
  def f(x):
    x = with_sharding_constraint(x, [P('dp', 'mp'), P('mp', 'dp')])
    x = x.copy()
    x[0]["a"] *= 2
    return x

  shape = (1, 8)
  v = np.arange(np.prod(shape)).reshape(shape)
  x = [{"a": v, "b": v * 2}, v * 3]
  actual = f(x)


if __name__ == '__main__':
  np.random.seed(0)
  config = {
      'model_name': '117M',
      'cores_per_replica': jax.device_count(), #1,
      'seq': 1024,
      'n_vocab': 50257,
      'n_heads': 12,
      'layers': 12,
      'd_model': 768,
      'pe': 'fixed',
      # 'sampler': sampling.nucleaus_sample,
      'sampler': softmax_sample,
      'per_replica_batch': 1,
      }

  tokenizer = encoder.get_encoder(config['model_name'])

  cores_per_replica = config['cores_per_replica']
  mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
  devices = np.array(jax.devices()).reshape(mesh_shape)
  maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')), ())

  with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
    network = load(config)

    network.cx.name2val = network.params

    # xs, tree = tree_util.tree_flatten(network.cx)
    # actual = tree_util.tree_unflatten(tree, xs)

    compare_with_tensorflow = False
    if compare_with_tensorflow:
      tf_model.tf.get_variable = get_variable
      X_bt = np.zeros((1, 1024), dtype=jnp.int32)
      Y_bt = np.zeros((1, 1024), dtype=jnp.int32)
      X = tf.convert_to_tensor(np.zeros((1, 1025), dtype=jnp.int32))
      jx_loss = network.eval({ 'obs': X_bt, 'target': Y_bt, })
      tf_loss = network.eval({ 'obs': X_bt, 'target': Y_bt, }, use_tf=True)
      print(jx_loss, tf_loss.numpy())

    per_replica_batch = config["per_replica_batch"]
    total_batch = per_replica_batch * jax.device_count() // cores_per_replica

    for i in range(8):
      # prompt = input("Type input:")
      prompt = "Hello, my name is"
      tokens = tokenizer.encode(prompt)

      start = time.time()

      provided_ctx = len(tokens)
      pad_amount = config['seq'] - provided_ctx

      padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
      batched_tokens = np.array([padded_tokens] * total_batch)
      length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

      output = network.generate(batched_tokens, length, 16, {#"top_p": np.ones(total_batch) * 0.9,
                                                             "temp": np.ones(total_batch) * 0.75})

      print(f"completion done in {time.time() - start:06}s")
      completion = prompt + tokenizer.decode(np.squeeze(output[1]))
      print(repr(completion))



