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
import math
import re

gState = {}

if jax.default_backend() == 'tpu' and bool(int(os.environ.get('BF16', '1'))):
  default_dtype = jnp.bfloat16
  print('Using bfloat16')
else:
  default_dtype = jnp.float32

def nextpow2(x):
    """Returns the next power of 2 greater than or equal to `x`"""
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

def nextbucket(x, size):
  return (x + (size - 1)) // size * size

def load_tensor(reader, name: str, dtype=None) -> np.array:
  #name = '.'.join(name.split('/')[1:])
  value = reader.get_tensor(name)
  value = jnp.array(np.squeeze(value), dtype=dtype or default_dtype)
  #key = '.'.join(name.split('/'))
  key = pathutil.normpath('///' + name)
  # if value.shape and value.shape[0] == 1:
  #   value = np.squeeze(value, axis=0)
  return key, value

def load_state(name: str, dtype=None):
  from tensorflow.python.training import py_checkpoint_reader
  reader = py_checkpoint_reader.NewCheckpointReader( 'models/{name}/model.ckpt'.format(name=name) )
  state_dict = dict([load_tensor(reader, k, dtype=dtype) for k in list(reader.get_variable_to_shape_map().keys())])
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

# Normalize the case of a pathname.  Trivial in Posix, string.lower on Mac.
# On MS-DOS this may also turn slashes into backslashes; however, other
# normalizations (such as optimizing '../' away) are not allowed
# (another function should be defined to do that).

def normcase(s):
    """Normalize case of pathname.  Has no effect under Posix"""
    return os.fspath(s)


import genericpath


class pathutil:
    def _get_sep(path):
        if isinstance(path, bytes):
            return b'/'
        else:
            return '/'

    # Return whether a path is absolute.
    # Trivial in Posix, harder on the Mac or MS-DOS.

    @classmethod
    def isabs(cls, s):
        """Test whether a path is absolute"""
        s = os.fspath(s)
        sep = cls._get_sep(s)
        return s.startswith(sep)


    # Join pathnames.
    # Ignore the previous parts if a part is absolute.
    # Insert a '/' unless the first part is empty or already ends in '/'.

    @classmethod
    def join(cls, a, *p):
        """Join two or more pathname components, inserting '/' as needed.
        If any component is an absolute path, all previous path components
        will be discarded.  An empty last part will result in a path that
        ends with a separator."""
        a = os.fspath(a)
        sep = cls._get_sep(a)
        path = a
        try:
            if not p:
                path[:0] + sep  #23780: Ensure compatible data type even if p is null.
            for b in map(os.fspath, p):
                if b.startswith(sep):
                    path = b
                elif not path or path.endswith(sep):
                    path += b
                else:
                    path += sep + b
        except (TypeError, AttributeError, BytesWarning):
            genericpath._check_arg_types('join', a, *p)
            raise
        return path


    # Split a path in head (everything up to the last '/') and tail (the
    # rest).  If the path ends in '/', tail will be empty.  If there is no
    # '/' in the path, head  will be empty.
    # Trailing '/'es are stripped from head unless it is the root.

    @classmethod
    def split(cls, p):
        """Split a pathname.  Returns tuple "(head, tail)" where "tail" is
        everything after the final slash.  Either part may be empty."""
        p = os.fspath(p)
        sep = cls._get_sep(p)
        i = p.rfind(sep) + 1
        head, tail = p[:i], p[i:]
        if head and head != sep*len(head):
            head = head.rstrip(sep)
        return head, tail

    # Normalize a path, e.g. A//B, A/./B and A/foo/../B all become A/B.
    # It should be understood that this may change the meaning of the path
    # if it contains symbolic links!

    def normpath(path):
        """Normalize path, eliminating double slashes, etc."""
        path = os.fspath(path)
        if isinstance(path, bytes):
            sep = b'/'
            empty = b''
            dot = b'.'
            dotdot = b'..'
        else:
            sep = '/'
            empty = ''
            dot = '.'
            dotdot = '..'
        if path == empty:
            return dot
        initial_slashes = path.startswith(sep)
        # POSIX allows one or two initial slashes, but treats three or more
        # as single slash.
        if (initial_slashes and
            path.startswith(sep*2) and not path.startswith(sep*3)):
            initial_slashes = 2
        comps = path.split(sep)
        new_comps = []
        for comp in comps:
            if comp in (empty, dot):
                continue
            if (comp != dotdot or (not initial_slashes and not new_comps) or
                 (new_comps and new_comps[-1] == dotdot)):
                new_comps.append(comp)
            elif new_comps:
                new_comps.pop()
        comps = new_comps
        path = sep.join(comps)
        if initial_slashes:
            path = sep*initial_slashes + path
        return path or dot


    # Return the tail (basename) part of a path, same as split(path)[1].

    @classmethod
    def basename(cls, p):
        """Returns the final component of a pathname"""
        p = os.fspath(p)
        sep = cls._get_sep(p)
        i = p.rfind(sep) + 1
        return p[i:]


    # Return the head (dirname) part of a path, same as split(path)[0].

    @classmethod
    def dirname(cls, p):
        """Returns the directory component of a pathname"""
        p = os.fspath(p)
        sep = cls._get_sep(p)
        i = p.rfind(sep) + 1
        head = p[:i]
        if head and head != sep*len(head):
            head = head.rstrip(sep)
        return head



# ================================================================
# Tf-like framework for Jax
# ================================================================

def create_root_context(state=None, *, prefix='/', **static_kwargs):
    if state is None:
        state = {}
    return VariableContext(state, prefix=prefix, **static_kwargs)

@tree_util.register_pytree_node_class
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

    def get_variable(self, name, initializer=None):
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
        return pathutil.normpath(pathutil.join(*xs))

    def variables_list(self):
        return list(self.name2val.values())

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

from jax.nn import gelu

# @partial(jax.jit, static_argnames=['eps', 'axis'])
@jax.named_call
def _norm(x, g, b, *, eps=1e-5, axis=-1):
    u = jnp.mean(x, axis=axis, keepdims=True)
    s = jnp.mean(jnp.square(x-u), axis=axis, keepdims=True)
    x = (x - u) / jnp.sqrt(s + eps)
    assert g is not None and b is not None
    x = x * g + b
    return x

# @partial(jax.jit, static_argnames=['cx'])
# @partial(jax.jit, static_argnames=['eps', 'axis'])
@jax.named_call
def norm(cx, x, *, eps=1e-5, axis=-1):
    n_state = x.shape[axis]
    g = cx.get_variable("g", initializer=lambda : np.ones(n_state, 'f'))
    b = cx.get_variable("b", initializer=lambda : np.zeros(n_state, 'f'))
    return _norm(x, g, b, eps=eps, axis=axis)

# @partial(jax.jit, static_argnames=['nd', 'ns', 'dtype'])
@jax.named_call
def attention_mask(nd, ns, *, dtype):
    i = jnp.arange(nd)[:,None]
    j = jnp.arange(ns)
    m = i >= j - ns + nd
    return m.astype(dtype)

# @jax.jit
@jax.named_call
def mask_attn_weights(w):
    *_, nd, ns = w.shape
    if nd <= 1:
      return w
    b = attention_mask(nd, ns, dtype=w.dtype)
    b = jnp.reshape(b, (1, 1, nd, ns))
    w = w * b - jnp.array(1e9, dtype=w.dtype) * (1 - b)
    return w

# @partial(jax.jit, static_argnames=['F'])
@jax.named_call
def _dense(X_btk, W_kf, b_f, F):
    B, T, K = X_btk.shape
    X_bt_k = jnp.reshape(X_btk, (-1, K))
    Y_bt_f = jnp.matmul(X_bt_k, W_kf) + b_f
    return jnp.reshape(Y_bt_f, (B, T, F))

# @partial(jax.jit, static_argnames=['F'])
@jax.named_call
def dense(cx, X_btk, F):
    B, T, K = X_btk.shape
    W_kf = cx.get_variable("w", initializer=lambda: normc(K, F))
    b_f = cx.get_variable("b", initializer=lambda: np.zeros(F,'f'))
    return _dense(X_btk, W_kf, b_f, F)

# @partial(jax.jit, static_argnames=['axis'])
@jax.named_call
def unstack(a, axis=0):
    return [jnp.squeeze(e, axis) for e in jnp.split(a, a.shape[axis], axis = axis)]

def past_length(past):
  if past is None:
    return 0
  elif isinstance(past, (list, tuple)):
    K_bthr, V_bthr = past[0]
    return V_bthr.shape[-3]
  else:
    KV_bthr = past
    return KV_bthr.shape[-3]

# @partial(jax.jit, static_argnames=['n_state', 'n_head'])
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

# @partial(jax.jit, static_argnames=[])
@jax.named_call
def mlp(cx, X_bts):
    S = X_bts.shape[-1]
    n_hid = S * 4
    H_bth = gelu(dense(cx.scope('c_fc'), X_bts, n_hid))
    Y_bts = dense(cx.scope('c_proj'), H_bth, S)
    return Y_bts

def block(cx, x, past):
    a, present = attn(cx.scope('attn'), norm(cx.scope('ln_1'), x), past)
    x = x + a
    m = mlp(cx.scope('mlp'), norm(cx.scope('ln_2'), x))
    x = x + m
    return x, present

@jax.named_call
def initial_embed(cx, tok_bt, past_len=None):
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

# @jax.jit
@jax.named_call
def final_embed(cx, last_bts):
    tokembs_qe = cx.get_variable('wte')
    last_bts = norm(cx.scope('ln_f'), last_bts)
    logits_btq = jnp.matmul(last_bts, tokembs_qe.T)
    return logits_btq

@jax.named_call
def transformer(cx, tok_bt, past=None, past_len=None):
    if past_len is None:
      past_len = past_length(past)
    last_bts = initial_embed(cx, tok_bt, past_len)
    presents = []
    pasts = past if past is not None else [None] * cx.n_layer
    for layer in range(cx.n_layer):
        name = f'h{layer:d}'
        last_bts, present = jax.named_call(block, name='block_'+name)(cx.scope(name), last_bts, pasts[layer])
        presents.append(present)
    logits_btq = final_embed(cx, last_bts)
    return logits_btq, presents

@jax.named_call
def scan(f, init, xs, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, jnp.stack(ys)


@jax.named_call
def padpasts(pasts, past_len):
  n = past_length(pasts)
  k = past_len - n
  if k > 0:
    return jax.tree_util.tree_map(lambda x: jnp.pad(x, ((0,0),(0,k),(0,0),(0,0))), pasts)
  return pasts



class TransformerV3:
  def __init__(self, config, params=None, prefix='/model'):
    self.config = config

    model_name = config['model_name']
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        self.hparams = json.load(f)

    self.cx = create_root_context(params, prefix=prefix, **self.hparams)

    self.model = transformer
    # self.model_jit = jax.jit(self.model)

    self.loss(jnp.zeros((1, self.cx.n_ctx+1), dtype=jnp.int32)) # Just create variables
    self.cx.allow_new = False
    print_variables(self.cx)

    cx_spec = jax.tree_util.tree_map(lambda x: re.sub(r'[0-9]+', '_', str(x.shape)), self.cx)
    past_spec_i = [['(_, n, _, _)', '(_, n, _, _)'] for _ in range(self.cx.n_layer)]
    past_spec_o = [['(_, n + 1, _, _)', '(_, n + 1, _, _)'] for _ in range(self.cx.n_layer)]
    @partial(mask, in_shapes=[cx_spec, '(1, 1)', past_spec_i, ''], out_shape=('(1, 1, _)', past_spec_o))
    @jax.named_call
    def generate_once(cx, next_token, decode_state, decode_len):
      next_logits, next_presents = self.model(cx, next_token, decode_state, decode_len)
      return next_logits, next_presents
    #self.generate_once_masked = jax.jit(generate_once)
    self.generate_once_masked = generate_once
    @jax.named_call
    def generate_once_wrapped(cx, next_token, decode_state, decode_len):
        pp(dict(name='_generate_once_wrapped', next_token=next_token, decode_state=decode_state, decode_len=decode_len))
        n = decode_len.item()
        k = (n + 15) // 16 * 16
        padded_state = padpasts(decode_state, k)
        return self.generate_once_masked([cx, next_token, padded_state, decode_len], dict(n=n))
    self.generate_once_wrapped = generate_once_wrapped


  @jax.named_call
  def gen_once(self, cx, next_token, decode_state, decode_len):
    output, new_state = self.generate_once_masked([cx, next_token, decode_state, decode_len], dict(n=decode_len))
    clipped_state = jax.tree_util.tree_map(lambda x: x[:, 0:-1, :, :], new_state)
    return output, clipped_state
    

  @jax.named_call
  def loss(self, XY_bt, past=None):
    X_bt = XY_bt[:, :-1]
    B, T = X_bt.shape
    Y_bt = XY_bt[:, 1:]
    logits_btq, presents = self.model(self.cx, X_bt, past)
    logprobs_btq = stax.logsoftmax(logits_btq)
    loglosses_bt = - logprobs_btq.reshape((B*T, -1))[ np.arange(B*T), Y_bt.reshape((-1,))]
    return loglosses_bt.mean(), presents

  def tf_hparams(self):
    hparams = tf_model.default_hparams()
    hparams.override_from_dict(self.hparams)
    return hparams

  def tf_loss(self, context, past=None):
    X_bt = context[:, :-1]
    B, T = X_bt.shape
    Y_bt = context[:, 1:]
    output = tf_model.model(self.tf_hparams(), tf.convert_to_tensor(X_bt), past=past)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=context[:, 1:],
        logits=output['logits'])
    return tf.reduce_mean(loss), output['present']
    #loglosses_bt = - logprobs_btq.reshape((B*T, -1))[ np.arange(B*T), Y_bt.reshape((-1,))]
    #return loglosses_bt.mean()

  @jax.named_call
  def generate_one_token(self, context, sample_key=None, past=None, **sampler_options):
    logprobs_btq, presents = self.model(self.cx, context, past)
    sampler_input = None
    if sample_key is None:
      sample_key = jax.random.PRNGKey(random.randint(0, 2 ** 60))
    sample_key, new_key = jax.random.split(sample_key)
    sampler = self.config['sampler']
    logits = logprobs_btq[..., -1:, :]
    next_token, sample_info = sampler(sample_key, logits, sampler_input, **sampler_options)
    return next_token, new_key, presents

  @jax.named_call
  def generate_tokens(self, tokenizer, prompt, length, sample_key=None, *, temp=0.8, **sampler_options):
    for i in range(length):
      context = np.array(tokenizer.encode(prompt))[None, :]
      token, sample_key, past = self.generate_one_token(context, sample_key=sample_key, temp=temp, **sampler_options)
      prompt += tokenizer.decode(token[0])
    return prompt

  @jax.named_call
  def generate_tokens2(self, tokenizer, prompt, length, sample_key=None, *, temp=0.8, **sampler_options):
    context = np.array(tokenizer.encode(prompt))[None, :]
    past = None
    for i in range(length):
      context, sample_key, presents = self.generate_one_token(context, sample_key, past, temp=temp, **sampler_options)
      if past is None:
        past = presents
      else:
        past = np.concatenate([past, presents], axis=-3)
      prompt += tokenizer.decode(context[0])
    return prompt

  @jax.named_call
  def eval_xmap(self, state, obs, target, ctx_length, past=None, *, use_tf=False):
    XY_bt = jnp.concatenate([obs, target[:, -1:]], axis=-1)
    return (self.tf_loss if use_tf else self.loss)(XY_bt, past)

  @jax.named_call
  def eval(self, sample, use_tf=False):
    print("eval sample", sample["obs"].shape)
    print("eval target", sample["target"].shape)

    start = time.time()

    if "ctx_length" in sample:
        ctx_length = sample["ctx_length"]
    else:
        ctx_length = np.array([len(sample["obs"][0])] * len(sample["obs"]))

    out, presents = self.eval_xmap(self.cx, sample["obs"], sample["target"], ctx_length, use_tf=use_tf)
    print(f"eval dispatched in {time.time() - start:.06}s")

    # np.array(out["loss"])
    print(f"eval done in {time.time() - start:.06}s")
    return out

  @jax.named_call
  def generate_initial(self, cx, context, ctx_length, key, gen_length=0):
    count = ctx_length[-1]
    assert (ctx_length == count).all()
    initial_context = context[..., context.shape[-1] - count:-1]
    initial_token = context[..., -1:]
    logits, presents = self.model(cx, initial_context)
    initial_logits = logits[..., -1, :]
    initial_padding = 1 if gen_length <= 0 else nextbucket(gen_length, self.config.get('bucket_size', 16))
    pp(dict(_name='generate_initial', count=count, initial_padding=initial_padding, gen_length=gen_length, key=key))
    initial_presents = padpasts(presents, initial_padding)
    initial_len = jnp.array(past_length(presents))
    return initial_logits, (cx, initial_token, initial_presents, initial_len, key)

  @jax.named_call
  def generate_once(self, next_token, decode_state, decode_len):
    next_logits, next_presents = self.model(self.cx, next_token, decode_state, decode_len)
    return next_logits, next_presents

  @jax.named_call
  def generate_xmap(self, cx, key, ctx, ctx_length, aux, sampler_options):
    sampler = self.config["sampler"]
    gen_length = self.gen_length

    if not hasattr(self, 'generate_sample'):
      @partial(jax.jit, static_argnames=['gen_length'])
      @jax.named_call
      def generate_sample(initial_state, gen_length, sampler_options):

        @jax.named_call
        def generate_scan_fn(carry, sampler_input):
          state, next_token, decode_state, decode_len, sample_key = carry
          sample_key, new_key = jax.random.split(sample_key)

          output, new_state = self.gen_once(state, next_token, decode_state, decode_len)
          next_token, sample_info = sampler(sample_key, output, sampler_input, **sampler_options)

          output = next_token
          new_carry = (state, next_token, new_state, decode_len + 1, new_key)
          return new_carry, output

        final_state, outputs = jax.lax.scan(generate_scan_fn, initial_state, xs=None, length=gen_length)
        # final_state, outputs = scan(generate_scan_fn, initial_state, xs=None, length=gen_length)
        return final_state, outputs[None, ...]
      self.generate_sample = generate_sample

    # generate_fn = hk.transform(generate_sample).apply
    # return generate_fn(state["params"], key, ctx, ctx_length, aux)
    _, initial_state = self.generate_initial(cx, ctx, ctx_length, key[0], gen_length)
    result = self.generate_sample(initial_state, gen_length, sampler_options)
    return result

  @jax.named_call
  def generate(self, ctx, ctx_length, gen_length, sampler_options, seed=None):
    if seed is None:
      seed = random.randint(0, 2 ** 60)
    key = hk.PRNGSequence(seed)

    batch_size = ctx.shape[0]
    aux = jnp.zeros((batch_size, gen_length), dtype=jnp.uint32)
    self.gen_length = gen_length

    return self.generate_xmap(self.cx,
                              jnp.array(key.take(batch_size)),
                              ctx,
                              jnp.array(ctx_length, dtype=jnp.uint32),
                              aux,
                              sampler_options)
  

# takes in a logit distribution, softmax and then sample
@jax.named_call
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


from jax._src.numpy import lax_numpy
from jax import core
from jax.interpreters.masking import UniqueId, Poly, UndefinedPoly, eval_poly_shape
import opt_einsum

def _einsum_contract_path(*operands, **kwargs):
  """Like opt_einsum.contract_path, with support for DimPolynomial shapes.

  We use opt_einsum.contract_path to compute the schedule, using a fixed
  constant for all dimension variables. This is safe because we throw an
  error if there are more than 1 contractions. Essentially, we just use
  opt_einsum.contract_path to parse the specification.
  """

  # Replace the polymorphic shapes with some concrete shapes for calling
  # into opt_einsum.contract_path, because the latter wants to compute the
  # sizes of operands and intermediate results.
  fake_ops = []
  for operand in operands:
    # We replace only array operands
    if not hasattr(operand, "dtype"):
      fake_ops.append(operand)
    else:
      shape = np.shape(operand)
      def fake_dim(d):
        if core.is_constant_dim(d):
          return d
        else:
          if not isinstance(d, Poly):
            raise TypeError(f"Encountered unexpected shape dimension {d}")
          # It is Ok to replace all polynomials with the same value. We may miss
          # here some errors due to non-equal dimensions, but we catch them
          # later.
          return 8
      fake_ops.append(jax.ShapeDtypeStruct(tuple(map(fake_dim, shape)),
                                           operand.dtype))

  contract_fake_ops, contractions = opt_einsum.contract_path(*fake_ops,
                                                             **kwargs)
  if len(contractions) > 1:
    msg = ("Shape polymorphism is not yet supported for einsum with more than "
           f"one contraction {contractions}")
    raise ValueError(msg)
  contract_operands = []
  for operand in contract_fake_ops:
    idx = tuple(i for i, fake_op in enumerate(fake_ops) if operand is fake_op)
    assert len(idx) == 1
    contract_operands.append(operands[idx[0]])
  return contract_operands, contractions

lax_numpy._polymorphic_einsum_contract_path_handlers[Poly] = _einsum_contract_path



if __name__ == '__main__':
  np.random.seed(0)
  config = {
      'model_name': os.environ.get('MODEL_NAME', '345M'),
      'cores_per_replica': jax.device_count(), #1,
      # 'seq': 1024,
      # 'n_vocab': 50257,
      # 'n_heads': 12,
      # 'layers': 12,
      # 'd_model': 768,
      # 'pe': 'fixed',
      # 'sampler': sampling.nucleaus_sample,
      'sampler': softmax_sample,
      'per_replica_batch': 1,
      'bucket_size': int(os.environ.get('BUCKET_SIZE', '16')),
      }

  if bool(int(os.environ.get('TPU_CACHE', '0'))):
    try:
      from jax.experimental.compilation_cache import compilation_cache as cc
      cc.initialize_cache('cache')
      print('Using compilation cache')
    except AttributeError:
      print('Failed to initialize the compilation cache')

  tokenizer = encoder.get_encoder(config['model_name'])

  cores_per_replica = config['cores_per_replica']
  mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
  devices = np.array(jax.devices()).reshape(mesh_shape)
  maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')), ())

  with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
    network = load(config)

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

    def prepare_sample(prompt, seed=None, batch_size=1, temp=0.75, **sampler_options):
      sampler_options = jax.tree_util.tree_map(lambda x: np.ones(batch_size) * x, {'temp': temp, **sampler_options})
      seed = seed or random.randint(0, 2**60)
      key = jnp.array(hk.PRNGSequence(seed).take(batch_size))
      if isinstance(prompt, str):
        tokens = tokenizer.encode(prompt)
      else:
        tokens = prompt
      provided_ctx = len(tokens)
      pad_amount = max(0, network.cx.n_ctx - provided_ctx)
      padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
      batched_tokens = np.array([padded_tokens] * batch_size)
      length = np.ones(batch_size, dtype=np.uint32) * len(tokens)
      length = jnp.array(length, dtype=jnp.uint32)
      return batched_tokens, length, key, sampler_options

    def sample(prompt, gen_length, temp=0.75, seed=None, echo=True, batch_size=1):
      batched_tokens, length, key, sampler_options = prepare_sample(prompt, batch_size=batch_size, temp=temp, seed=seed)
      start = time.time()
      _, output = network.generate(batched_tokens, length, gen_length, sampler_options, seed=seed)
      output.block_until_ready()
      print(f"completion done in {time.time() - start:06}s")
      start = time.time()
      output_ = np.squeeze(output)
      print(f"squeezed in {time.time() - start:06}s")
      start = time.time()
      completion = tokenizer.decode(output_)
      print(f"decoded in {time.time() - start:06}s")
      if echo:
        completion = prompt + completion
      return completion

    for i in range(3):
      # prompt = input("Type input:")
      prompt = os.environ.get('PROMPT', "Hello, my name is")
      tokens = tokenizer.encode(prompt)

      completion = sample(prompt, int(os.environ.get('MAX_TOKENS', '128')), seed=i+1)
      print(repr(completion))

    if True:
      breakpoint()
      print('')



