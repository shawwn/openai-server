import os
import functools
import numpy as np
import tensorflow.compat.v1 as tf
from typing import Any, List, Tuple, Union

def op_scope(f, name=None):
  if name is None:
    name = f.__name__
  @functools.wraps(f)
  def __f(*args, **keys):
    with tf.name_scope(name):
      return f(*args, **keys)
  return __f


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def override_from_dict(self, d) -> None:
        self.update(d)

def default_hparams():
    return EasyDict(
        n_vocab=50257,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
        res_dropout=0.0,
        attn_dropout=0.0,
        dtype=tf.float32
    )

def init_variable(name, shape, **kws):
    v = get_variable(name)
    if v is not None:
        return v
    return tf.get_variable(name, shape, **kws)

def get_variable(name):
    name = os.path.join(tf.get_variable_scope().name, name)
    vs = tf.trainable_variables()
    for x in vs:
        if x.name.startswith(name + ':'):
            return x

@op_scope
def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

@op_scope
def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

@op_scope
def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

@op_scope
def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

@op_scope
def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

@op_scope
def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

@op_scope
def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

@op_scope
def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


@op_scope
def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    @op_scope
    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    @op_scope
    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    @op_scope
    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    @op_scope
    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(shape_list(v)[-1], w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        #breakpoint()
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present


@op_scope
def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2


@op_scope
def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present


# def block(x, scope, *, past, hparams):
#     with tf.variable_scope(scope):
#         nx = shape_list(x)[-1]
#         x_norm = norm(x, 'ln_1')
#         a, present = attn(x_norm, 'attn', nx, past=past, hparams=hparams)
#         x1 = tf.add(x, a)
#         x1_norm = norm(x1, 'ln_2')
#         m = mlp(x1_norm, 'mlp', nx*4, hparams=hparams)
#         x2 = tf.add(x1, m)
#         return x2, present
#
#
# from functools import partial
#
#
# def block(x, scope, *, past, hparams):
#     with tf.variable_scope(scope):
#         nx = shape_list(x)[-1]
#         o0 = tf.identity
#         o1 = partial(norm, scope='ln_1') # x_norm = norm(x, 'ln_1')
#         o2 = partial(attn, scope='attn', n_state=nx, past=past, hparams=hparams) # a, present = attn(x_norm, 'attn', nx, past=past, hparams=hparams)
#         o3 = partial(tf.add, x) # x1 = tf.add(x, a)
#         o4 = partial(norm, scope='ln_2') # x1_norm = norm(x1, 'ln_2')
#         o5 = partial(mlp, scope='mlp', n_state=nx*4, hparams=hparams) # m = mlp(x1_norm, 'mlp', nx*4, hparams=hparams)
#         o6 = partial(tf.add, o3) # x2 = tf.add(x1, m)
#         return o6, present # return x2, present


@op_scope
def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

@op_scope
def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

@op_scope
def positions_for(tokens, past_length):
    # import pdb; pdb.set_trace()
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


@op_scope
def model(hparams, X, past=None, scope='model', reuse=False):
    # import pdb; pdb.set_trace()
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results
