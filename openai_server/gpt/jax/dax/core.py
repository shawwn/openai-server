from __future__ import annotations
from typing import NamedTuple, Callable

class Primitive(NamedTuple):
  name: str

add_p = Primitive('add')
mul_p = Primitive('mul')
neg_p = Primitive("neg")
sin_p = Primitive("sin")
cos_p = Primitive("cos")
reduce_sum_p = Primitive("reduce_sum")
greater_p = Primitive("greater")
less_p = Primitive("less")
transpose_p = Primitive("transpose")
broadcast_p = Primitive("broadcast")

def add(x, y): return bind1(add_p, x, y)
def mul(x, y): return bind1(mul_p, x, y)
def neg(x): return bind1(neg_p, x)
def sin(x): return bind1(sin_p, x)
def cos(x): return bind1(cos_p, x)
def reduce_sum(x, axis=None): return bind1(reduce_sum_p, x, axis=axis)
def greater(x, y): return bind1(greater_p, x, y)
def less(x, y): return bind1(less_p, x, y)
def transpose(x, perm): return bind1(transpose_p, x, perm=perm)
def broadcast(x, shape, axes): return bind1(broadcast_p, x, shape=shape, axes=axes)

def bind1(prim, *args, **params):
  out, = bind(prim, *args, **params)
  return out



from contextlib import contextmanager
from typing import Type, List, Tuple, Sequence, Optional, Any

class MainTrace(NamedTuple):
  level: int
  trace_type: Type['Trace']
  global_data: Optional[Any]

trace_stack: List[MainTrace] = []
dynamic_trace: Optional[MainTrace] = None  # to be employed in Part 3

@contextmanager
def new_main(trace_type: Type['Trace'], global_data=None):
  level = len(trace_stack)
  main = MainTrace(level, trace_type, global_data)
  trace_stack.append(main)

  try:
    yield main
  finally:
    trace_stack.pop()



class Trace:
  main: MainTrace

  def __init__(self, main: MainTrace) -> None:
    self.main = main

  def pure(self, val): raise NotImplementedError()  # must override
  def lift(self, val): raise NotImplementedError()  # must override

  def process_primitive(self, primitive, tracers, params) -> Sequence[Any]:
    raise NotImplementedError()  # must override



import numpy as np

class Tracer:
  _trace: Trace

  __array_priority__ = 1000

  @property
  def aval(self) -> ShapedArray:
    raise NotImplementedError()  # must override

  def full_lower(self):
    return self  # default implementation

  def __neg__(self): return self.aval._neg(self)
  def __add__(self, other): return self.aval._add(self, other)
  def __radd__(self, other): return self.aval._radd(self, other)
  def __mul__(self, other): return self.aval._mul(self, other)
  def __rmul__(self, other): return self.aval._rmul(self, other)
  def __gt__(self, other): return self.aval._gt(self, other)
  def __lt__(self, other): return self.aval._lt(self, other)
  def __bool__(self): return self.aval._bool(self)
  def __nonzero__(self): return self.aval._nonzero(self)

  def __getattr__(self, name):
    try:
      return getattr(self.aval, name)
    except AttributeError:
      raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

def swap(f): return lambda x, y: f(y, x)





class ShapedArray:
  array_abstraction_level = 1
  shape: Tuple[int]
  dtype: np.dtype

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype

  @property
  def ndim(self):
    return len(self.shape)

  _neg = staticmethod(neg)
  _add = staticmethod(add)
  _radd = staticmethod(swap(add))
  _mul = staticmethod(mul)
  _rmul = staticmethod(swap(mul))
  _gt = staticmethod(greater)
  _lt = staticmethod(less)

  @staticmethod
  def _bool(tracer):
    raise Exception("ShapedArray can't be unambiguously converted to bool")

  @staticmethod
  def _nonzero(tracer):
    raise Exception("ShapedArray can't be unambiguously converted to bool")

  def str_short(self):
    return f'{self.dtype.name}[{",".join(str(d) for d in self.shape)}]'

  def __hash__(self):
    return hash((self.shape, self.dtype))

  def __eq__(self, other):
    return (type(self) is type(other) and
            self.shape == other.shape and self.dtype == other.dtype)

  def __repr__(self):
    return f"ShapedArray(shape={self.shape}, dtype={self.dtype})"

class ConcreteArray(ShapedArray):
  array_abstraction_level = 2
  val: np.ndarray

  def __init__(self, val):
    super().__init__(val.shape, val.dtype)
    self.val = val

  @staticmethod
  def _bool(tracer):
    return bool(tracer.aval.val)

  @staticmethod
  def _nonzero(tracer):
    return bool(tracer.aval.val)

def get_aval(x):
  if isinstance(x, Tracer):
    return x.aval
  elif type(x) in jax_types:
    return ConcreteArray(np.asarray(x))
  else:
    raise TypeError(x)

jax_types = {bool, int, float,
             np.bool_,
             np.int8, np.int16, np.int32, np.int64,
             np.uint8, np.uint16, np.uint32, np.uint64,
             np.float16, np.float32, np.float64,
             np.ndarray}



def bind(prim, *args, **params):
  top_trace = find_top_trace(args)
  tracers = [full_raise(top_trace, arg) for arg in args]
  outs = top_trace.process_primitive(prim, tracers, params)
  return [full_lower(out) for out in outs]




import operator as op

def find_top_trace(xs) -> Trace:
  top_main = max((x._trace.main for x in xs if isinstance(x, Tracer)),
                 default=trace_stack[0], key=op.attrgetter('level'))
  if dynamic_trace and dynamic_trace.level > top_main.level:
    top_main = dynamic_trace
  return top_main.trace_type(top_main)




def full_lower(val: Any):
  if isinstance(val, Tracer):
    return val.full_lower()
  else:
    return val

def full_raise(trace: Trace, val: Any) -> Tracer:
  if not isinstance(val, Tracer):
    assert type(val) in jax_types
    return trace.pure(val)
  level = trace.main.level
  if val._trace.main is trace.main:
    return val
  elif val._trace.main.level < level:
    return trace.lift(val)
  elif val._trace.main.level > level:
    raise Exception(f"Can't lift level {val._trace.main.level} to {level}.")
  else:  # val._trace.level == level
    raise Exception(f"Different traces at same level: {val._trace}, {trace}.")



# Evaluation interpreter

class EvalTrace(Trace):
  pure = lift = lambda self, x: x  # no boxing in Tracers needed

  def process_primitive(self, primitive, tracers, params):
    return impl_rules[primitive](*tracers, **params)

trace_stack.append(MainTrace(0, EvalTrace, None))  # special bottom of the stack

# NB: in JAX, instead of a dict we attach impl rules to the Primitive instance
impl_rules = {}

impl_rules[add_p] = lambda x, y: [np.add(x, y)]
impl_rules[mul_p] = lambda x, y: [np.multiply(x, y)]
impl_rules[neg_p] = lambda x: [np.negative(x)]
impl_rules[sin_p] = lambda x: [np.sin(x)]
impl_rules[cos_p] = lambda x: [np.cos(x)]
impl_rules[reduce_sum_p] = lambda x, *, axis: [np.sum(x, axis)]
impl_rules[greater_p] = lambda x, y: [np.greater(x, y)]
impl_rules[less_p] = lambda x, y: [np.less(x, y)]
impl_rules[transpose_p] = lambda x, *, perm: [np.transpose(x, perm)]

def broadcast_impl(x, *, shape, axes):
  for axis in sorted(axes):
    x = np.expand_dims(x, axis)
  return [np.broadcast_to(x, shape)]

impl_rules[broadcast_p] = broadcast_impl


# Pytrees

#from .util import unzip2

def unzip2(pairs):
  lst1, lst2 = [], []
  for x1, x2 in pairs:
    lst1.append(x1)
    lst2.append(x2)
  return lst1, lst2

map_ = map
def map(f, *xs) -> List[Any]:
  return list(map_(f, *xs))

zip_ = zip
def zip(*args) -> List[Any]:
  fst, *rest = args = map(list, args)
  n = len(fst)
  for arg in rest:
    assert len(arg) == n
  return list(zip_(*args))

from jax.tree_util import tree_flatten, tree_unflatten, PyTreeDef

def flatten_fun(f, in_tree):
  store = Store()

  def flat_fun(*args_flat):
    pytree_args = tree_unflatten(in_tree, args_flat)
    out = f(*pytree_args)
    out_flat, out_tree = tree_flatten(out)
    store.set_value(out_tree)
    return out_flat

  return flat_fun, store

class Empty: pass
empty = Empty()

class Store:
  val = empty

  def set_value(self, val):
    assert self.val is empty
    self.val = val

  def __call__(self):
    return self.val


# Vectorized batching with vmap

def mapped_aval(batch_dim, aval):
  shape = list(aval.shape)
  del shape[batch_dim]
  return ShapedArray(tuple(shape), aval.dtype)

def move_batch_axis(axis_size, src, dst, x):
  if src is not_mapped:
    target_shape = list(np.shape(x))
    target_shape.insert(dst, axis_size)
    return broadcast(x, target_shape, [dst])
  elif src == dst:
    return x
  else:
    return moveaxis(x, src, dst)

def moveaxis(x, src: int, dst: int):
  perm = [i for i in range(np.ndim(x)) if i != src]
  perm.insert(dst, src)
  return transpose(x, perm)


from typing import Union

class NotMapped: pass
not_mapped = NotMapped()

BatchAxis = Union[NotMapped, int]

class BatchTracer(Tracer):
  def __init__(self, trace, val, batch_dim: BatchAxis):
    self._trace = trace
    self.val = val
    self.batch_dim = batch_dim

  @property
  def aval(self):
    if self.batch_dim is not_mapped:
      return get_aval(self.val)
    else:
      return mapped_aval(self.batch_dim, get_aval(self.val))

  def full_lower(self):
    if self.batch_dim is not_mapped:
      return full_lower(self.val)
    else:
      return self

class BatchTrace(Trace):
  pure = lift = lambda self, val: BatchTracer(self, val, not_mapped)

  def process_primitive(self, primitive, tracers, params):
    vals_in, bdims_in = unzip2((t.val, t.batch_dim) for t in tracers)
    vmap_rule = vmap_rules[primitive]
    val_outs, bdim_outs = vmap_rule(self.axis_size, vals_in, bdims_in, **params)
    return [BatchTracer(self, x, bd) for x, bd in zip(val_outs, bdim_outs)]

  @property
  def axis_size(self):
    return self.main.global_data

vmap_rules = {}


from functools import partial

def binop_batching_rule(op, axis_size, vals_in, dims_in):
  (x, y), (x_bdim, y_bdim) = vals_in, dims_in
  if x_bdim != y_bdim:
    if x_bdim is not_mapped:
      x = move_batch_axis(axis_size, x_bdim, y_bdim, x)
      x_bdim = y_bdim
    else:
      y = move_batch_axis(axis_size, y_bdim, x_bdim, y)
  return [op(x, y)], [x_bdim]
vmap_rules[add_p] = partial(binop_batching_rule, add)
vmap_rules[mul_p] = partial(binop_batching_rule, mul)

def vectorized_unop_batching_rule(op, axis_size, vals_in, dims_in):
  (x,), (x_bdim,) = vals_in, dims_in
  return [op(x)], [x_bdim]
vmap_rules[sin_p] = partial(vectorized_unop_batching_rule, sin)
vmap_rules[cos_p] = partial(vectorized_unop_batching_rule, cos)
vmap_rules[neg_p] = partial(vectorized_unop_batching_rule, neg)

def reduce_sum_batching_rule(axis_size, vals_in, dims_in, *, axis):
  (x,), (x_bdim,) = vals_in, dims_in
  new_axis = axis + (x_bdim <= axis)
  out_bdim = x_bdim - (new_axis < x_bdim)
  return [reduce_sum(x, new_axis)], [out_bdim]
vmap_rules[reduce_sum_p] = reduce_sum_batching_rule


def vmap_flat(f, in_axes, *args):
  axis_size, = {x.shape[ax] for x, ax in zip(args, in_axes)
                if ax is not not_mapped}
  with new_main(BatchTrace, axis_size) as main:
    trace = BatchTrace(main)
    tracers_in = [BatchTracer(trace, x, ax) if ax is not None else x
                  for x, ax in zip(args, in_axes)]
    outs = f(*tracers_in)
    tracers_out = [full_raise(trace, out) for out in outs]
    vals_out, bdims_out = unzip2((t.val, t.batch_dim) for t in tracers_out)
  outs_transposed = [move_batch_axis(axis_size, bdim, 0, val_out)
                     for val_out, bdim in zip(vals_out, bdims_out)]
  return outs_transposed

def vmap(f, in_axes):
  def batched_f(*args):
    args_flat, in_tree = tree_flatten(args)
    in_axes_flat, in_tree2 = tree_flatten(in_axes)
    if in_tree != in_tree2: raise TypeError
    f_flat, out_tree = flatten_fun(f, in_tree)
    outs_flat = vmap_flat(f_flat, in_axes_flat, *args_flat)
    return tree_unflatten(out_tree(), outs_flat)
  return batched_f


# Part 2: Jaxprs

from typing import Set, Dict

class Var:
  aval: ShapedArray
  def __init__(self, aval): self.aval = aval

class Lit:
  val: Any
  aval: ShapedArray

  def __init__(self, val):
    self.aval = aval = raise_to_shaped(get_aval(val))
    self.val = np.array(val, aval.dtype)

Atom = Union[Var, Lit]

class JaxprEqn(NamedTuple):
  primitive: Primitive
  inputs: List[Atom]
  params: Dict[str, Any]
  out_binders: List[Var]

class Jaxpr(NamedTuple):
  in_binders: List[Var]
  eqns: List[JaxprEqn]
  outs: List[Atom]

  def __hash__(self): return id(self)
  __eq__ = op.is_

def raise_to_shaped(aval):
  return ShapedArray(aval.shape, aval.dtype)




class JaxprType(NamedTuple):
  in_types:  List[ShapedArray]
  out_types: List[ShapedArray]

  def __repr__(self):
    in_types = ', '.join(aval.str_short() for aval in self.in_types)
    out_types = ', '.join(aval.str_short() for aval in self.out_types)
    return f'({in_types}) -> ({out_types})'

def typecheck_jaxpr(jaxpr: Jaxpr) -> JaxprType:
  env: Set[Var] = set()

  for v in jaxpr.in_binders:
    if v in env: raise TypeError
    env.add(v)

  for eqn in jaxpr.eqns:
    in_types = [typecheck_atom(env, x) for x in eqn.inputs]
    out_types = abstract_eval_rules[eqn.primitive](*in_types, **eqn.params)
    for out_binder, out_type in zip(eqn.out_binders, out_types):
      if not out_type == out_binder.aval: raise TypeError
    for out_binder in eqn.out_binders:
      if out_binder in env: raise TypeError
      env.add(out_binder)

  in_types = [v.aval for v in jaxpr.in_binders]
  out_types = [typecheck_atom(env, x) for x in jaxpr.outs]
  return JaxprType(in_types, out_types)

def typecheck_atom(env: Set[Var], x: Atom) -> ShapedArray:
  if isinstance(x, Var):
    if x not in env: raise TypeError("unbound variable")
    return x.aval
  elif isinstance(x, Lit):
    return raise_to_shaped(get_aval(x.val))
  else:
    assert False




def eval_jaxpr(jaxpr: Jaxpr, args: Sequence[Any]) -> List[Any]:
  env: Dict[Var, Any] = {}

  def read(x: Atom) -> Any:
    return env[x] if type(x) is Var else x.val

  def write(v: Var, val: Any) -> None:
    assert v not in env  # single-assignment
    env[v] = val

  map(write, jaxpr.in_binders, args)
  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.inputs)
    outs = bind(eqn.primitive, *in_vals, **eqn.params)
    map(write, eqn.out_binders, outs)
  return map(read, jaxpr.outs)

def jaxpr_as_fun(jaxpr: Jaxpr):
  return lambda *args: eval_jaxpr(jaxpr, args)


# Building jaxprs with tracing

#from .util import (split_list, partition_list)
from .util import partition_list
from . import util

def split_list(lst: List[Any], n: Union[int, List[int]]) -> Tuple[List[Any], List[Any]]:
  if isinstance(n, tuple):
    n = list(n)
  if not isinstance(n, list):
    n = [n]
  return util.split_list(lst, n)

# def split_list(lst: List[Any], n: int) -> Tuple[List[Any], List[Any]]:
#   assert 0 <= n <= len(lst)
#   return lst[:n], lst[n:]
#
# def partition_list(bs: List[bool], l: List[Any]) -> Tuple[List[Any], List[Any]]:
#   assert len(bs) == len(l)
#   lists = lst1, lst2 = [], []
#   for b, x in zip(bs, l):
#     lists[b].append(x)
#   return lst1, lst2


# NB: the analogous class in JAX is called 'DynamicJaxprTracer'
class JaxprTracer(Tracer):
  __slots__ = ['aval']
  aval: ShapedArray

  def __init__(self, trace, aval):
    self._trace = trace
    self.aval = aval

# NB: the analogous class in JAX is called 'DynamicJaxprTrace'
class JaxprTrace(Trace):
  def new_arg(self, aval: ShapedArray) -> JaxprTracer:
    aval = raise_to_shaped(aval)
    tracer = self.builder.new_tracer(self, aval)
    self.builder.tracer_to_var[id(tracer)] = Var(aval)
    return tracer

  def get_or_make_const_tracer(self, val: Any) -> JaxprTracer:
    tracer = self.builder.const_tracers.get(id(val))
    if tracer is None:
      tracer = self.builder.new_tracer(self, raise_to_shaped(get_aval(val)))
      self.builder.add_const(tracer, val)
    return tracer
  pure = lift = get_or_make_const_tracer

  def process_primitive(self, primitive, tracers, params):
    avals_in = [t.aval for t in tracers]
    avals_out = abstract_eval_rules[primitive](*avals_in, **params)
    out_tracers = [self.builder.new_tracer(self, a) for a in avals_out]
    inputs = [self.builder.getvar(t) for t in tracers]
    outvars = [self.builder.add_var(t) for t in out_tracers]
    self.builder.add_eqn(JaxprEqn(primitive, inputs, params, outvars))
    return out_tracers

  @property
  def builder(self):
    return self.main.global_data

# NB: in JAX, we instead attach abstract eval rules to Primitive instances
abstract_eval_rules = {}


class JaxprBuilder:
  eqns: List[JaxprEqn]
  tracer_to_var: Dict[int, Var]
  const_tracers: Dict[int, JaxprTracer]
  constvals: Dict[Var, Any]
  tracers: List[JaxprTracer]

  def __init__(self):
    self.eqns = []
    self.tracer_to_var = {}
    self.const_tracers = {}
    self.constvals = {}
    self.tracers = []

  def new_tracer(self, trace: JaxprTrace, aval: ShapedArray) -> JaxprTracer:
    tracer = JaxprTracer(trace, aval)
    self.tracers.append(tracer)
    return tracer

  def add_eqn(self, eqn: JaxprEqn) -> None:
    self.eqns.append(eqn)

  def add_var(self, tracer: JaxprTracer) -> Var:
    assert id(tracer) not in self.tracer_to_var
    var = self.tracer_to_var[id(tracer)] = Var(tracer.aval)
    return var

  def getvar(self, tracer: JaxprTracer) -> Var:
    var = self.tracer_to_var.get(id(tracer))
    assert var is not None
    return var

  def add_const(self, tracer: JaxprTracer, val: Any) -> Var:
    var = self.add_var(tracer)
    self.const_tracers[id(val)] = tracer
    self.constvals[var] = val
    return var

  def build(self, in_tracers: List[Tracer], out_tracers: List[Tracer]
            ) -> Tuple[Jaxpr, List[Any]]:
    constvars, constvals = unzip2(self.constvals.items())
    t2v = lambda t: self.tracer_to_var[id(t)]
    in_binders = constvars + [t2v(t) for t in in_tracers]
    out_vars = [t2v(t) for t in out_tracers]
    jaxpr = Jaxpr(in_binders, self.eqns, out_vars)
    typecheck_jaxpr(jaxpr)
    jaxpr, constvals = _inline_literals(jaxpr, constvals)
    return jaxpr, constvals



def _inline_literals(jaxpr: Jaxpr, consts: List[Any]) -> Tuple[Jaxpr, List[Any]]:
  const_binders, other_binders = split_list(jaxpr.in_binders, len(consts))
  scalars = [type(x) in jax_types and not get_aval(x).shape for x in consts]
  new_const_binders, lit_binders = partition_list(scalars, const_binders)
  new_consts, lit_vals = partition_list(scalars, consts)
  literals = dict(zip(lit_binders, map(Lit, lit_vals)))
  new_eqns = [JaxprEqn(eqn.primitive, [literals.get(x, x) for x in eqn.inputs],
                       eqn.params, eqn.out_binders) for eqn in jaxpr.eqns]
  new_outs = [literals.get(x, x) for x in jaxpr.outs]
  new_jaxpr = Jaxpr(new_const_binders + other_binders, new_eqns, new_outs)
  typecheck_jaxpr(new_jaxpr)
  return new_jaxpr, new_consts





def binop_abstract_eval(x: ShapedArray, y: ShapedArray) -> List[ShapedArray]:
  if not isinstance(x, ShapedArray) or not isinstance(y, ShapedArray):
    raise TypeError
  if raise_to_shaped(x) != raise_to_shaped(y): raise TypeError
  return [ShapedArray(x.shape, x.dtype)]

abstract_eval_rules[add_p] = binop_abstract_eval
abstract_eval_rules[mul_p] = binop_abstract_eval

def compare_abstract_eval(x: ShapedArray, y: ShapedArray) -> List[ShapedArray]:
  if not isinstance(x, ShapedArray) or not isinstance(y, ShapedArray):
    raise TypeError
  if x.shape != y.shape: raise TypeError
  return [ShapedArray(x.shape, np.dtype('bool'))]
abstract_eval_rules[greater_p] = compare_abstract_eval
abstract_eval_rules[less_p] = compare_abstract_eval

def vectorized_unop_abstract_eval(x: ShapedArray) -> List[ShapedArray]:
  return [ShapedArray(x.shape, x.dtype)]

abstract_eval_rules[sin_p] = vectorized_unop_abstract_eval
abstract_eval_rules[cos_p] = vectorized_unop_abstract_eval
abstract_eval_rules[neg_p] = vectorized_unop_abstract_eval

def reduce_sum_abstract_eval(x: ShapedArray, *, axis: int) -> List[ShapedArray]:
  new_shape = [d for i, d in enumerate(x.shape) if i != axis]
  return [ShapedArray(tuple(new_shape), x.dtype)]
abstract_eval_rules[reduce_sum_p] = reduce_sum_abstract_eval

def broadcast_abstract_eval(x: ShapedArray, *, shape: Sequence[int],
                            axes: Sequence[int]) -> List[ShapedArray]:
  return [ShapedArray(tuple(shape), x.dtype)]
abstract_eval_rules[broadcast_p] = broadcast_abstract_eval




from functools import lru_cache

# @lru_cache()  # ShapedArrays are hashable
def make_jaxpr_v1(f, *avals_in):
  avals_in, in_tree = tree_flatten(avals_in)
  f, out_tree = flatten_fun(f, in_tree)

  builder = JaxprBuilder()
  with new_main(JaxprTrace, builder) as main:
    trace = JaxprTrace(main)
    tracers_in = [trace.new_arg(aval) for aval in avals_in]
    outs = f(*tracers_in)
    tracers_out = [full_raise(trace, out) for out in outs]
    jaxpr, consts = builder.build(tracers_in, tracers_out)
  return jaxpr, consts, out_tree()





@contextmanager
def new_dynamic(main: MainTrace):
  global dynamic_trace
  prev_dynamic_trace, dynamic_trace = dynamic_trace, main
  try:
    yield
  finally:
    dynamic_trace = prev_dynamic_trace

@lru_cache()
def make_jaxpr(f: Callable, *avals_in: ShapedArray,
               ) -> Tuple[Jaxpr, List[Any], PyTreeDef]:
  avals_in, in_tree = tree_flatten(avals_in)
  f, out_tree = flatten_fun(f, in_tree)

  builder = JaxprBuilder()
  with new_main(JaxprTrace, builder) as main:
    with new_dynamic(main):
      trace = JaxprTrace(main)
      tracers_in = [trace.new_arg(aval) for aval in avals_in]
      outs = f(*tracers_in)
      tracers_out = [full_raise(trace, out) for out in outs]
      jaxpr, consts = builder.build(tracers_in, tracers_out)
  return jaxpr, consts, out_tree()



# Part 3: jit

def jit(f):
  def f_jitted(*args):
    avals_in = [raise_to_shaped(get_aval(x)) for x in args]
    jaxpr, consts, out_tree = make_jaxpr(f, *avals_in)
    outs = bind(xla_call_p, *consts, *args, jaxpr=jaxpr, num_consts=len(consts))
    return tree_unflatten(out_tree, outs)
  return f_jitted

xla_call_p = Primitive('xla_call')


class IDHashable:
  val: Any

  def __init__(self, val):
    self.val = val

  def __hash__(self) -> int:
    return id(self.val)

  def __eq__(self, other):
    return type(other) is IDHashable and id(self.val) == id(other.val)



from ._src.lib import xla_bridge as xb
#from jax._src.lib import xla_client as xc
from ._src.lib import xla_client as xc
xe = xc._xla
xops = xc._xla.ops

def xla_call_impl(*args, jaxpr: Jaxpr, num_consts: int):
  consts, args = args[:num_consts], args[num_consts:]
  hashable_consts: Tuple[IDHashable, ...] = tuple(map(IDHashable, consts))
  execute = xla_callable(IDHashable(jaxpr), hashable_consts)
  return execute(*args)
impl_rules[xla_call_p] = xla_call_impl

def get_backend(platform='dax'):
  return xb.get_backend(platform)

@lru_cache()
def xla_callable(hashable_jaxpr: IDHashable, hashable_consts: Tuple[IDHashable]):
  jaxpr: Jaxpr = hashable_jaxpr.val
  typecheck_jaxpr(jaxpr)
  consts = [x.val for x in hashable_consts]
  in_avals = [v.aval for v in jaxpr.in_binders[len(consts):]]
  c = xc.XlaBuilder('xla_call')
  xla_consts = _xla_consts(c, consts)
  xla_params = _xla_params(c, in_avals)
  outs = jaxpr_subcomp(c, jaxpr, xla_consts + xla_params)
  out = xops.Tuple(c, outs)
  compiled = get_backend().compile(c.build(out))
  return partial(execute_compiled, compiled, [v.aval for v in jaxpr.outs])

def _xla_consts(c: xe.XlaBuilder, consts: List[Any]) -> List[xe.XlaOp]:
  unique_consts = {id(cnst): cnst for cnst in consts}
  xla_consts = {
      id_: xops.ConstantLiteral(c, cnst) for id_, cnst in unique_consts.items()}
  return [xla_consts[id(cnst)] for cnst in consts]

def _xla_params(c: xe.XlaBuilder, avals_in: List[ShapedArray]) -> List[xe.XlaOp]:
  return [xb.parameter(c, i, _xla_shape(a)) for i, a in enumerate(avals_in)]

def _xla_shape(aval: ShapedArray) -> xe.Shape:
  return xc.Shape.array_shape(xc.dtype_to_etype(aval.dtype), aval.shape)



def jaxpr_subcomp(c: xe.XlaBuilder, jaxpr: Jaxpr, args: List[xe.XlaOp]
                  ) -> List[xe.XlaOp]:
  env: Dict[Var, xe.XlaOp] = {}

  def read(x: Atom) -> xe.XlaOp:
    return env[x] if type(x) is Var else xops.Constant(c, np.asarray(x.val))

  def write(v: Var, val: xe.XlaOp) -> None:
    env[v] = val

  map(write, jaxpr.in_binders, args)
  for eqn in jaxpr.eqns:
    in_avals = [x.aval for x in eqn.inputs]
    in_vals = map(read, eqn.inputs)
    rule = xla_translations[eqn.primitive]
    out_vals = rule(c, in_avals, in_vals, **eqn.params)
    map(write, eqn.out_binders, out_vals)
  return map(read, jaxpr.outs)

def execute_compiled(compiled, out_avals, *args):
  input_bufs = [input_handlers[type(x)](x) for x in args]
  out_bufs = compiled.execute(input_bufs)
  return [handle_result(aval, buf) for aval, buf in zip(out_avals, out_bufs)]

default_input_handler = get_backend().buffer_from_pyval
input_handlers = {ty: default_input_handler for ty in
                  [bool, int, float, np.ndarray, np.float64, np.float32]}

def handle_result(aval: ShapedArray, buf):
  del aval  # Unused for now
  return buf.to_py()

xla_translations = {}



def direct_translation(op, c, in_avals, in_vals):
  del c, in_avals
  return [op(*in_vals)]

xla_translations[add_p] = partial(direct_translation, xops.Add)
xla_translations[mul_p] = partial(direct_translation, xops.Mul)
xla_translations[neg_p] = partial(direct_translation, xops.Neg)
xla_translations[sin_p] = partial(direct_translation, xops.Sin)
xla_translations[cos_p] = partial(direct_translation, xops.Cos)
xla_translations[greater_p] = partial(direct_translation, xops.Gt)
xla_translations[less_p] = partial(direct_translation, xops.Lt)

def reduce_sum_translation(c, in_avals, in_vals, *, axis):
  (x_aval,), (x,) = in_avals, in_vals
  zero = xops.ConstantLiteral(c, np.array(0, x_aval.dtype))
  subc = xc.XlaBuilder('add')
  shape = _xla_shape(ShapedArray((), x_aval.dtype))
  xops.Add(xops.Parameter(subc, 0, shape), xops.Parameter(subc, 1, shape))
  return [xops.Reduce(c, [x], [zero], subc.build(), [axis])]
xla_translations[reduce_sum_p] = reduce_sum_translation

def broadcast_translation(c, in_avals, in_vals, *, shape, axes):
  x, = in_vals
  dims_complement = [i for i in range(len(shape)) if i not in axes]
  return [xops.BroadcastInDim(x, shape, dims_complement)]
xla_translations[broadcast_p] = broadcast_translation



if 'jvp_rules' in globals():

  def xla_call_jvp_rule(primals, tangents, *, jaxpr, num_consts):
    del num_consts  # Unused
    new_jaxpr, new_consts = jvp_jaxpr(jaxpr)
    outs = bind(xla_call_p, *new_consts, *primals, *tangents, jaxpr=new_jaxpr,
                num_consts=len(new_consts))
    n = len(outs) // 2
    primals_out, tangents_out = outs[:n], outs[n:]
    return primals_out, tangents_out
  jvp_rules[xla_call_p] = xla_call_jvp_rule

  @lru_cache()
  def jvp_jaxpr(jaxpr: Jaxpr) -> Tuple[Jaxpr, List[Any]]:
    def jvp_traceable(*primals_and_tangents):
      n = len(primals_and_tangents) // 2
      primals, tangents = primals_and_tangents[:n], primals_and_tangents[n:]
      return jvp(jaxpr_as_fun(jaxpr), primals, tangents)

    in_avals = [v.aval for v in jaxpr.in_binders]
    new_jaxpr, new_consts, _ = make_jaxpr(jvp_traceable, *in_avals, *in_avals)
    return new_jaxpr, new_consts

def xla_call_vmap_rule(axis_size, vals_in, dims_in, *, jaxpr, num_consts):
  del num_consts  # Unused
  new_jaxpr, new_consts = vmap_jaxpr(jaxpr, axis_size, tuple(dims_in))
  outs = bind(xla_call_p, *new_consts, *vals_in, jaxpr=new_jaxpr,
              num_consts=len(new_consts))
  return outs, [0] * len(outs)
vmap_rules[xla_call_p] = xla_call_vmap_rule

@lru_cache()
def vmap_jaxpr(jaxpr: Jaxpr, axis_size: int, bdims_in: Tuple[BatchAxis, ...]
               ) -> Tuple[Jaxpr, List[Any]]:
  vmap_traceable = vmap(jaxpr_as_fun(jaxpr), tuple(bdims_in))
  in_avals = [unmapped_aval(axis_size, d, v.aval)
              for v, d in zip(jaxpr.in_binders, bdims_in)]
  new_jaxpr, new_consts, _ = make_jaxpr(vmap_traceable, *in_avals)
  return new_jaxpr, new_consts

def unmapped_aval(axis_size: int, batch_dim: BatchAxis, aval: ShapedArray
                  ) -> ShapedArray:
  if batch_dim is not_mapped:
    return aval
  else:
    shape = list(aval.shape)
    shape.insert(batch_dim, axis_size)
    return ShapedArray(tuple(shape), aval.dtype)
def xla_call_abstract_eval_rule(*in_types, jaxpr, num_consts):
  del num_consts  # Unused
  jaxpr_type = typecheck_jaxpr(jaxpr)
  if not all(t1 == t2 for t1, t2 in zip(jaxpr_type.in_types, in_types)):
    raise TypeError
  return jaxpr_type.out_types
abstract_eval_rules[xla_call_p] = xla_call_abstract_eval_rule

def xla_call_translation(c, in_avals, in_vals, *, jaxpr, num_consts):
  del num_consts  # Only used at top-level.
  # Calling jaxpr_subcomp directly would inline. We generate a Call HLO instead.
  subc = xc.XlaBuilder('inner xla_call')
  xla_params = _xla_params(subc, in_avals)
  outs = jaxpr_subcomp(subc, jaxpr, xla_params)
  subc = subc.build(xops.Tuple(subc, outs))
  return destructure_tuple(c, xops.Call(c, subc, in_vals))
xla_translations[xla_call_p] = xla_call_translation

def destructure_tuple(c, tup):
  num_elements = len(c.get_shape(tup).tuple_shapes())
  return [xops.GetTupleElement(tup, i) for i in range(num_elements)]



# DeviceArrays

def handle_result(aval: ShapedArray, buf):  # noqa: F811
  return DeviceArray(aval, buf)

class DeviceArray:
  buf: Any
  aval: ShapedArray

  def __init__(self, aval, buf):
    self.aval = aval
    self.buf = buf

  dtype = property(lambda self: self.aval.dtype)
  shape = property(lambda self: self.aval.shape)
  ndim  = property(lambda self: self.aval.ndim)

  def __array__(self): return self.buf.to_py()
  def __repr__(self):  return repr(self.buf.to_py())
  def __str__(self):   return str(self.buf.to_py())

  _neg = staticmethod(neg)
  _add = staticmethod(add)
  _radd = staticmethod(add)
  _mul = staticmethod(mul)
  _rmul = staticmethod(mul)
  _gt = staticmethod(greater)
  _lt = staticmethod(less)
input_handlers[DeviceArray] = lambda x: x.buf

jax_types.add(DeviceArray)


# Part 5: control flow primitives

def cond(pred, true_fn, false_fn, *operands):
  avals_in = [raise_to_shaped(get_aval(x)) for x in operands]
  true_jaxpr, true_consts, out_tree = make_jaxpr(true_fn, *avals_in)
  false_jaxpr, false_consts, out_tree_ = make_jaxpr(false_fn, *avals_in)
  if out_tree != out_tree_: raise TypeError
  true_jaxpr, false_jaxpr = _join_jaxpr_consts(
      true_jaxpr, false_jaxpr, len(true_consts), len(false_consts))
  if typecheck_jaxpr(true_jaxpr) != typecheck_jaxpr(false_jaxpr):
    raise TypeError
  outs = bind_cond(pred, *true_consts, *false_consts, *operands,
                   true_jaxpr=true_jaxpr, false_jaxpr=false_jaxpr)
  return tree_unflatten(out_tree, outs)
cond_p = Primitive('cond')

def _join_jaxpr_consts(jaxpr1: Jaxpr, jaxpr2: Jaxpr, n1: int, n2: int
                       ) -> Tuple[Jaxpr, Jaxpr]:
  jaxpr1_type, jaxpr2_type = typecheck_jaxpr(jaxpr1), typecheck_jaxpr(jaxpr2)
  assert jaxpr1_type.in_types[n1:] == jaxpr2_type.in_types[n2:]
  consts1, rest1 = split_list(jaxpr1.in_binders, n1)
  consts2, rest2 = split_list(jaxpr2.in_binders, n2)
  new_jaxpr1 = Jaxpr(consts1 + consts2 + rest1, jaxpr1.eqns, jaxpr1.outs)
  new_jaxpr2 = Jaxpr(consts1 + consts2 + rest2, jaxpr2.eqns, jaxpr2.outs)
  return new_jaxpr1, new_jaxpr2

def bind_cond(pred, *args, true_jaxpr, false_jaxpr):
  assert len(args) == len(true_jaxpr.in_binders) == len(false_jaxpr.in_binders)
  return bind(cond_p, pred, *args, true_jaxpr=true_jaxpr, false_jaxpr=false_jaxpr)


def cond_impl(pred, *operands, true_jaxpr, false_jaxpr):
  if pred:
    return eval_jaxpr(true_jaxpr, operands)
  else:
    return eval_jaxpr(false_jaxpr, operands)
impl_rules[cond_p] = cond_impl



if 'jvp_rules' in globals():
  def cond_jvp_rule(primals, tangents, *, true_jaxpr, false_jaxpr):
    pred, *primals = primals
    _   , *tangents = tangents
    true_jaxpr , true_consts  = jvp_jaxpr(true_jaxpr)
    false_jaxpr, false_consts = jvp_jaxpr(false_jaxpr)
    true_jaxpr, false_jaxpr = _join_jaxpr_consts(
        true_jaxpr, false_jaxpr, len(true_consts), len(false_consts))
    assert typecheck_jaxpr(true_jaxpr) == typecheck_jaxpr(false_jaxpr)
    outs = bind_cond(pred, *true_consts, *false_consts, *primals, *tangents,
                     true_jaxpr=true_jaxpr, false_jaxpr=false_jaxpr)
    primals_out, tangents_out = split_half(outs)
    return primals_out, tangents_out
  jvp_rules[cond_p] = cond_jvp_rule


def cond_vmap_rule(axis_size, vals_in, dims_in, *, true_jaxpr, false_jaxpr):
  pred    , *vals_in = vals_in
  pred_dim, *dims_in = dims_in
  if pred_dim is not not_mapped: raise NotImplementedError  # TODO
  true_jaxpr, true_consts = vmap_jaxpr(true_jaxpr, axis_size, tuple(dims_in))
  false_jaxpr, false_consts = vmap_jaxpr(false_jaxpr, axis_size, tuple(dims_in))
  true_jaxpr, false_jaxpr = _join_jaxpr_consts(
      true_jaxpr, false_jaxpr, len(true_consts), len(false_consts))
  assert typecheck_jaxpr(true_jaxpr) == typecheck_jaxpr(false_jaxpr)
  outs = bind_cond(pred, *true_consts, *false_consts, *vals_in,
                   true_jaxpr=true_jaxpr, false_jaxpr=false_jaxpr)
  return outs, [0] * len(outs)
vmap_rules[cond_p] = cond_vmap_rule



def cond_abstract_eval(pred_type, *in_types, true_jaxpr, false_jaxpr):
  if pred_type != ShapedArray((), np.dtype('bool')): raise TypeError
  jaxpr_type = typecheck_jaxpr(true_jaxpr)
  if jaxpr_type != typecheck_jaxpr(false_jaxpr):
    raise TypeError
  if not all(t1 == t2 for t1, t2 in zip(jaxpr_type.in_types, in_types)):
    raise TypeError
  return jaxpr_type.out_types
abstract_eval_rules[cond_p] = cond_abstract_eval

def cond_translation(c, in_avals, in_vals, *, true_jaxpr, false_jaxpr):
  del in_avals  # Unused
  pred, *in_vals = in_vals
  flat_vals, in_tree = tree_flatten(in_vals)
  operand = xops.Tuple(c, flat_vals)
  operand_shape = c.get_shape(operand)

  def make_comp(name: str, jaxpr: Jaxpr) -> xe.XlaComputation:
    c = xc.XlaBuilder(name)
    operand = xb.parameter(c, 0, operand_shape)
    operands = tree_unflatten(in_tree, destructure_tuple(c, operand))
    outs = jaxpr_subcomp(c, jaxpr, operands)
    return c.build(xops.Tuple(c, outs))

  true_comp = make_comp('true_fn', true_jaxpr)
  false_comp = make_comp('false_fn', false_jaxpr)

  int_etype = xc.dtype_to_etype(np.dtype('int32'))
  out = xops.Conditional(xops.ConvertElementType(pred, int_etype),
                         [false_comp, true_comp], [operand] * 2)
  return destructure_tuple(c, out)
xla_translations[cond_p] = cond_translation
