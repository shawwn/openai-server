from . import ops
from jax._src.lib import xla_client as _xc
import inspect as _inspect
for k, v in _inspect.getmembers(_xc._xla):
  if k.startswith('__'):
    continue
  if k == 'ops':
    continue
  globals()[k] = v

from .xla_builder import (XlaOp, XlaBuilder, XlaComputation, Shape, dtype_to_etype, _Op)
