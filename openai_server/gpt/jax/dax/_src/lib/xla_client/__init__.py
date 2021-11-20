from jax._src.lib import xla_client as _xc
import inspect as _inspect

for k, v in _inspect.getmembers(_xc):
  if k.startswith('__'):
    continue
  if k == '_xla':
    continue
  globals()[k] = v

del _xc
del _inspect

from . import _xla
from ._xla.xla_builder import (XlaOp, XlaBuilder, XlaComputation, Shape, dtype_to_etype, _Op)
