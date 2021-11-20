# from jax._src.lib import xla_client as _xc
# from ... import xla_client as _dxc

# class XlaOp(_xc._xla.XlaOp):
#   def __init__(self, *args, **kws):
#     super().__init__(*args, **kws)

# if _xc._xla.XlaOp.__bases__[0].__module__ == 'pybind11_builtins':
#   _pybind11_object = _xc._xla.XlaOp.__bases__[0]
#   _OldXlaOp = _xc._xla.XlaOp
#   # class _XlaOp(_pybind11_object):
#   #   pass
#   # class XlaOp(_XlaOp):
#   #   def __init__(self, *args, **kws):
#   #     super().__init__(*args, **kws)
#   # _xc._xla.XlaOp.__bases__ = (XlaOp,)
#   class _XlaOp:
#     pass
#   class XlaOpExt:
#     def __init__(self, *args, **kws):
#       # super().__init__(*args, **kws)
#       pass
#
#   XlaOp = type('XlaOp', (_OldXlaOp, XlaOpExt, object), {})

# if _xc._xla.XlaOp.__bases__[0].__module__ == 'pybind11_builtins':
#   _pybind11_object = _xc._xla.XlaOp.__bases__[0]
#   _OldXlaOp = _xc._xla.XlaOp
#   class XlaOp(_OldXlaOp):
#     def __init__(self, *args, **kws):
#       super().__init__(*args, **kws)
#   _xc._xla.XlaOp = XlaOp

