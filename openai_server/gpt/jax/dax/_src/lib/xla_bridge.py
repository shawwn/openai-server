from jax._src.lib.xla_bridge import *

from . import xla_client as _xc

xops = _xc._xla.ops


def parameter(builder, num, shape, name=None, replicated=None):
  if name is None:
    name = ''
  if replicated is None:
    replicated = []
  elif isinstance(replicated, bool):
    replicated = [replicated] * shape.leaf_count()
  shape2 = shape.with_major_to_minor_layout_if_absent()
  param = xops.Parameter(builder, num, shape2, name, replicated)
  return param


class DaxBackend:
  def __init__(self, *args, **kws):
    self._local_devices = [DaxDevice(0)]

  def device_count(self):
    return len(self._local_devices)

  @property
  def platform(self):
    return 'dax'

  @property
  def platform_version(self):
    return '<unknown>'

  def process_index(self):
    return 0

  def devices(self):
    return list(self._local_devices)

  def local_devices(self):
    return list(self._local_devices)

  def buffer_from_pyval(self, val):
    # return jnp.array(val)
    return _cpu().buffer_from_pyval(val)

  def get_default_device_assignment(self, arg0: int, arg1: int = None):
    assert arg0 == 1
    if arg1 is None:
      return self.local_devices()
    else:
      return [self.local_devices()]

  def compile(self, built_c, compile_options=None):
    # print('compile', built_c, compile_options)
    if compile_options:
      result = _cpu().compile(built_c, compile_options)
    else:
      result = _cpu().compile(built_c)
    # print(result)
    return result


class DaxDevice:
  def __init__(self, index):
    self.id = index
    self.host_id = index
    self.device_kind = 'dax'
    self.platform = 'dax'
    self.process_index = 0


register_backend_factory('dax', lambda *args, **kws: (print(args, kws) or DaxBackend(*args, **kws)), priority=500)

__cpu = None


def _cpu():
  global __cpu
  if __cpu is None:
    __cpu = get_backend('cpu')
  return __cpu
