from . import xla_data_pb2 as _xd

import numpy as np


def dtype_of(type):
  return np.dtype(type)


def PrimitiveTypeToDtype(type: _xd.PrimitiveType):
  if type == _xd.PRED:
    return dtype_of(np.bool_)
  elif type == _xd.S8:
    return dtype_of(np.int8)
  elif type == _xd.S16:
    return dtype_of(np.int16)
  elif type == _xd.S32:
    return dtype_of(np.int32)
  elif type == _xd.S64:
    return dtype_of(np.int64)
  elif type == _xd.U8:
    return dtype_of(np.uint8)
  elif type == _xd.U16:
    return dtype_of(np.uint16)
  elif type == _xd.U32:
    return dtype_of(np.uint32)
  elif type == _xd.U64:
    return dtype_of(np.uint64)
  elif type == _xd.BF16:
    import jax.numpy as jnp
    return dtype_of(jnp.bfloat16)
  elif type == _xd.F16:
    return dtype_of(np.dtype("e"))  # PEP 3118 code for float16
  elif type == _xd.F32:
    return dtype_of(np.float32)
  elif type == _xd.F64:
    return dtype_of(np.float64)
  elif type == _xd.C64:
    return dtype_of(np.complex64)
  elif type == _xd.C128:
    return dtype_of(np.complex128)
  else:
    raise NotImplementedError(f'Unimplemented primitive type {_xd.PrimitiveType.Name(type)}')

# xla::StatusOr<py::dtype> PrimitiveTypeToDtype(PrimitiveType type) {
#   switch (type) {
#     case PRED:
#       return py::dtype::of<bool>();
#     case S8:
#       return py::dtype::of<int8>();
#     case S16:
#       return py::dtype::of<int16>();
#     case S32:
#       return py::dtype::of<int32>();
#     case S64:
#       return py::dtype::of<int64>();
#     case U8:
#       return py::dtype::of<uint8>();
#     case U16:
#       return py::dtype::of<uint16>();
#     case U32:
#       return py::dtype::of<uint32>();
#     case U64:
#       return py::dtype::of<uint64>();
#     case BF16: {
#       py::handle bfloat16(tensorflow::Bfloat16Dtype());
#       return py::dtype::from_args(py::reinterpret_borrow<py::object>(bfloat16));
#     }
#     case F16:
#       return py::dtype("e");  // PEP 3118 code for "float16
#     case F32:
#       return py::dtype::of<float>();
#     case F64:
#       return py::dtype::of<double>();
#     case C64:
#       return py::dtype::of<std::complex<float>>();
#     case C128:
#       return py::dtype::of<std::complex<double>>();
#     default:
#       return Unimplemented("Unimplemented primitive type %s",
#                            PrimitiveType_Name(type));
#   }
# }
