from typing import Sequence, Any
from jax._src.lib import xla_client as _xc
_xops = _xc._xla.ops
import inspect as _inspect
for k, v in _inspect.getmembers(_xc._xla.ops):
  if k.startswith('__'):
    continue
  globals()[k] = v

from . import xla_builder as _dxb
from . import xla_data_pb2 as _xd

_LiteralSlice = Any
_XlaBuilder = _dxb.XlaBuilder
_XlaComputation = _xc.XlaComputation
_XlaOp = _dxb.XlaOp
_PrimitiveType = _xd.PrimitiveType
_Shape = _xc.Shape
_Op = _dxb._Op



def Broadcast(operand: _XlaOp, sizes: Sequence[int]) -> _XlaOp:
  return _xops.Broadcast(_Op(operand), sizes)

def BroadcastInDim(operand: _XlaOp,
                   shape: Sequence[int],
                   broadcast_dimensions: Sequence[int]) -> _XlaOp:
  return _xops.BroadcastInDim(_Op(operand), shape, broadcast_dimensions)

def Call(builder: _XlaBuilder,
         computation: _XlaComputation,
         operands: Sequence[_XlaOp]) -> _XlaOp:
  return _xops.Call(builder, computation, _Op(operands))

def Conditional(branch_index: _XlaOp,
                branch_computations: Sequence[_XlaComputation],
                branch_operands: Sequence[_XlaOp]) -> _XlaOp:
  return _xops.Conditional(_Op(branch_index), branch_computations, _Op(branch_operands))

def ConstantLiteral(builder: _XlaBuilder, value: _LiteralSlice) -> _XlaOp:
  return _xops.ConstantLiteral(builder, value)

Constant = ConstantLiteral

def ConvertElementType(
        operand: _XlaOp,
        new_element_type: _PrimitiveType) -> _XlaOp:
  return _xops.ConvertElementType(_Op(operand), new_element_type)

def GetDimensionSize(operand: _XlaOp, index: int) -> _XlaOp:
  return _xops.GetDimensionSize(_Op(operand), index)

def GetTupleElement(tuple_data: _XlaOp, index: int) -> _XlaOp:
  return _xops.GetTupleElement(_Op(tuple_data), index)

def Parameter(
        builder: _XlaBuilder,
        parameter_number: int,
        shape: _Shape,
        name: str = None,
        replicated_at_leaf_buffers: Sequence[bool] = None) -> _XlaOp:
  if name is None:
    name = ''
  # return _xops.Parameter(builder, parameter_number, shape, name, replicated_at_leaf_buffers)
  return builder.Parameter(parameter_number, shape, name, replicated_at_leaf_buffers)

def Reduce(
        builder: _XlaBuilder,
        operands: Sequence[_XlaOp],
        init_values: Sequence[_XlaOp],
        computation: _XlaComputation,
        dimensions_to_reduce: Sequence[int]) -> _XlaOp:
  return _xops.Reduce(builder, _Op(operands), _Op(init_values), computation, dimensions_to_reduce)

def Tuple(builder: _XlaBuilder, elements: Sequence[_XlaOp]) -> _XlaOp:
  return _xops.Tuple(builder, _Op(elements))



def Eq(lhs: _XlaOp, rhs: _XlaOp, broadcast_dimensions: Sequence[int] = None) -> _XlaOp:
  return _xops.Eq(_Op(lhs), _Op(rhs), broadcast_dimensions)

def Ne(lhs: _XlaOp, rhs: _XlaOp, broadcast_dimensions: Sequence[int] = None) -> _XlaOp:
  return _xops.Ne(_Op(lhs), _Op(rhs), broadcast_dimensions)

def Ge(lhs: _XlaOp, rhs: _XlaOp, broadcast_dimensions: Sequence[int] = None) -> _XlaOp:
  return _xops.Ge(_Op(lhs), _Op(rhs), broadcast_dimensions)

def Gt(lhs: _XlaOp, rhs: _XlaOp, broadcast_dimensions: Sequence[int] = None) -> _XlaOp:
  return _xops.Gt(_Op(lhs), _Op(rhs), broadcast_dimensions)

def Lt(lhs: _XlaOp, rhs: _XlaOp, broadcast_dimensions: Sequence[int] = None) -> _XlaOp:
  return _xops.Lt(_Op(lhs), _Op(rhs), broadcast_dimensions)

def Le(lhs: _XlaOp, rhs: _XlaOp, broadcast_dimensions: Sequence[int] = None) -> _XlaOp:
  return _xops.Le(_Op(lhs), _Op(rhs), broadcast_dimensions)

def Add(lhs: _XlaOp, rhs: _XlaOp, broadcast_dimensions: Sequence[int] = None) -> _XlaOp:
  return _xops.Add(_Op(lhs), _Op(rhs), broadcast_dimensions)

def Sub(lhs: _XlaOp, rhs: _XlaOp, broadcast_dimensions: Sequence[int] = None) -> _XlaOp:
  return _xops.Sub(_Op(lhs), _Op(rhs), broadcast_dimensions)

def Mul(lhs: _XlaOp, rhs: _XlaOp, broadcast_dimensions: Sequence[int] = None) -> _XlaOp:
  return _xops.Mul(_Op(lhs), _Op(rhs), broadcast_dimensions)

def Div(lhs: _XlaOp, rhs: _XlaOp, broadcast_dimensions: Sequence[int] = None) -> _XlaOp:
  return _xops.Div(_Op(lhs), _Op(rhs), broadcast_dimensions)

def Neg(arg: _XlaOp) -> _XlaOp:
  return _xops.Neg(_Op(arg))

def Sin(arg: _XlaOp) -> _XlaOp:
  return _xops.Sin(_Op(arg))

def Cos(arg: _XlaOp) -> _XlaOp:
  return _xops.Cos(_Op(arg))

def Tan(arg: _XlaOp) -> _XlaOp:
  return _xops.Tan(_Op(arg))

def Tanh(arg: _XlaOp) -> _XlaOp:
  return _xops.Tanh(_Op(arg))
