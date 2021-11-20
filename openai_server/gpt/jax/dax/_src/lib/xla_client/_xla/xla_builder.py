from __future__ import annotations
from jax._src.lib import xla_client as _xc
from typing import Optional, Sequence, List, Any, NamedTuple, Set, Union
from . import xla_data_pb2 as _xd
from . import hlo_pb2 as _hlo
from . import hlo_opcode
from . import types as _xla_types

_XlaOpMetadata = Any
Shape = _xc.Shape
XlaComputation = _xc.XlaComputation
dtype_to_etype = _xc.dtype_to_etype


def _is_namedtuple(obj):
  return hasattr(obj, '_fields') and isinstance(obj, tuple)


def _Op(op: Union[_xc.XlaOp, Sequence[_xc.XlaOp]]) -> Union[_xc.XlaOp, Sequence[_xc.XlaOp]]:
  if op is not None:
    if isinstance(op, tuple) and not _is_namedtuple(op):
      return tuple(_Op(list(op)))
    if isinstance(op, list):
      return [_Op(x) for x in op]
    if isinstance(op, dict):
      return {k: _Op(v) for k, v in op.items()}
    return getattr(op, 'op', op)


def AsShape(sh) -> _xc.Shape:
  if isinstance(sh, _xc.Shape):
    return sh
  if isinstance(sh, bytes):
    sh = _xd.ShapeProto.FromString(sh)
  if sh.element_type == _xd.TUPLE:
    return _xc.Shape.tuple_shape([AsShape(x) for x in sh.tuple_shapes])
  if sh.element_type == _xd.TOKEN:
    return _xc.Shape.token_shape()
  assert hasattr(sh, 'layout')
  assert hasattr(sh, 'is_dynamic_dimension')
  if len(sh.layout.minor_to_major) < len(sh.dimensions):
    while len(sh.layout.minor_to_major) > 0:
      sh.layout.minor_to_major.pop()
    sh.layout.minor_to_major.extend(reversed(range(len(sh.dimensions))))
  while len(sh.is_dynamic_dimension) < len(sh.dimensions):
    sh.is_dynamic_dimension.append(False)
  return _xc.Shape.array_shape(
    _xla_types.PrimitiveTypeToDtype(sh.element_type),
    sh.dimensions,
    sh.layout.minor_to_major,
    sh.is_dynamic_dimension)


def AsShapeProto(sh: Any) -> _xd.ShapeProto:
  shape = AsShape(sh)
  proto = _xd.ShapeProto.FromString(shape.to_serialized_proto())
  return proto


class XlaOp(NamedTuple):
  handle: int
  builder: XlaBuilderExt
  op: _xc.XlaOp = None


class XlaBuilderExt(_xc._xla.XlaBuilder):
  # A temporary metadata that will only be applied to the next op created.
  _one_shot_metadata: Optional[_xd.OpMetadata]

  #  The metadata to attach to each op. This is structured as a "modal"-like
  #  operation, in order to simplify client code (and not sprinkle this metadata
  #  throughout the TensorFlow op kernel implementations).
  _metadata: _xd.OpMetadata

  _instructions: List[_hlo.HloInstructionProto]

  _instruction_shapes: List[_xc.Shape]

  # The unique parameter numbers.
  _parameter_numbers: Set[int]

  def __init__(self, name: str):
    super().__init__(name)
    self._next_id = 0
    self._name = name
    self._one_shot_metadata = None
    self._metadata = _xd.OpMetadata()
    self._handle_to_index = {}
    self._instructions = []
    self._instruction_shapes = []
    self._parameter_numbers = set()

  @property
  def name(self):
    return self._name

  def GetNextId(self) -> int:
    self._next_id += 1
    return self._next_id

  def AddInstruction(self,
                     instr: _hlo.HloInstructionProto,
                     opcode: hlo_opcode.HloOpcodeType,
                     operands: Optional[Sequence[XlaOp]] = None,
                     xla_op: Optional[_xc.XlaOp] = None) -> XlaOp:
    if operands is None:
      operands = []
    handle = self.GetNextId()
    instr.id = handle
    instr.opcode = hlo_opcode.HloOpcodeString(opcode)
    if not instr.name:
      instr.name = instr.opcode
    for operand in operands:
      if operand.builder is None:
        raise ValueError(f'invalid XlaOp with handle {operand.handle}')
      if operand.builder is not self:
        raise ValueError(f'Do not add XlaOp from builder {operand.builder.name} to builder {self.name}')
      instr.operand_ids.append(operand.handle)
    if self._one_shot_metadata is not None:
      instr.metadata.CopyFrom(self._one_shot_metadata)
      self._one_shot_metadata = None
    else:
      instr.metadata.CopyFrom(self._metadata)
    # TODO: sharding
    #   if (sharding_) {
    #     *instr.mutable_sharding() = *sharding_;
    #   }
    # TODO: frontend attributes
    #   *instr.mutable_frontend_attributes() = frontend_attributes_;
    #
    #   handle_to_index_[handle] = instructions_.size();
    self._handle_to_index[handle] = len(self._instructions)
    #   instructions_.push_back(std::move(instr));
    self._instructions.append(instr)
    #   instruction_shapes_.push_back(
    #       absl::make_unique<Shape>(instructions_.back().shape()));
    self._instruction_shapes.append(AsShape(self._instructions[-1].shape))
    #
    #   XlaOp op(handle, this);
    #   return op;
    op = XlaOp(handle, self, xla_op)
    return op

  def Parameter(self,
                parameter_number: int,
                shape: _xc.Shape,
                name: str,
                replicated_at_leaf_buffers: Optional[Sequence[bool]] = None,
                xla_op: Optional[_xc.XlaOp] = None) -> XlaOp:
    if replicated_at_leaf_buffers is None:
      replicated_at_leaf_buffers = []
    instr = _hlo.HloInstructionProto()
    if parameter_number in self._parameter_numbers:
      raise ValueError(f"parameter {parameter_number} already registered")
    self._parameter_numbers.add(parameter_number)
    instr.parameter_number = parameter_number
    instr.name = name
    instr.shape.CopyFrom(AsShapeProto(shape))
    if replicated_at_leaf_buffers:
      for replicated in replicated_at_leaf_buffers:
        instr.parameter_replication.replicated_at_leaf_buffers.append(replicated)
    if xla_op is None:
      xla_op = _xc._xla.ops.Parameter(self, parameter_number, shape, name, replicated_at_leaf_buffers)
    return self.AddInstruction(instr, hlo_opcode.HloOpcode.kParameter, xla_op=xla_op)


class XlaBuilder(XlaBuilderExt):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def Build(self, root: Optional[_xc._xla.XlaOp] = None) -> _xc._xla.XlaComputation:
    return super().Build(_Op(root))

  def GetShape(self, __op: _xc._xla.XlaOp) -> _xc._xla.Shape:
    return super().GetShape(_Op(__op))

  build = Build

  def clear_op_metadata(self) -> None:
    return super().clear_op_metadata()

  get_shape = GetShape

  def get_program_shape(self, root: Optional[_xc._xla.XlaOp] = None) -> _xc._xla.ProgramShape:
    return super().get_program_shape(_Op(root))

  def is_constant(self, __op: _xc._xla.XlaOp) -> bool:
    return super().is_constant(_Op(__op))

  def set_op_metadata(self, metadata: _XlaOpMetadata) -> None:
    return super().set_op_metadata(metadata)

  def set_sharding(self, sharding: _xc._xla.OpSharding_Type) -> None:
    return super().set_sharding(sharding)

  def clear_sharding(self) -> None:
    return super().clear_sharding()

  def setup_alias(
          self,
          __output_index: Sequence[int],
          __param_number: int,
          __param_index: Sequence[int]) -> None:
    return super().setup_alias(__output_index, __param_number, __param_index)
