
from typing import (Any, Callable, List, NamedTuple, Optional, Sequence, Union, Tuple)

Array = Any
DType = Any

def index_in_dim(operand: Array, index: int, axis: int = 0,
                 keepdims: bool = True):
  """Convenience wrapper around slice to perform int indexing."""
  index, axis = int(index), int(axis)
  axis_size = operand.shape[axis]
  wrapped_index = index + axis_size if index < 0 else index
  if not 0 <= wrapped_index < axis_size:
    raise IndexError(f'index {index} is out of bounds for axis {axis} with size {axis_size}')
  result = slice_in_dim(operand, wrapped_index, wrapped_index + 1, 1, axis)
  if keepdims:
    return result
  else:
    return squeeze(result, (axis,))

def squeeze(array: Array, dimensions: Tuple[int, ...]) -> Array:
  """Squeeze any number of size 1 dimensions from an array."""
