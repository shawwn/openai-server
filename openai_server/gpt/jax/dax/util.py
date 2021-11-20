from jax._src.util import (safe_map, safe_zip, prod, canonicalize_axis, moveaxis, ceil_of_ratio, unzip2, unzip3,
                           split_list, partition_list)

# def canonicalize_axis(axis, num_dims) -> int:
#   """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
#   axis = operator.index(axis)