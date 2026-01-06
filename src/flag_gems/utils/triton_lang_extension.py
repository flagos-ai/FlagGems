"""
Triton device functions.

Custom triton device functions that we need to use.

NOTE:
Do not try to add triton builtin-style functions(functions with an ir builder in its
arguments) here. We only define device-functions(triton.jit decorated functions with
return statement) here.

These functions can be used in kernel progamming and are not bound to any grid.
"""

import triton
from triton import language as tl


@triton.jit
def program_id(
    axis: int,
) -> tl.tensor:
    return tl.program_id(axis).to(tl.int64)


@triton.jit
def num_programs(
    axis: int,
) -> tl.tensor:
    return tl.num_programs(axis).to(tl.int64)


@triton.jit
def promote_to_tensor(x):
    # Addition promotes to tensor for us
    return x + tl.zeros((1,), tl.int1)


@triton.jit
def is_floating(x):
    return promote_to_tensor(x).dtype.is_floating()


@triton.jit
def minimum_with_index_tie_break_right(a_value, a_index, b_value, b_index):
    mask = a_value < b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan and not b_isnan
        # Consider NaNs as equal
        equal |= a_isnan and b_isnan

    # Prefer highest index if values are equal
    mask |= equal & (a_index > b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)
