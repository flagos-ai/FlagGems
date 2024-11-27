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
