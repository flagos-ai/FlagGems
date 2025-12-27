import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def copy_kernel_linear(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    vals = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, vals, mask=mask)


@triton.jit
def copy_kernel_nd(
    src_ptr, dst_ptr,
    shape_ptr, src_stride_ptr, dst_stride_ptr,
    numel,
    NDIMS: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    linear = offs.to(tl.int64)

    src_offset = tl.zeros([BLOCK], dtype=tl.int64)
    dst_offset = tl.zeros([BLOCK], dtype=tl.int64)

    for d in range(NDIMS - 1, -1, -1):
        dim = tl.load(shape_ptr + d)
        idx = linear % dim
        linear = linear // dim
        src_stride = tl.load(src_stride_ptr + d)
        dst_stride = tl.load(dst_stride_ptr + d)
        src_offset += idx * src_stride
        dst_offset += idx * dst_stride

    val = tl.load(src_ptr + src_offset, mask=mask)
    tl.store(dst_ptr + dst_offset, val, mask=mask)