# Kunlunxin (XPU) override of slice_backward.
#
# The generic kernel iterates grad_output (contiguous read) and SCATTERS each
# element into grad_input at a step-strided `input_offset` (stride>1 write),
# on top of a full `torch.zeros` alloc. On XPU a strided write degrades to
# fully discrete stores -> catastrophic (~150-630ms, gems speedup ~0.004).
#
# Fix: never do a strided write.
#   * Fast path (dim is last / inner==1, start==0, step divides the dim and the
#     slice covers it -- exactly the benchmark 2D shapes): write grad_input
#     contiguously and GATHER grad_output as the affine `idx // step` with
#     `step` a constexpr (division compiled away), selecting with `idx%step==0`.
#   * General path (inner>1, 3D/4D shapes): allocate zeros, then launch one
#     program per selected (outer, slice) row and copy its whole `inner` block
#     with `tl.arange` over the inner dim. Both the grad_output read and the
#     grad_input write are then base(scalar)+arange -> provable stride-1 block
#     DMA. Only selected rows run (no wasted zero-writes, no per-element mask).
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def slice_backward_fast_kernel(
    grad_input_ptr,
    grad_output_ptr,
    total_elements,
    step: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # inner==1, start==0, full coverage: grad_input[i] = (i % step == 0) ?
    # grad_output[i // step] : 0. step constexpr -> no runtime division.
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    sel = idx % step == 0
    go = tl.load(grad_output_ptr + (idx // step), mask=mask & sel, other=0)
    result = tl.where(sel, go, 0.0)
    tl.store(grad_input_ptr + idx, result, mask=mask)


@triton.jit
def slice_backward_copy_kernel(
    grad_input_ptr,
    grad_output_ptr,
    dim_size,
    inner,
    slice_len,
    start,
    step,
    BLOCK_INNER: tl.constexpr,
):
    # One program per selected (outer, slice) row; copies the contiguous `inner`
    # block from grad_output into its remapped position in grad_input. Both
    # addresses are scalar_base + arange -> stride-1 block DMA. grad_input was
    # pre-zeroed so non-selected rows need no work.
    s = tl.program_id(0)
    pid_i = tl.program_id(1)
    outer_idx = s // slice_len
    slice_idx = s % slice_len
    dim_idx = start + slice_idx * step
    inner_off = pid_i * BLOCK_INNER + tl.arange(0, BLOCK_INNER)
    imask = inner_off < inner
    gi_base = (outer_idx * dim_size + dim_idx) * inner
    go_base = (outer_idx * slice_len + slice_idx) * inner
    v = tl.load(grad_output_ptr + go_base + inner_off, mask=imask)
    tl.store(grad_input_ptr + gi_base + inner_off, v, mask=imask)


def _block_for(numel):
    if numel <= (1 << 14):
        return 1024, 4
    if numel <= (1 << 18):
        return 8192, 8
    return 16384, 8


def slice_backward(grad_output, input_sizes, dim, start, end, step):
    logger.debug("GEMS_KUNLUNXIN SLICE_BACKWARD")
    shape = list(input_sizes)
    if dim < 0:
        dim += len(shape)

    outer = 1
    for i in range(dim):
        outer *= shape[i]
    inner = 1
    for i in range(dim + 1, len(shape)):
        inner *= shape[i]
    dim_size = shape[dim]
    slice_len = grad_output.shape[dim]
    if start < 0:
        start += dim_size
    start = max(0, min(start, dim_size))

    grad_output = grad_output.contiguous()

    fast = (
        inner == 1
        and start == 0
        and dim_size % step == 0
        and slice_len * step == dim_size
    )
    if fast:
        grad_input = torch.empty(
            shape, device=grad_output.device, dtype=grad_output.dtype
        )
        total = grad_input.numel()
        block_size, num_warps = _block_for(total)
        grid = (triton.cdiv(total, block_size),)
        slice_backward_fast_kernel[grid](
            grad_input,
            grad_output,
            total,
            step=step,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return grad_input

    grad_input = torch.zeros(shape, device=grad_output.device, dtype=grad_output.dtype)
    if slice_len == 0 or inner == 0:
        return grad_input
    block_inner = min(triton.next_power_of_2(inner), 8192)
    grid = (outer * slice_len, triton.cdiv(inner, block_inner))
    slice_backward_copy_kernel[grid](
        grad_input,
        grad_output,
        dim_size,
        inner,
        slice_len,
        start,
        step,
        BLOCK_INNER=block_inner,
        num_warps=8,
    )
    return grad_input
