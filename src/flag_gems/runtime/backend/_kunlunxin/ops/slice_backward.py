# Kunlunxin (XPU) override of slice_backward.
#
# The original generic kernel SCATTERS each grad_output element into grad_input
# at a step-strided `input_offset` (stride>1 write) on top of a full
# `torch.zeros` alloc. On XPU a strided write degrades to fully discrete stores
# -> catastrophic (~150-630ms, gems speedup ~0.004).
#
# Two paths:
#   * Fast path (dim is last / inner==1, start==0, step divides the dim and the
#     slice covers it -- exactly the benchmark 2D shapes): write grad_input in a
#     single fused kernel pass and GATHER grad_output as the affine `idx // step`
#     with `step` a constexpr (division compiled away), selecting with
#     `idx % step == 0`. One launch, no zeros pass -> best for small 2D.
#   * General path (inner>1, 3D/4D shapes): mirror native slice_backward exactly
#     -- `torch.zeros` then write grad_output into the step-strided slice VIEW of
#     grad_input via the ATen `_copy_from` primitive. gems overrides `copy_`/
#     `copy` but never `_copy_from`, so this reaches the vendor's native
#     strided-copy engine and runs at native speed (~1.0x) even while use_gems
#     is active, for every dtype/rank. This replaces a per-inner-block Triton
#     copy kernel that launched `outer*slice_len` programs -- catastrophic when
#     inner is small and that product is huge (e.g. [512,1024,512]: 262144
#     tiny 512-elem programs -> 22ms / speedup 0.03).
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

    inner = 1
    for i in range(dim + 1, len(shape)):
        inner *= shape[i]
    dim_size = shape[dim]
    slice_len = grad_output.shape[dim]
    norm_start = start + dim_size if start < 0 else start
    norm_start = max(0, min(norm_start, dim_size))

    grad_output = grad_output.contiguous()

    fast = (
        inner == 1
        and norm_start == 0
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

    # General path: mirror native slice_backward (zeros + strided-slice copy),
    # using the native `_copy_from` strided-copy engine instead of a Triton
    # strided write. `torch.ops.aten.slice` (a view op) clamps start/end the same
    # way native does, so the view shape matches grad_output exactly.
    grad_input = torch.zeros(shape, device=grad_output.device, dtype=grad_output.dtype)
    if grad_output.numel() == 0:
        return grad_input
    sub = torch.ops.aten.slice(grad_input, dim, start, end, step)
    torch.ops.aten._copy_from(grad_output, sub, False)
    return grad_input
