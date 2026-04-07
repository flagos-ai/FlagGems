import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# upsample_nearest2d_backward: Computes the gradient of upsample_nearest2d.
# Each input (grad_output) pixel maps to a region in the output (grad_input).
# Multiple output pixels may contribute to the same input pixel, so we
# accumulate gradients using atomic_add.
@libentry()
@triton.jit
def upsample_nearest2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decode output linear index to (nc, oh, ow)
    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    nc = tmp // OH

    # Map output position to input position (same as forward)
    ih = tl.minimum((oh.to(tl.float32) * reciprocal_scale_h).to(tl.int32), IH - 1)
    iw = tl.minimum((ow.to(tl.float32) * reciprocal_scale_w).to(tl.int32), IW - 1)

    # Load gradient from output and cast to float32 for precise accumulation
    grad_val = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute input index and accumulate
    input_idx = nc * IH * IW + ih * IW + iw
    tl.atomic_add(grad_input_ptr + input_idx, grad_val, mask=mask)


def upsample_nearest2d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int],
    input_size: Tuple[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE_NEAREST2D_BACKWARD")

    N, C, IH, IW = input_size
    OH, OW = output_size

    if scales_h is not None:
        reciprocal_scale_h = 1 / scales_h
    else:
        reciprocal_scale_h = IH / OH
    if scales_w is not None:
        reciprocal_scale_w = 1 / scales_w
    else:
        reciprocal_scale_w = IW / OW

    grad_output = grad_output.contiguous()
    grad_input = torch.zeros(
        (N, C, IH, IW), device=grad_output.device, dtype=grad_output.dtype
    )

    n_elements = grad_output.numel()
    if n_elements == 0:
        return grad_input.to(grad_output.dtype)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    with torch_device_fn.device(grad_output.device):
        upsample_nearest2d_backward_kernel[grid](
            grad_output,
            grad_input,
            N,
            C,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return grad_input.to(grad_output.dtype)
