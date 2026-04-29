import logging
import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)


@triton.jit
def upsample_nearest2d_backward_kernel(
    grad_output,
    grad_input,
    total,
    OH,
    OW,
    IH,
    IW,
    inverse_scale_h,
    inverse_scale_w,
    BLOCK_SIZE: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    MAX_OH: tl.constexpr,
    MAX_OW: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    spatial = IH * IW
    nc = offsets // spatial
    rem = offsets - nc * spatial
    ih = rem // IW
    iw = rem - ih * IW

    if SAME_H:
        oh_start = ih
        oh_end = ih + 1
    else:
        oh_start = tl.ceil(ih.to(tl.float32) * inverse_scale_h).to(tl.int32)
        oh_end = tl.ceil((ih.to(tl.float32) + 1.0) * inverse_scale_h).to(tl.int32)
        oh_end = tl.where(ih == IH - 1, OH, oh_end)
        oh_start = tl.minimum(tl.maximum(oh_start, 0), OH)
        oh_end = tl.minimum(tl.maximum(oh_end, oh_start), OH)
    if SAME_W:
        ow_start = iw
        ow_end = iw + 1
    else:
        ow_start = tl.ceil(iw.to(tl.float32) * inverse_scale_w).to(tl.int32)
        ow_end = tl.ceil((iw.to(tl.float32) + 1.0) * inverse_scale_w).to(tl.int32)
        ow_end = tl.where(iw == IW - 1, OW, ow_end)
        ow_start = tl.minimum(tl.maximum(ow_start, 0), OW)
        ow_end = tl.minimum(tl.maximum(ow_end, ow_start), OW)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for dh in tl.static_range(MAX_OH):
        oh = oh_start + dh
        oh_valid = oh < oh_end
        for dw in tl.static_range(MAX_OW):
            ow = ow_start + dw
            ow_valid = ow < ow_end
            out_offsets = (nc * OH + oh) * OW + ow
            valid = mask & oh_valid & ow_valid
            acc += tl.load(grad_output + out_offsets, mask=valid, other=0.0).to(
                tl.float32
            )

    tl.store(grad_input + offsets, acc, mask=mask)


def upsample_nearest2d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int],
    input_size: Tuple[int, int, int, int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE NEAREST2D BACKWARD")
    assert grad_output.device.type == device
    assert grad_output.ndim == 4, "The ndim of grad_output must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    assert len(input_size) == 4, "The len of input_size must be 4"

    N, C, IH, IW = input_size
    OH, OW = output_size
    if scales_h is not None:
        reciprocal_scale_h = float(
            torch.tensor(1.0 / scales_h, dtype=torch.float32).item()
        )
    else:
        reciprocal_scale_h = float(
            (
                torch.tensor(IH, dtype=torch.float32)
                / torch.tensor(OH, dtype=torch.float32)
            ).item()
        )
    if scales_w is not None:
        reciprocal_scale_w = float(
            torch.tensor(1.0 / scales_w, dtype=torch.float32).item()
        )
    else:
        reciprocal_scale_w = float(
            (
                torch.tensor(IW, dtype=torch.float32)
                / torch.tensor(OW, dtype=torch.float32)
            ).item()
        )

    grad_output = grad_output.contiguous()
    grad_input = torch.empty(
        (N, C, IH, IW), device=grad_output.device, dtype=grad_output.dtype
    )
    total = grad_input.numel()
    if total == 0 or grad_input.numel() == 0:
        return grad_input

    inverse_scale_h = 1.0 / reciprocal_scale_h
    inverse_scale_w = 1.0 / reciprocal_scale_w
    max_oh = max(1, min(OH, math.ceil(inverse_scale_h) + 1))
    max_ow = max(1, min(OW, math.ceil(inverse_scale_w) + 1))

    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(grad_output.device):
        upsample_nearest2d_backward_kernel[grid](
            grad_output,
            grad_input,
            total,
            OH,
            OW,
            IH,
            IW,
            inverse_scale_h,
            inverse_scale_w,
            SAME_H=IH == OH,
            SAME_W=IW == OW,
            BLOCK_SIZE=256,
            MAX_OH=max_oh,
            MAX_OW=max_ow,
        )
    return grad_input
