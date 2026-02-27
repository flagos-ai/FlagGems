import logging
from typing import List, Optional

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

device = device.name
logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("upsample_bilinear2d_backward"),
    key=["N", "C", "IH", "IW"],
)
@triton.jit
def upsample_bilinear2d_backward_kernel(
    grad_input_ptr,
    grad_output_ptr,
    N,
    C,
    IH,
    IW,
    OH,
    OW,
    rheight,
    rwidth,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward pass for bilinear 2D upsampling.

    For each input pixel, find all output pixels that used it in the forward pass
    and accumulate the weighted gradients.
    """
    pid = tl.program_id(axis=0)
    nc = tl.program_id(axis=1)
    n = nc // C
    c = nc % C

    # Each thread handles one input element
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    iw = idx % IW
    ih = idx // IW

    mask = (idx < IH * IW)

    # Initialize gradient accumulator
    grad_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # For each input pixel (ih, iw), we need to find all output pixels (oh, ow)
    # that bilinearly interpolated from it.
    #
    # Forward mapping: For output (oh, ow), source coords are:
    # align_corners=False: h_src = (oh + 0.5) * rheight - 0.5
    # align_corners=True:  h_src = oh * rheight
    #
    # Backward: input pixel ih contributes if floor(h_src) == ih or floor(h_src) + 1 == ih
    #
    # Compute range of output pixels that could use this input pixel

    if align_corners:
        # h_src = oh * rheight
        # ih contributes if floor(oh * rheight) == ih or floor(oh * rheight) + 1 == ih
        # oh_min: smallest oh such that oh * rheight >= ih - 1
        # oh_max: largest oh such that oh * rheight < ih + 1
        if rheight > 0:
            oh_min_float = (ih.to(tl.float32) - 1) / rheight
            oh_max_float = (ih.to(tl.float32) + 1) / rheight
        else:
            oh_min_float = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            oh_max_float = tl.full((BLOCK_SIZE,), OH, dtype=tl.float32)
    else:
        # h_src = (oh + 0.5) * rheight - 0.5
        # h_src in [ih-1, ih+1) means this input pixel contributes
        # ih - 1 <= (oh + 0.5) * rheight - 0.5 < ih + 1
        # (ih - 0.5) / rheight - 0.5 <= oh < (ih + 1.5) / rheight - 0.5
        oh_min_float = (ih.to(tl.float32) - 0.5) / rheight - 0.5
        oh_max_float = (ih.to(tl.float32) + 1.5) / rheight - 0.5

    oh_start = tl.maximum(oh_min_float.to(tl.int32), 0)
    oh_end = tl.minimum((oh_max_float + 1).to(tl.int32), OH)

    if align_corners:
        if rwidth > 0:
            ow_min_float = (iw.to(tl.float32) - 1) / rwidth
            ow_max_float = (iw.to(tl.float32) + 1) / rwidth
        else:
            ow_min_float = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            ow_max_float = tl.full((BLOCK_SIZE,), OW, dtype=tl.float32)
    else:
        ow_min_float = (iw.to(tl.float32) - 0.5) / rwidth - 0.5
        ow_max_float = (iw.to(tl.float32) + 1.5) / rwidth - 0.5

    ow_start = tl.maximum(ow_min_float.to(tl.int32), 0)
    ow_end = tl.minimum((ow_max_float + 1).to(tl.int32), OW)

    # Base offset for grad_output
    grad_out_base = (n * C + c) * OH * OW

    # Iterate over candidate output pixels
    # For an S-times upsample, each input contributes to ~2S output pixels per dimension
    # Use a larger fixed bound to handle various upsampling factors (up to ~8x)
    for oh_offset in range(16):  # handles up to ~8x upsample
        oh = oh_start + oh_offset
        oh_valid = (oh < oh_end) & (oh < OH) & (oh >= 0)

        # Compute source h coordinate for this output
        if align_corners:
            h_src = oh.to(tl.float32) * rheight
        else:
            h_src = tl.maximum((oh.to(tl.float32) + 0.5) * rheight - 0.5, 0.0)

        h0 = tl.minimum(h_src.to(tl.int32), IH - 1)
        h1p = tl.minimum(h0 + 1, IH - 1)
        h_lambda = h_src - h0.to(tl.float32)
        h0_lambda = 1.0 - h_lambda

        # Check if this input pixel ih is h0 or h1p
        # Note: if h0 == h1p (at boundary), both conditions can be true
        # and we need to sum both weights
        is_h0 = (ih == h0)
        is_h1 = (ih == h1p)
        # When h0 == h1p (boundary), both weights should be added
        h_weight = tl.where(is_h0, h0_lambda, 0.0) + tl.where(is_h1, h_lambda, 0.0)

        for ow_offset in range(16):  # handles up to ~8x upsample
            ow = ow_start + ow_offset
            ow_valid = (ow < ow_end) & (ow < OW) & (ow >= 0)

            # Compute source w coordinate
            if align_corners:
                w_src = ow.to(tl.float32) * rwidth
            else:
                w_src = tl.maximum((ow.to(tl.float32) + 0.5) * rwidth - 0.5, 0.0)

            w0 = tl.minimum(w_src.to(tl.int32), IW - 1)
            w1p = tl.minimum(w0 + 1, IW - 1)
            w_lambda = w_src - w0.to(tl.float32)
            w0_lambda = 1.0 - w_lambda

            # Check if this input pixel iw is w0 or w1p
            # When w0 == w1p (boundary), both weights should be added
            is_w0 = (iw == w0)
            is_w1 = (iw == w1p)
            w_weight = tl.where(is_w0, w0_lambda, 0.0) + tl.where(is_w1, w_lambda, 0.0)

            # Combined weight and validity
            weight = h_weight * w_weight
            valid = mask & oh_valid & ow_valid & (weight > 0.0)

            # Load grad_output for this (oh, ow)
            grad_out_offset = grad_out_base + oh * OW + ow
            grad_out = tl.load(grad_output_ptr + grad_out_offset, mask=valid, other=0.0)

            # Accumulate weighted gradient
            grad_acc += tl.where(valid, grad_out.to(tl.float32) * weight, 0.0)

    # Store accumulated gradient to grad_input
    grad_in_offset = (n * C + c) * IH * IW + ih * IW + iw
    tl.store(grad_input_ptr + grad_in_offset, grad_acc, mask=mask)


def upsample_bilinear2d_backward(
    grad_output: torch.Tensor,
    output_size: List[int],
    input_size: List[int],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    """
    Backward pass for bilinear 2D upsampling.

    Args:
        grad_output: Gradient w.r.t. output of forward pass, shape [N, C, OH, OW]
        output_size: [OH, OW] - size of the forward output
        input_size: [N, C, IH, IW] - size of the forward input
        align_corners: Whether to align corners in interpolation
        scales_h: Optional scale factor for height
        scales_w: Optional scale factor for width

    Returns:
        Gradient w.r.t. input of forward pass, shape [N, C, IH, IW]
    """
    logger.debug("GEMS UPSAMPLE_BILINEAR2D_BACKWARD")

    assert grad_output.device.type == device
    assert grad_output.ndim == 4, "grad_output must be 4D tensor"
    assert len(output_size) == 2, "output_size must have 2 elements"
    assert len(input_size) == 4, "input_size must have 4 elements"

    N, C, IH, IW = input_size
    OH, OW = output_size

    # Verify grad_output shape matches
    assert grad_output.shape == (N, C, OH, OW), \
        f"grad_output shape {grad_output.shape} doesn't match expected {(N, C, OH, OW)}"

    # Compute scale factors (input to output ratio)
    if align_corners:
        if OH > 1:
            rheight = (IH - 1) / (OH - 1) if IH > 1 else 0.0
        else:
            rheight = 0.0
        if OW > 1:
            rwidth = (IW - 1) / (OW - 1) if IW > 1 else 0.0
        else:
            rwidth = 0.0
    else:
        rheight = IH / OH
        rwidth = IW / OW

    # Allocate output gradient (grad_input)
    # Use float32 for accumulation to avoid precision issues
    grad_input = torch.empty(
        (N, C, IH, IW),
        device=grad_output.device,
        dtype=torch.float32
    )

    # Make grad_output contiguous
    grad_output = grad_output.contiguous()

    # Launch kernel
    total_input_elements = IH * IW
    grid = lambda META: (
        triton.cdiv(total_input_elements, META["BLOCK_SIZE"]),
        N * C,
    )

    with torch_device_fn.device(grad_output.device):
        upsample_bilinear2d_backward_kernel[grid](
            grad_input,
            grad_output,
            N,
            C,
            IH,
            IW,
            OH,
            OW,
            rheight,
            rwidth,
            align_corners,
        )

    # Convert back to original dtype
    if grad_output.dtype != torch.float32:
        grad_input = grad_input.to(grad_output.dtype)

    return grad_input
