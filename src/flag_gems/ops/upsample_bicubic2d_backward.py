import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as tle

device = device.name

logger = logging.getLogger(__name__)


def compute_scale(in_size, out_size, align_corners, scale):
    if align_corners:
        return (in_size - 1.0) / (out_size - 1.0) if out_size > 1 else 0.0
    else:
        return 1.0 / scale if scale is not None and scale > 0 else in_size / out_size


@triton.jit
def cubic_convolution1(x, A: tl.constexpr):
    # For |x| < 1: ((A + 2) * x - (A + 3)) * x * x + 1
    return ((A + 2) * x - (A + 3)) * x * x + 1


@triton.jit
def cubic_convolution2(x, A: tl.constexpr):
    # For 1 <= |x| < 2: ((A * x - 5 * A) * x + 8 * A) * x - 4 * A
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A


@triton.jit
def upsample_bicubic2d_backward_kernel(
    grad_input_ptr,
    grad_output_ptr,
    N,
    C,
    IH,
    IW,
    OH,
    OW,
    scale_h,
    scale_w,
    ALIGN_CORNERS: tl.constexpr,
):
    # Gather approach: each input pixel gathers contributions from output pixels
    pid = tle.program_id(axis=0)
    A = -0.75

    # Compute n, c, ih, iw from linear index
    total_pixels = N * C * IH * IW
    if pid >= total_pixels:
        return

    iw = pid % IW
    tmp = pid // IW
    ih = tmp % IH
    tmp = tmp // IH
    c = tmp % C
    n = tmp // C

    grad_acc = 0.0

    # Iterate over all output pixels
    for oh in range(OH):
        # Compute source coordinate
        if ALIGN_CORNERS:
            src_h = scale_h * oh
        else:
            src_h = scale_h * (oh + 0.5) - 0.5

        floor_h = tl.floor(src_h).to(tl.int32)
        t_h = src_h - floor_h.to(tl.float32)
        t_h = tl.minimum(tl.maximum(t_h, 0.0), 1.0)  # clamp to [0, 1]

        # The 4 sample positions for height: floor_h - 1, floor_h, floor_h + 1, floor_h + 2
        # Check if our target ih matches any of these (after clamping)

        for ow in range(OW):
            if ALIGN_CORNERS:
                src_w = scale_w * ow
            else:
                src_w = scale_w * (ow + 0.5) - 0.5

            floor_w = tl.floor(src_w).to(tl.int32)
            t_w = src_w - floor_w.to(tl.float32)
            t_w = tl.minimum(tl.maximum(t_w, 0.0), 1.0)

            # Compute weight contributions
            weight_sum = 0.0

            # Check all 4x4 positions
            for dh in range(-1, 3):
                ih_sample = floor_h + dh
                # Clamp to valid range
                ih_clamped = tl.minimum(tl.maximum(ih_sample, 0), IH - 1)

                # Check if this clamped position matches our target
                if ih_clamped == ih:
                    # Compute weight based on dh and t_h
                    if dh == -1:
                        weight_h = cubic_convolution2(t_h + 1.0, A)
                    elif dh == 0:
                        weight_h = cubic_convolution1(t_h, A)
                    elif dh == 1:
                        weight_h = cubic_convolution1(1.0 - t_h, A)
                    else:  # dh == 2
                        weight_h = cubic_convolution2(2.0 - t_h, A)

                    for dw in range(-1, 3):
                        iw_sample = floor_w + dw
                        iw_clamped = tl.minimum(tl.maximum(iw_sample, 0), IW - 1)

                        if iw_clamped == iw:
                            if dw == -1:
                                weight_w = cubic_convolution2(t_w + 1.0, A)
                            elif dw == 0:
                                weight_w = cubic_convolution1(t_w, A)
                            elif dw == 1:
                                weight_w = cubic_convolution1(1.0 - t_w, A)
                            else:  # dw == 2
                                weight_w = cubic_convolution2(2.0 - t_w, A)

                            weight_sum += weight_h * weight_w

            # Load grad_output and accumulate
            go_offset = ((n * C + c) * OH + oh) * OW + ow
            grad_out = tl.load(grad_output_ptr + go_offset)
            grad_acc += weight_sum * grad_out

    # Store result
    gi_offset = ((n * C + c) * IH + ih) * IW + iw
    tl.store(grad_input_ptr + gi_offset, grad_acc)


def upsample_bicubic2d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int],
    input_size: Tuple[int, int, int, int],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE_BICUBIC2D_BACKWARD")

    assert grad_output.device.type == device
    assert grad_output.ndim == 4, "grad_output must be 4D (N, C, H, W)"
    assert len(output_size) == 2, "output_size must be a tuple of 2 integers"
    assert len(input_size) == 4, "input_size must be a tuple of 4 integers"

    N, C, IH, IW = input_size
    OH, OW = output_size

    assert grad_output.shape == (N, C, OH, OW), \
        f"grad_output shape {grad_output.shape} doesn't match expected {(N, C, OH, OW)}"

    # Compute scale factors (same as PyTorch)
    scale_h = compute_scale(IH, OH, align_corners, scales_h)
    scale_w = compute_scale(IW, OW, align_corners, scales_w)

    # Allocate output
    grad_input = torch.empty(
        (N, C, IH, IW),
        device=grad_output.device,
        dtype=grad_output.dtype
    )

    total_pixels = N * C * IH * IW
    grid = (total_pixels,)

    with torch_device_fn.device(grad_output.device):
        upsample_bicubic2d_backward_kernel[grid](
            grad_input,
            grad_output,
            N,
            C,
            IH,
            IW,
            OH,
            OW,
            scale_h,
            scale_w,
            align_corners,
        )

    return grad_input
