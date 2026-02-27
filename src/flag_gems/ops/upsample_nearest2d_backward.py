import logging
from typing import Optional, Tuple

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
    configs=runtime.get_tuned_config("upsample_nearest2d_backward"),
    key=["N", "C", "IH", "IW"],
)
@triton.heuristics(runtime.get_heuristic_config("upsample_nearest2d_backward"))
@triton.jit
def upsample_nearest2d_backward_kernel(
    ptr_grad_input,
    ptr_grad_output,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    scale_h,
    scale_w,
    BLOCK_SIZE: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    """
    Backward pass for nearest neighbor 2D upsampling.

    For each input position (ih, iw), accumulate gradients from all output
    positions (oh, ow) that mapped to this input in the forward pass.

    Forward mapping: ih = floor(oh * IH / OH), iw = floor(ow * IW / OW)
    Backward: grad_input[ih, iw] = sum of grad_output[oh, ow] for all (oh, ow)
              that mapped to (ih, iw)
    """
    if USE_INT32_IDX:
        pid = tl.program_id(axis=0)
    else:
        pid = tl.program_id(axis=0).to(tl.int64)

    nc_stride = tl.num_programs(axis=1)
    NC = N * C
    nc_iter = tl.program_id(axis=1)

    # Each thread block processes BLOCK_SIZE input positions
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    iw = idx % IW
    ih = idx // IW % IH

    # Compute the range of output positions that map to each input position
    # For forward: ih = floor(oh * IH / OH)
    # So: ih <= oh * IH / OH < ih + 1
    # => oh >= ih * OH / IH and oh < (ih + 1) * OH / IH
    if SAME_H:
        oh_start = ih
        oh_end = ih + 1
    else:
        # For forward: ih = floor(oh * IH / OH) = floor(oh / scale_h)
        # Backward: find oh such that floor(oh / scale_h) == ih
        # => ih <= oh / scale_h < ih + 1
        # => ih * scale_h <= oh < (ih + 1) * scale_h
        # oh_start = ceil(ih * scale_h), oh_end = ceil((ih + 1) * scale_h)
        oh_start_f = ih.to(tl.float32) * scale_h
        oh_end_f = (ih + 1).to(tl.float32) * scale_h
        oh_start = tl.math.ceil(oh_start_f).to(tl.int32)
        oh_end = tl.math.ceil(oh_end_f).to(tl.int32)
        oh_start = tl.maximum(oh_start, 0)
        oh_end = tl.minimum(oh_end, OH)

    if SAME_W:
        ow_start = iw
        ow_end = iw + 1
    else:
        ow_start_f = iw.to(tl.float32) * scale_w
        ow_end_f = (iw + 1).to(tl.float32) * scale_w
        ow_start = tl.math.ceil(ow_start_f).to(tl.int32)
        ow_end = tl.math.ceil(ow_end_f).to(tl.int32)
        ow_start = tl.maximum(ow_start, 0)
        ow_end = tl.minimum(ow_end, OW)

    # Accumulate gradients
    grad_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    input_stride = nc_stride * IH * IW
    output_stride = nc_stride * OH * OW

    while nc_iter < NC:
        # Reset accumulator for this nc position
        grad_acc_local = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        # For each input position, sum gradients from all contributing output positions
        # We need to iterate over the output region that maps to each input
        # Since the region size depends on scale, we use a fixed maximum iteration count
        # and mask out invalid positions

        # For typical upsampling (scale >= 1), region is small
        # For downsampling (scale < 1), region could be 1 element
        # We'll use a loop with maximum reasonable extent

        # Compute gradients for this block
        # For each oh in [oh_start, oh_end) and ow in [ow_start, ow_end)
        if SAME_H and SAME_W:
            # Identity case: just copy
            offset_o = (nc_iter * OH + ih) * OW + iw
            mask = idx < IH * IW
            grad_acc_local = tl.load(ptr_grad_output + offset_o, mask=mask, other=0.0)
        else:
            # General case: accumulate over output region
            # Since region sizes vary, we iterate over the maximum possible extent
            # For nearest neighbor with integer scales, h_extent = ceil(scale_h)
            h_extent = tl.maximum(oh_end - oh_start, 1)
            w_extent = tl.maximum(ow_end - ow_start, 1)

            # We need a fixed loop bound, so estimate max extent
            # For safety, iterate over all possible outputs and check bounds
            for delta_h in range(16):  # Max scale factor of 16
                for delta_w in range(16):
                    oh = oh_start + delta_h
                    ow = ow_start + delta_w
                    valid = (oh < oh_end) & (ow < ow_end) & (oh >= 0) & (ow >= 0)
                    valid = valid & (oh < OH) & (ow < OW)
                    valid = valid & (idx < IH * IW)

                    offset_o = (nc_iter * OH + oh) * OW + ow
                    grad_val = tl.load(ptr_grad_output + offset_o, mask=valid, other=0.0)
                    grad_acc_local += grad_val

        # Store accumulated gradient
        offset_i = (nc_iter * IH + ih) * IW + iw
        mask = idx < IH * IW
        tl.store(ptr_grad_input + offset_i, grad_acc_local, mask=mask)

        nc_iter += nc_stride


def upsample_nearest2d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int],
    input_size: Tuple[int, int, int, int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE_NEAREST2D_BACKWARD")
    assert grad_output.device.type == device
    assert grad_output.ndim == 4, "The ndim of grad_output must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    assert len(input_size) == 4, "The len of input_size must be 4"

    OH, OW = output_size
    N, C, IH, IW = input_size

    # Validate dimensions match
    assert grad_output.shape[0] == N, "Batch size mismatch"
    assert grad_output.shape[1] == C, "Channel size mismatch"
    assert grad_output.shape[2] == OH, "Height mismatch"
    assert grad_output.shape[3] == OW, "Width mismatch"

    # Compute scale factors (output_size / input_size)
    if scales_h is not None:
        scale_h = scales_h
    else:
        scale_h = OH / IH
    if scales_w is not None:
        scale_w = scales_w
    else:
        scale_w = OW / IW

    # Make input contiguous
    grad_output = grad_output.contiguous()

    # Allocate output
    grad_input = torch.empty(
        (N, C, IH, IW), device=grad_output.device, dtype=grad_output.dtype
    )

    if grad_input.numel() == 0:
        return grad_input

    total_threads = IH * IW
    grid = lambda META: (
        triton.cdiv(total_threads, META["BLOCK_SIZE"]),
        triton.cdiv(N * C, 4),
    )

    with torch_device_fn.device(grad_output.device):
        upsample_nearest2d_backward_kernel[grid](
            grad_input,
            grad_output,
            N,
            C,
            OH,
            OW,
            IH,
            IW,
            scale_h,
            scale_w,
        )

    return grad_input
