import logging
import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_nearest2d"), key=["N", "C", "OH", "OW"]
)
@triton.heuristics(runtime.get_heuristic_config("upsample_nearest2d"))
@triton.jit
def upsample_nearest2d_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        pid = tl.program_id(axis=0)
    else:
        pid = tl.program_id(axis=0).to(tl.int64)
    nc_stride = tl.num_programs(axis=1)
    NC = N * C
    nc_iter = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ow = idx % OW
    oh = idx // OW % OH
    if SAME_H:
        ih = oh
    else:
        # tl.floor() cannot be found in 2.3.1, using int trunc
        ih = tl.minimum((oh * reciprocal_scale_h).to(tl.int32), IH - 1)
    if SAME_W:
        iw = ow
    else:
        iw = tl.minimum((ow * reciprocal_scale_w).to(tl.int32), IW - 1)

    offset_o = (nc_iter * OH + oh) * OW + ow
    offset_i = (nc_iter * IH + ih) * IW + iw
    src_index_stride = nc_stride * IH * IW
    dst_index_stride = nc_stride * OH * OW
    while nc_iter < NC:
        data = tl.load(ptr_i + offset_i)
        tl.store(ptr_o + offset_o, data)
        ptr_i += src_index_stride
        ptr_o += dst_index_stride
        nc_iter += nc_stride


def upsample_nearest2d(
    input: torch.Tensor,
    output_size: Tuple[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE NEAREST2D")
    assert input.device.type == device
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    OH, OW = output_size
    N, C, IH, IW = input.shape
    if scales_h is not None:
        reciprocal_scale_h = 1 / scales_h
    else:
        reciprocal_scale_h = IH / OH
    if scales_w is not None:
        reciprocal_scale_w = 1 / scales_w
    else:
        reciprocal_scale_w = IW / OW
    # allocate output
    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    total_threads = OH * OW
    grid = lambda META: (
        triton.cdiv(total_threads, META["BLOCK_SIZE"]),
        triton.cdiv(N * C, 4),
    )

    with torch_device_fn.device(input.device):
        upsample_nearest2d_kernel[grid](
            output,
            input,
            N,
            C,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
        )
    return output


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_nearest2d_backward"),
    key=["N", "C", "IH", "IW"],
)
@triton.jit
def upsample_nearest2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    N,
    C,
    IH,
    IW,
    OH,
    OW,
    scale_h,
    scale_w,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
    MAX_REGION_H: tl.constexpr,
    MAX_REGION_W: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    nc_stride = tl.num_programs(axis=1)
    NC = N * C
    nc_iter = tl.program_id(axis=1)

    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    iw = idx % IW
    ih = idx // IW

    mask = ih < IH

    if SAME_H:
        oh_start = ih
        oh_end = ih + 1
    else:
        oh_start = (ih * scale_h).to(tl.int32)
        oh_start = tl.where(
            tl.minimum((oh_start * reciprocal_scale_h).to(tl.int32), IH - 1) < ih,
            oh_start + 1,
            oh_start,
        )
        oh_end = ((ih + 1) * scale_h).to(tl.int32)
        oh_end = tl.where(
            tl.minimum((oh_end * reciprocal_scale_h).to(tl.int32), IH - 1) <= ih,
            oh_end + 1,
            oh_end,
        )
        oh_end = tl.minimum(oh_end, OH)

    if SAME_W:
        ow_start = iw
        ow_end = iw + 1
    else:
        ow_start = (iw * scale_w).to(tl.int32)
        ow_start = tl.where(
            tl.minimum((ow_start * reciprocal_scale_w).to(tl.int32), IW - 1) < iw,
            ow_start + 1,
            ow_start,
        )
        ow_end = ((iw + 1) * scale_w).to(tl.int32)
        ow_end = tl.where(
            tl.minimum((ow_end * reciprocal_scale_w).to(tl.int32), IW - 1) <= iw,
            ow_end + 1,
            ow_end,
        )
        ow_end = tl.minimum(ow_end, OW)

    gi_stride = nc_stride * IH * IW
    go_stride = nc_stride * OH * OW
    offset_gi = (nc_iter * IH + ih) * IW + iw
    base_go = nc_iter * OH * OW

    while nc_iter < NC:
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for dh in range(MAX_REGION_H):
            if SAME_H:
                oh = oh_start
            else:
                oh = oh_start + dh
            for dw in range(MAX_REGION_W):
                if SAME_W:
                    ow = ow_start
                else:
                    ow = ow_start + dw
                if SAME_H and SAME_W:
                    load_mask = mask
                elif SAME_H:
                    load_mask = mask & (ow < ow_end)
                elif SAME_W:
                    load_mask = mask & (oh < oh_end)
                else:
                    load_mask = mask & (oh < oh_end) & (ow < ow_end)
                go_offset = base_go + oh * OW + ow
                val = tl.load(
                    grad_output_ptr + go_offset,
                    mask=load_mask,
                    other=0.0,
                )
                acc += val.to(tl.float32)
        tl.store(grad_input_ptr + offset_gi, acc, mask=mask)
        offset_gi += gi_stride
        base_go += go_stride
        nc_iter += nc_stride


def upsample_nearest2d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int],
    input_size: Tuple[int, int, int, int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE NEAREST2D BACKWARD")
    assert grad_output.device.type == device
    N, C, IH, IW = input_size
    OH, OW = output_size

    if scales_h is not None:
        reciprocal_scale_h = 1 / scales_h
        scale_h_val = scales_h
    else:
        reciprocal_scale_h = IH / OH
        scale_h_val = OH / IH
    if scales_w is not None:
        reciprocal_scale_w = 1 / scales_w
        scale_w_val = scales_w
    else:
        reciprocal_scale_w = IW / OW
        scale_w_val = OW / IW

    SAME_H = OH == IH
    SAME_W = OW == IW
    MAX_REGION_H = 1 if SAME_H else math.ceil(scale_h_val) + 1
    MAX_REGION_W = 1 if SAME_W else math.ceil(scale_w_val) + 1

    grad_input = torch.empty(
        (N, C, IH, IW), device=grad_output.device, dtype=grad_output.dtype
    )
    total_threads = IH * IW
    grid = lambda META: (
        triton.cdiv(total_threads, META["BLOCK_SIZE"]),
        triton.cdiv(N * C, 4),
    )

    with torch_device_fn.device(grad_output.device):
        upsample_nearest2d_backward_kernel[grid](
            grad_output,
            grad_input,
            N,
            C,
            IH,
            IW,
            OH,
            OW,
            scale_h_val,
            scale_w_val,
            reciprocal_scale_h,
            reciprocal_scale_w,
            MAX_REGION_H=MAX_REGION_H,
            MAX_REGION_W=MAX_REGION_W,
            SAME_H=SAME_H,
            SAME_W=SAME_W,
        )
    return grad_input
