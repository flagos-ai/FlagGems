import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_nearest2d"),
    key=["N", "C", "OH", "OW"],
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
    pid_spatial = tl.program_id(axis=0)
    pid_nc = tl.program_id(axis=1)

    if USE_INT32_IDX:
        nc = pid_nc
    else:
        nc = pid_nc.to(tl.int64)

    NC = N * C
    if nc >= NC:
        return

    idx = pid_spatial * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < (OH * OW)

    ow = idx % OW
    oh = idx // OW

    base_o = nc * OH * OW
    base_i = nc * IH * IW

    if (
        (not SAME_H)
        and (not SAME_W)
        and reciprocal_scale_h == 0.5
        and reciprocal_scale_w == 0.5
    ):
        ih = oh >> 1
        iw = ow >> 1
    else:
        if SAME_H:
            ih = oh
        else:
            ih = tl.minimum((oh * reciprocal_scale_h).to(tl.int32), IH - 1)

        if SAME_W:
            iw = ow
        else:
            iw = tl.minimum((ow * reciprocal_scale_w).to(tl.int32), IW - 1)

    offset_o = base_o + oh * OW + ow
    offset_i = base_i + ih * IW + iw

    data = tl.load(ptr_i + offset_i, mask=mask, other=0)
    tl.store(ptr_o + offset_o, data, mask=mask)


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

    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)

    total_spatial = OH * OW
    total_nc = N * C

    grid = lambda META: (
        triton.cdiv(total_spatial, META["BLOCK_SIZE"]),
        total_nc,
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
@triton.heuristics(runtime.get_heuristic_config("upsample_nearest2d_backward"))
@triton.jit
def upsample_nearest2d_backward_kernel(
    ptr_grad_o,
    ptr_grad_i,
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
    pid_spatial = tl.program_id(axis=0)  # IH * IW
    pid_nc = tl.program_id(axis=1)  # N * C

    nc = pid_nc if USE_INT32_IDX else pid_nc.to(tl.int64)
    if nc >= N * C:
        return

    offs = pid_spatial * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < (IH * IW)

    iw = offs % IW
    ih = offs // IW

    base_o = nc * OH * OW
    base_i = nc * IH * IW

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    if SAME_H:
        oh0 = ih
        oh1 = ih + 1
    else:
        oh0 = tl.minimum((ih * reciprocal_scale_h).to(tl.int32), OH - 1)
        oh1 = tl.minimum(((ih + 1) * reciprocal_scale_h).to(tl.int32), OH)

    if SAME_W:
        ow0 = iw
        ow1 = iw + 1
    else:
        ow0 = tl.minimum((iw * reciprocal_scale_w).to(tl.int32), OW - 1)
        ow1 = tl.minimum(((iw + 1) * reciprocal_scale_w).to(tl.int32), OW)

    MAX_K: tl.constexpr = 4

    for dh in tl.static_range(0, MAX_K):
        oh = oh0 + dh
        hmask = oh < oh1
        for dw in tl.static_range(0, MAX_K):
            ow = ow0 + dw
            wmask = ow < ow1
            valid = mask & hmask & wmask
            offset_o = base_o + oh * OW + ow
            grad = tl.load(ptr_grad_o + offset_o, mask=valid, other=0.0)
            acc += grad

    offset_i = base_i + ih * IW + iw
    tl.store(ptr_grad_i + offset_i, acc, mask=mask)


def upsample_nearest2d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int],
    input_size: Tuple[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE NEAREST2D BACKWARD")
    assert grad_output.device.type == device
    assert grad_output.ndim == 4, "The ndim of grad_output must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    assert len(input_size) == 4, "The len of input_size must be 4"

    OH, OW = output_size
    N, C, IH, IW = input_size

    if scales_h is not None:
        reciprocal_scale_h = 1 / scales_h
    else:
        reciprocal_scale_h = IH / OH

    if scales_w is not None:
        reciprocal_scale_w = 1 / scales_w
    else:
        reciprocal_scale_w = IW / OW

    grad_input = torch.zeros(
        (N, C, IH, IW), device=grad_output.device, dtype=grad_output.dtype
    )

    total_spatial = OH * OW
    total_nc = N * C

    grid = lambda META: (
        triton.cdiv(total_spatial, META["BLOCK_SIZE"]),
        total_nc,
    )

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
        )

    return grad_input
