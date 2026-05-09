import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)


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
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
    INTEGER_SCALE: tl.constexpr,
    SCALE_2X: tl.constexpr,
):
    if USE_INT32_IDX:
        pid = tl.program_id(axis=0)
    else:
        pid = tl.program_id(axis=0).to(tl.int64)
    nc_stride = tl.num_programs(axis=1)
    NC = N * C
    nc_iter = tl.program_id(axis=1)

    if INTEGER_SCALE:
        sh = OH // IH
        sw = OW // IW
        total_input = IH * IW
        idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < total_input
        iw = idx % IW
        ih = idx // IW % IH

        if SCALE_2X:
            while nc_iter < NC:
                o_base = nc_iter * OH * OW + ih * 2 * OW + iw * 2
                g_top = tl.load(
                    ptr_grad_output + o_base[:, None] + tl.arange(0, 2)[None, :],
                    mask=mask[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                )
                g_bottom = tl.load(
                    ptr_grad_output + (o_base + OW)[:, None] + tl.arange(0, 2)[None, :],
                    mask=mask[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                )
                grad = g_top[:, 0] + g_top[:, 1] + g_bottom[:, 0] + g_bottom[:, 1]
                i_offset = nc_iter * IH * IW + ih * IW + iw
                tl.store(ptr_grad_input + i_offset, grad, mask=mask)
                nc_iter += nc_stride
        else:
            while nc_iter < NC:
                grad = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
                i_offset = nc_iter * IH * IW + ih * IW + iw
                oh_base = ih * sh
                ow_base = iw * sw
                for hh in range(sh):
                    oh = oh_base + hh
                    o_row_offset = nc_iter * OH * OW + oh * OW + ow_base
                    for ww in range(sw):
                        g = tl.load(
                            ptr_grad_output + o_row_offset + ww,
                            mask=mask,
                            other=0.0,
                            cache_modifier=".cg",
                        )
                        grad += g
                tl.store(ptr_grad_input + i_offset, grad, mask=mask)
                nc_iter += nc_stride
    else:
        total_spatial = OH * OW
        idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < total_spatial
        ow = idx % OW
        oh = idx // OW % OH
        if SAME_H:
            ih = oh
        else:
            ih = tl.minimum((oh * reciprocal_scale_h).to(tl.int32), IH - 1)
        if SAME_W:
            iw = ow
        else:
            iw = tl.minimum((ow * reciprocal_scale_w).to(tl.int32), IW - 1)

        offset_o = (nc_iter * OH + oh) * OW + ow
        offset_i = (nc_iter * IH + ih) * IW + iw
        src_index_stride = nc_stride * OH * OW
        dst_index_stride = nc_stride * IH * IW
        while nc_iter < NC:
            data = tl.load(ptr_grad_output + offset_o, mask=mask, cache_modifier=".cg")
            tl.atomic_add(ptr_grad_input + offset_i, data, mask=mask, sem="relaxed")
            ptr_grad_output += src_index_stride
            ptr_grad_input += dst_index_stride
            nc_iter += nc_stride


def upsample_nearest2d_backward(
    grad_output: torch.Tensor,
    output_size: list,
    input_size: list,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE NEAREST2D BACKWARD")
    assert grad_output.device.type == device
    assert grad_output.ndim == 4, "The ndim of grad_output must be 4"
    N, C, IH, IW = input_size
    OH, OW = output_size
    assert grad_output.shape == (N, C, OH, OW), (
        f"grad_output shape {grad_output.shape} != expected ({N}, {C}, {OH}, {OW})"
    )

    if scales_h is not None:
        reciprocal_scale_h = 1 / scales_h
    else:
        reciprocal_scale_h = IH / OH
    if scales_w is not None:
        reciprocal_scale_w = 1 / scales_w
    else:
        reciprocal_scale_w = IW / OW

    is_integer_scale = (
        OH % IH == 0 and OW % IW == 0 and (OH // IH > 1 or OW // IW > 1)
    )
    if is_integer_scale:
        grad_input = torch.empty((N, C, IH, IW), device=grad_output.device, dtype=grad_output.dtype)
    else:
        grad_input = torch.zeros((N, C, IH, IW), device=grad_output.device, dtype=grad_output.dtype)
    total_threads = (IH * IW) if is_integer_scale else (OH * OW)
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
            reciprocal_scale_h,
            reciprocal_scale_w,
        )
    return grad_input
