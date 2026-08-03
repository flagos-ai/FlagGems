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
    configs=runtime.get_tuned_config("upsample_nearest2d"), key=["N", "C", "OH", "OW"]
)
@triton.heuristics(runtime.get_heuristic_config("upsample_nearest2d"))
@triton.jit
def upsample_nearest2d_backward_kernel(
    ptr_go,
    ptr_gi,
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
    pid = tl.program_id(axis=0)
    nc_stride = tl.num_programs(axis=1)
    NC = N * C
    nc_iter = tl.program_id(axis=1)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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
        grad = tl.load(ptr_go + offset_o).to(tl.float32)
        tl.atomic_add(ptr_gi + offset_i, grad)
        ptr_go += src_index_stride
        nc_iter += nc_stride


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

    grad_input = torch.zeros((N, C, IH, IW), device=grad_output.device, dtype=torch.float32)
    total_threads = OH * OW
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
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
        )

    if grad_input.dtype == grad_output.dtype:
        return grad_input
    result = torch.empty((N, C, IH, IW), device=grad_output.device, dtype=grad_output.dtype)
    with torch_device_fn.device(grad_output.device):
        result.copy_(grad_input)
    return result
