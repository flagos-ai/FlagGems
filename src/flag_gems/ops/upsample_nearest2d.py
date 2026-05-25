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


@triton.jit
def upsample_nearest2d_backward_kernel(
    ptr_go,
    ptr_gi,
    n_elements_grad_out,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements_grad_out

    # Decode flat index into (n, c, oh, ow)
    ow = idx % OW
    tmp = idx // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    nc = tmp2

    # Map output position to input position (nearest neighbor)
    ih = tl.minimum((oh * reciprocal_scale_h).to(tl.int32), IH - 1)
    iw = tl.minimum((ow * reciprocal_scale_w).to(tl.int32), IW - 1)

    # Input flat index
    offset_gi = (nc * IH + ih) * IW + iw

    grad = tl.load(ptr_go + idx, mask=mask)
    tl.atomic_add(ptr_gi + offset_gi, grad, mask=mask)


def upsample_nearest2d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int],
    input_size: Tuple[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE NEAREST2D BACKWARD")
    assert grad_output.ndim == 4
    assert len(output_size) == 2
    assert len(input_size) == 4
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

    grad_output = grad_output.contiguous()
    grad_input = torch.zeros(
        (N, C, IH, IW), device=grad_output.device, dtype=grad_output.dtype
    )
    n_elements = grad_output.numel()
    if n_elements == 0:
        return grad_input

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(grad_output.device):
        upsample_nearest2d_backward_kernel[grid](
            grad_output,
            grad_input,
            n_elements,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return grad_input


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
