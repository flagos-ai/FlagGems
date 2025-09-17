import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn

from ..utils import MAX_GRID_SIZE_X, TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
device = device.name


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
):
    pid = tl.program_id(axis=0) + tl.program_id(axis=1) * tl.num_programs(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ow = idx % OW
    oh = idx // OW % OH
    c = idx // OW // OH % C
    n = idx // OW // OH // C % N
    if SAME_H:
        ih = oh
    else:
        # tl.floor() cannot be found in 2.3.1, using int trunc
        ih = tl.minimum((oh * reciprocal_scale_h).to(tl.int32), IH - 1)
    if SAME_W:
        iw = ow
    else:
        iw = tl.minimum((ow * reciprocal_scale_w).to(tl.int32), IW - 1)
    offset_o = ((n * C + c) * OH + oh) * OW + ow
    offset_i = ((n * C + c) * IH + ih) * IW + iw
    data = tl.load(ptr_i + offset_i)
    tl.store(ptr_o + offset_o, data)


def configs2():
    block_h = [8, 16, 32, 64, 128, 256]
    num_stage = [1, 3]
    return [
        triton.Config({"BLOCK_H": bh}, num_warps=1, num_stages=s)
        for s in num_stage
        for bh in block_h
    ]


@triton.autotune(configs=configs2(), key=["N", "C", "OH", "OW"])
@triton.jit
def upsample_nearest2d_kernel_opt(
    ptr_o,
    ptr_i,
    N,
    C,
    OH,
    OW: tl.constexpr,
    IH,
    IW: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)

    nc_nums_per_job = (N * C + num_jobs - 1) // num_jobs
    nc_begin = pid * nc_nums_per_job
    nc_end = min(nc_begin + nc_nums_per_job, N * C)

    loop_num_h = (OH + BLOCK_H - 1) // BLOCK_H
    for idx in range((nc_end - nc_begin) * loop_num_h):
        nc_idx = nc_begin + (idx // loop_num_h)
        h_idx = (idx % loop_num_h) * BLOCK_H

        init_out = nc_idx * OH * OW
        init_in = nc_idx * IH * IW

        ih = h_idx // 2 + tl.arange(0, BLOCK_H // 2)
        iw = tl.arange(0, IW)
        offset_i = init_in + ih[:, None] * IW + iw

        oh = h_idx + tl.arange(0, BLOCK_H)
        ow = tl.arange(0, OW)
        offset_o = init_out + oh[:, None] * OW + ow

        data = tl.load(ptr_i + offset_i, mask=(ih[:, None] < IH))

        tmp = (
            data.reshape(BLOCK_H // 2, OW // 2, 1)
            .broadcast_to(BLOCK_H // 2, OW // 2, 2)
            .reshape(BLOCK_H // 2, 1, OW)
        )
        tmp1 = tmp.broadcast_to(BLOCK_H // 2, 2, OW).reshape(BLOCK_H, OW)

        tl.store(ptr_o + offset_o, tmp1, mask=(oh[:, None] < OH))


@triton.autotune(configs=configs2(), key=["N", "C", "OH", "OW"])
@triton.jit
def upsample_nearest2d_kernel_opt_tile_h(
    ptr_o,
    ptr_i,
    N,
    C,
    OH,
    OW: tl.constexpr,
    IH,
    IW: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)

    start = pid * BLOCK_H
    step = BLOCK_H * num_jobs
    loop_num_h = (OH - start + step - 1) // step

    for idx in range(N * C * loop_num_h):
        nc_idx = idx // loop_num_h
        h_idx = (idx % loop_num_h) * step + start

        init_out = nc_idx * OH * OW
        init_in = nc_idx * IH * IW

        ih = h_idx // 2 + tl.arange(0, BLOCK_H // 2)
        iw = tl.arange(0, IW)
        offset_i = init_in + ih[:, None] * IW + iw

        oh = h_idx + tl.arange(0, BLOCK_H)
        ow = tl.arange(0, OW)
        offset_o = init_out + oh[:, None] * OW + ow

        data = tl.load(ptr_i + offset_i, mask=(ih[:, None] < IH))

        tmp = (
            data.reshape(BLOCK_H // 2, OW // 2, 1)
            .broadcast_to(BLOCK_H // 2, OW // 2, 2)
            .reshape(BLOCK_H // 2, 1, OW)
        )
        tmp1 = tmp.broadcast_to(BLOCK_H // 2, 2, OW).reshape(BLOCK_H, OW)

        tl.store(ptr_o + offset_o, tmp1, mask=(oh[:, None] < OH))


def upsample_nearest2d(
    input: torch.Tensor,
    output_size: Tuple[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_CAMBRICON UPSAMPLE NEAREST2D")
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

    with torch_device_fn.device(input.device):
        if (
            reciprocal_scale_h == 0.5
            and reciprocal_scale_w == 0.5
            and IH / OH == 0.5
            and IW / OW == 0.5
        ):
            if N * C > 48:
                upsample_nearest2d_kernel_opt[TOTAL_CORE_NUM,](
                    output, input, N, C, OH, OW, IH, IW
                )
            else:
                upsample_nearest2d_kernel_opt_tile_h[TOTAL_CORE_NUM,](
                    output, input, N, C, OH, OW, IH, IW
                )
        else:
            total_threads = N * C * OH * OW

            # incase grid check error
            def grid_fn(META):
                num_threads = triton.cdiv(total_threads, META["BLOCK_SIZE"])
                grid_x = min(num_threads, MAX_GRID_SIZE_X)
                grid_y = triton.cdiv(num_threads, grid_x)
                return (
                    grid_x,
                    grid_y,
                )

            upsample_nearest2d_kernel[grid_fn](
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
