import logging
import math

import torch
import triton

from flag_gems.ops.topk import (
    topk_single_stage_kernel,
    topk_stage1_kernel,
    topk_stage2_kernel,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f"flag_gems.runtime.backend._mthreads.ops.{__name__.split('.')[-1]}"
)


def topk(x, k, dim=-1, largest=True, sorted=True):
    # Note: The shared topk implementation gates a radix-based TLE kernel
    # (`topk_kernel_radix_tle`) behind `x.is_cuda`. On mthreads the device is
    # `musa` (so `x.is_cuda` is False), yet the gate evaluates the same as the
    # other conditions. When this radix kernel is selected on mthreads hardware
    # (e.g. for `(N=256, k=256)` style inputs) it produces `-inf` garbage in the
    # trailing output positions, breaking accuracy. The mthreads override
    # therefore skips the radix TLE path entirely and routes to the same
    # single-stage / two-stage kernels that the shared implementation uses for
    # the non-radix case, which are correct on mthreads.
    logger.debug("GEMS_MTHREADS TOPK")
    # If dim equals to last dim, we set it to -1.
    if dim < 0:
        dim = dim + x.ndim

    assert dim == x.ndim - 1, "Currently only support topk in last dimension"

    # Early return for k=0 to avoid Triton kernel compilation error.
    # Triton's tl.arange(0, BLOCK_SIZE) requires BLOCK_SIZE > 0.
    # When k=0, stage2_elem_cnt becomes 0, leading to BLOCK_SIZE=0.
    if k == 0:
        out_shape = list(x.shape[:-1]) + [0]
        return (
            torch.empty(out_shape, device=x.device, dtype=x.dtype),
            torch.empty(out_shape, device=x.device, dtype=torch.int64),
        )

    descending = True
    if not largest:
        descending = False

    topk_elem_cnt = x.shape[dim]
    batch_size = math.prod(x.shape) // topk_elem_cnt

    if topk_elem_cnt < 4096:
        out_shape = x.shape[:-1] + (k,)

        y_vals = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        y_idx = torch.empty(out_shape, device=x.device, dtype=torch.int64)

        BLOCK_SIZE = triton.next_power_of_2(topk_elem_cnt)

        with torch_device_fn.device(x.device):
            topk_single_stage_kernel[(batch_size,)](
                y_vals,
                y_idx,
                x,
                k,
                topk_elem_cnt,
                BLOCK_SIZE,
                descending,
            )
        return (y_vals, y_idx)

    # Note(Zhengzekang): Maybe we should add a heuristic search in selecting a proper chunk size.
    if topk_elem_cnt < 1024:
        chunk_size = 256
    else:
        chunk_size = 1024

    # Note(Zhengzekang): We should promise chunk_size is larger than k.
    if chunk_size < k:
        chunk_size = triton.next_power_of_2(k)

    chunk_num = triton.cdiv(topk_elem_cnt, chunk_size)

    stage1_out = torch.empty(batch_size * chunk_num * k, device=x.device, dtype=x.dtype)
    stage1_out_idx = torch.empty(
        batch_size * chunk_num * k, device=x.device, dtype=torch.int64
    )

    out_shape = x.shape[:-1] + (k,)
    stage2_out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    stage2_out_idx = torch.empty(out_shape, device=x.device, dtype=torch.int64)

    with torch_device_fn.device(x.device):
        topk_stage1_kernel[
            batch_size,
            chunk_num,
        ](
            stage1_out,  # pointer to the output
            stage1_out_idx,  # pointer to the output
            x,  # pointer to the input
            k,
            topk_elem_cnt,
            chunk_size,
            descending,
        )
    stage2_elem_cnt = chunk_num * k
    BLOCK_SIZE = triton.next_power_of_2(stage2_elem_cnt)

    with torch_device_fn.device(x.device):
        topk_stage2_kernel[batch_size,](
            stage2_out,
            stage2_out_idx,
            stage1_out,
            stage1_out_idx,
            dim,
            k,
            stage2_elem_cnt,
            BLOCK_SIZE,
            descending,
        )

    return (stage2_out, stage2_out_idx)
