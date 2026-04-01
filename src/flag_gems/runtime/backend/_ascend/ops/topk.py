import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.topk import _get_finfo_val
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils import CORE_NUM

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@libentry()
@triton.jit()
def topk_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    k,
    batch_size,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    BATCHES_PER_CORE: tl.constexpr,
):
    pid = tl.program_id(0)

    for batch_offset in range(BATCHES_PER_CORE):
        cur_batch = pid * BATCHES_PER_CORE + batch_offset
        if cur_batch < batch_size:
            x_base = x_ptr + cur_batch * N
            y_base = y_ptr + cur_batch * k
            idx_base = index_ptr + cur_batch * k

            cols = tl.arange(0, BLOCK_SIZE)
            mask = cols < N

            mask_val = _get_finfo_val(x_ptr.dtype.element_ty, return_max=not DESCENDING)
            x_val = tl.load(x_base + cols, mask=mask, other=mask_val).to(tl.float32)

            cols_f = cols.to(tl.float32)

            for k_idx in range(k):
                if DESCENDING:
                    select_val = tl.max(x_val)
                    select_idx = tl.argmax(x_val, axis=0)
                else:
                    select_val = tl.min(x_val)
                    select_idx = tl.argmin(x_val, axis=0)

                tl.store(y_base + k_idx, select_val)
                tl.store(idx_base + k_idx, select_idx)

                select_idx_f = select_idx.to(tl.float32)
                if DESCENDING:
                    x_val = tl.where(
                        cols_f == select_idx_f,
                        _get_finfo_val(tl.float32, return_max=False),
                        x_val,
                    )
                else:
                    x_val = tl.where(
                        cols_f == select_idx_f,
                        _get_finfo_val(tl.float32, return_max=True),
                        x_val,
                    )


@libentry()
@triton.jit()
def topk_large_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    k,
    batch_size,
    N,
    BLOCK_SUB: tl.constexpr,
    DESCENDING: tl.constexpr,
    BATCHES_PER_CORE: tl.constexpr,
):
    pid = tl.program_id(0)

    for batch_offset in range(BATCHES_PER_CORE):
        cur_batch = pid * BATCHES_PER_CORE + batch_offset
        if cur_batch < batch_size:
            x_base = x_ptr + cur_batch * N
            y_base = y_ptr + cur_batch * k
            idx_base = index_ptr + cur_batch * k

            for k_idx in range(k):
                best_val = _get_finfo_val(
                    x_ptr.dtype.element_ty, return_max=not DESCENDING
                )
                best_val = best_val.to(tl.float32)
                best_idx = tl.zeros([], dtype=tl.int64)

                num_blocks = tl.cdiv(N, BLOCK_SUB)
                for blk in range(num_blocks):
                    blk_offset = blk * BLOCK_SUB
                    cols = tl.arange(0, BLOCK_SUB)
                    offsets = blk_offset + cols
                    mask = offsets < N

                    mask_val = _get_finfo_val(
                        x_ptr.dtype.element_ty, return_max=not DESCENDING
                    )
                    x_val = tl.load(
                        x_base + offsets, mask=mask, other=mask_val
                    ).to(tl.float32)

                    if DESCENDING:
                        local_val = tl.max(x_val)
                        local_idx = tl.argmax(x_val, axis=0)
                        if local_val > best_val:
                            best_val = local_val
                            best_idx = (blk_offset + local_idx).to(tl.int64)
                    else:
                        local_val = tl.min(x_val)
                        local_idx = tl.argmin(x_val, axis=0)
                        if local_val < best_val:
                            best_val = local_val
                            best_idx = (blk_offset + local_idx).to(tl.int64)

                tl.store(y_base + k_idx, best_val)
                tl.store(idx_base + k_idx, best_idx)

                # Invalidate the selected element for next iteration
                if DESCENDING:
                    inv_val = _get_finfo_val(tl.float32, return_max=False)
                else:
                    inv_val = _get_finfo_val(tl.float32, return_max=True)
                tl.store(x_base + best_idx, inv_val)


def topk(x, k, dim=-1, largest=True, sorted=True):
    logger.debug("GEMS_ASCEND TOPK")
    if dim < 0:
        dim = dim + x.ndim

    assert dim == x.ndim - 1, "Currently only support topk in last dimension"

    descending = largest

    topk_elem_cnt = x.shape[dim]
    batch_size = math.prod(x.shape) // topk_elem_cnt

    out_shape = x.shape[:-1] + (k,)
    out_val = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    out_idx = torch.empty(out_shape, device=x.device, dtype=torch.int64)

    # Determine grid size respecting coreDim <= 65535
    ncore = min(batch_size, CORE_NUM)
    batches_per_core = triton.cdiv(batch_size, ncore)

    # UB-safe threshold: float32 = 4 bytes, UB = 192KB
    # With overhead, use 32768 as max BLOCK_SIZE
    MAX_BLOCK = 32768

    if topk_elem_cnt <= MAX_BLOCK:
        BLOCK_SIZE = triton.next_power_of_2(topk_elem_cnt)
        with torch_device_fn.device(x.device):
            topk_kernel[ncore,](
                out_val,
                out_idx,
                x,
                k,
                batch_size,
                topk_elem_cnt,
                BLOCK_SIZE,
                descending,
                batches_per_core,
            )
    else:
        # For large N, use sub-blocking to avoid UB overflow
        x_work = x.contiguous().clone()
        BLOCK_SUB = 8192
        with torch_device_fn.device(x.device):
            topk_large_kernel[ncore,](
                out_val,
                out_idx,
                x_work,
                k,
                batch_size,
                topk_elem_cnt,
                BLOCK_SUB,
                descending,
                batches_per_core,
            )

    return (out_val, out_idx)
