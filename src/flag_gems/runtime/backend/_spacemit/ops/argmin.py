import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def argmin_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    max_value = get_dtype_max(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=max_value)
    min_val, min_index = tl.min(inp_val, axis=0, return_indices=True)
    min_index = min_index + pid * BLOCK_SIZE
    tl.store(mid_value + pid, min_val)
    tl.store(mid_index + pid, min_index)


@libentry()
@triton.jit
def argmin_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid_value.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    index_val = tl.argmin(mid_val, axis=0)
    out_val = tl.load(mid_index + index_val)
    tl.store(out, out_val)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("argmin"))
@triton.jit
def argmin_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tle.program_id(1)
    start_row = pid_m * BLOCK_M
    row_offsets = start_row + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < M

    dtype = inp_ptr.dtype.element_ty
    acc_type = (
        tl.float32
        if (dtype is tl.bfloat16 or dtype is tl.float16)
        else (
            tl.int32
            if (dtype is tl.int16 or dtype is tl.int8 or dtype is tl.uint8)
            else dtype
        )
    )
    max_value = get_dtype_max(dtype)
    max_value_acc = get_dtype_max(acc_type)
    row_min = tl.full((BLOCK_M,), max_value_acc, dtype=acc_type)
    row_argmin = tl.full((BLOCK_M,), -1, dtype=tl.int32)

    for block_start in range(0, N, BLOCK_N):
        col_offsets = block_start + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < N
        mask = row_mask[:, None] & col_mask[None, :]
        input_ptrs = (
            inp_ptr + row_offsets[:, None] * N * K + col_offsets[None, :] * K + pid_k
        )
        current_block = tl.load(input_ptrs, mask=mask, other=max_value).to(acc_type)

        block_min = tl.min(current_block, axis=1)
        block_argmin = tl.argmin(current_block, axis=1).to(tl.int32) + block_start

        update_mask = block_min < row_min
        tie_mask = (block_min == row_min) & (
            (row_argmin < 0) | (block_argmin < row_argmin)
        )
        choose_new = update_mask | tie_mask

        row_argmin = tl.where(choose_new, block_argmin, row_argmin)
        row_min = tl.where(update_mask, block_min, row_min)

    out_offsets = row_offsets * K + pid_k
    tl.store(
        out_ptr + out_offsets, row_argmin.to(out_ptr.dtype.element_ty), mask=row_mask
    )


def argmin(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_SPACEMIT ARGMIN")
    if dim is None:
        M = inp.numel()
        if dtype is None:
            dtype = inp.dtype
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid_value = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
        if keepdim:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=torch.int64, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int64, device=inp.device)

        with torch_device_fn.device(inp.device):
            argmin_kernel_1[(mid_size, 1, 1)](
                inp,
                mid_value,
                mid_index,
                M,
                block_size,
            )
            argmin_kernel_2[(1, 1, 1)](mid_value, mid_index, out, mid_size, block_mid)
        return out

    if dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
        )

    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    inp = inp.contiguous()

    shape_list = list(shape)
    shape_list[dim] = 1
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
    if not keepdim:
        out_index = torch.squeeze(out_index, dim)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), K)
    with torch_device_fn.device(inp.device):
        argmin_kernel[grid](
            inp,
            out_index,
            M,
            N,
            K,
        )

    return out_index
