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

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


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
    mid_value_ptr = mid_value + pid
    min_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, min_val)
    tl.store(min_index_ptr, min_index)


@libentry()
@triton.jit
def argmin_kernel_2(
    mid_value,
    mid_index,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid_value.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    index_val = tl.argmin(mid_val, axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("argmin"))
@triton.jit
def argmin_kernel(
    inp,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tle.program_id(0)
    # pid_k = tle.program_id(1)
    for pid_k in range(K):
        m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

        dtype = inp.type.element_ty
        acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
        max_value = get_dtype_max(dtype)
        min_values = tl.full([BLOCK_M], dtype=acc_type, value=max_value)
        argmin_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
        for start_n in range(0, N, BLOCK_N):
            n_offset = start_n + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
            mask = m_offset[:, None] < M and n_offset[None, :] < N
            inp_ptrs = inp + offset
            inp_vals = tl.load(inp_ptrs, mask=mask, other=max_value)
            # tl.bfloat is promoted to tl.float32 by tl.min
            local_min, local_argmin = tl.min(
                inp_vals, 1, return_indices=True, return_indices_tie_break_left=True
            )
            # if return indices is not supported, call a tl.argmin in addition
            # local_argmin = tl.argmin(inp_vals, 1)
            update = local_min < min_values
            min_values = tl.where(update, local_min, min_values)
            argmin_values = tl.where(update, start_n + local_argmin, argmin_values)

        offset_index = m_offset * K + pid_k
        out_index_ptrs = out_index + offset_index
        mask1 = m_offset < M
        tl.store(out_index_ptrs, argmin_values, mask=mask1)


def argmin(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ASCEND ARGMIN")
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
            argmin_kernel_2[(1, 1, 1)](
                mid_value,
                mid_index,
                out,
                mid_size,
                block_mid,
            )
        return out
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
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

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            # K,
        )
        with torch_device_fn.device(inp.device):
            argmin_kernel[grid](
                inp,
                out_index,
                M,
                N,
                K,
            )

        return out_index
