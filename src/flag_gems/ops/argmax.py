import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def argmax_kernel_1(
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
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
    max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
    max_index = max_index + pid * BLOCK_SIZE
    mid_value_ptr = mid_value + pid
    max_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, max_val)
    tl.store(max_index_ptr, max_index)


@libentry()
@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid_value.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    index_val = tl.argmax(mid_val, axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("argmax_non_inner"))
@triton.jit
def argmax_kernel_non_inner(
    inp,
    out_index,
    M,
    N,
    K,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    k_offset = pid_k * TILE_K + tl.arange(0, TILE_K)

    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    min_value = get_dtype_min(cdtype)

    if ONE_TILE_PER_CTA:
        n_offset = tl.arange(0, TILE_N)
        offset = pid_m * N * K + n_offset[:, None] * K + k_offset
        mask = k_offset < K and n_offset[:, None] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
        local_max, local_argmax = tl.max(
            inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
        )
        offset_index = pid_m * K + k_offset
        out_index_ptrs = out_index + offset_index
        mask1 = k_offset < K
        tl.store(out_index_ptrs, local_argmax, mask=mask1)
    else:
        max_values = tl.full([TILE_K], dtype=cdtype, value=min_value)
        argmax_values = tl.full([TILE_K], dtype=tl.int64, value=0)

        for start_n in range(0, N, TILE_N):
            n_offset = start_n + tl.arange(0, TILE_N)
            offset = pid_m * N * K + n_offset[:, None] * K + k_offset
            mask = k_offset < K and n_offset[:, None] < N
            inp_ptrs = inp + offset
            inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
            local_max, local_argmax = tl.max(
                inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
            )
            update = local_max > max_values
            max_values = tl.where(update, local_max, max_values)
            argmax_values = tl.where(update, start_n + local_argmax, argmax_values)
        offset_index = pid_m * K + k_offset
        out_index_ptrs = out_index + offset_index
        mask1 = k_offset < K
        tl.store(out_index_ptrs, argmax_values, mask=mask1)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("argmax_inner"))
@triton.jit
def argmax_kernel_inner(
    inp,
    out_index,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tle.program_id(0)

    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)

    if ONE_TILE_PER_CTA:
        n_offset = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offset
        mask = n_offset < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
        local_max, local_argmax = tl.max(
            inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
        )
        out_index_ptrs = out_index + pid_m
        tl.store(out_index_ptrs, local_argmax)
    else:
        max_values = min_value
        argmax_values = 0

        loop_time = N // TILE_N
        remainder = N % TILE_N
        for start_n in range(0, loop_time):
            n_offset = start_n * TILE_N + tl.arange(0, TILE_N)
            offset = pid_m * N + n_offset
            inp_ptrs = inp + offset
            inp_vals = tl.load(inp_ptrs)
            local_max, local_argmax = tl.max(
                inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
            )
            update = local_max > max_values
            max_values = tl.where(update, local_max, max_values)
            argmax_values = tl.where(
                update, start_n * TILE_N + local_argmax, argmax_values
            )

        if remainder:
            n_offset = loop_time * TILE_N + tl.arange(0, TILE_N)
            offset = pid_m * N + n_offset
            mask = n_offset < N
            inp_ptrs = inp + offset
            inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
            local_max, local_argmax = tl.max(
                inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
            )
            update = local_max > max_values
            max_values = tl.where(update, local_max, max_values)
            argmax_values = tl.where(
                update, loop_time * TILE_N + local_argmax, argmax_values
            )

        out_index_ptrs = out_index + pid_m
        tl.store(out_index_ptrs, argmax_values)


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS ARGMAX")
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
            argmax_kernel_1[(mid_size, 1, 1)](
                inp,
                mid_value,
                mid_index,
                M,
                block_size,
            )
            argmax_kernel_2[(1, 1, 1)](mid_value, mid_index, out, mid_size, block_mid)
        return out
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        shape = inp.shape
        dim = dim % inp.ndim
        if inp.numel() == 0:
            out_shape = list(shape)
            if keepdim:
                out_shape[dim] = 1
            else:
                del out_shape[dim]
            return torch.zeros(out_shape, dtype=torch.int64, device=inp.device)
        N = shape[dim]
        M = math.prod(shape[:dim])
        K = inp.numel() // M // N

        inp = inp.contiguous()

        shape_list = list(shape)
        shape_list[dim] = 1
        out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
        if not keepdim:
            out_index = torch.squeeze(out_index, dim)

        with torch_device_fn.device(inp.device):
            if K > 1:
                grid = lambda meta: (
                    M,
                    triton.cdiv(K, meta["TILE_K"]),
                )
                argmax_kernel_non_inner[grid](
                    inp,
                    out_index,
                    M,
                    N,
                    K,
                )
            else:
                grid = lambda meta: (M, 1, 1)
                argmax_kernel_inner[grid](
                    inp,
                    out_index,
                    M,
                    N,
                )
        return out_index
