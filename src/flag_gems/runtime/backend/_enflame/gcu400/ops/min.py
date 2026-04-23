import builtins
import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(__name__)

_min = builtins.min


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def min_kernel_inner_1d(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)

    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)

    if tl.constexpr(dtype == tl.float16) or tl.constexpr(dtype == tl.bfloat16):
        acc_dtype = tl.float32
    elif tl.constexpr(dtype == tl.int16) or tl.constexpr(dtype == tl.int8):
        acc_dtype = tl.int32
    else:
        acc_dtype = dtype

    for row_id in tl.range(pid, M, step, num_stages=num_stages):
        min_vals = tl.full([BLOCK_N], value=get_dtype_max(acc_dtype), dtype=acc_dtype)
        min_idxs = tl.zeros([BLOCK_N], dtype=tl.int32)

        base = row_id * N
        for col_off in tl.range(0, N, BLOCK_N):
            n_offset = col_off + tl.arange(0, BLOCK_N)
            mask = n_offset < N
            x = tl.load(inp + base + n_offset, mask=mask, other=max_value).to(acc_dtype)

            update = x < min_vals
            min_vals = tl.where(update, x, min_vals)
            min_idxs = tl.where(update, n_offset, min_idxs)

        final_min = tl.min(min_vals, axis=0)
        eq_mask = min_vals == final_min
        min_idxs_masked = tl.where(eq_mask, min_idxs, 2147483647)
        final_idx = tl.min(min_idxs_masked, axis=0)

        tl.store(out_value + row_id, final_min.to(dtype))
        tl.store(out_index + row_id, final_idx)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K", "num_k_tiles"])
def min_kernel_non_inner(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    num_k_tiles,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    total_work = M * num_k_tiles

    dtype = inp.type.element_ty
    if tl.constexpr(dtype == tl.float16) or tl.constexpr(dtype == tl.bfloat16):
        acc_dtype = tl.float32
    elif tl.constexpr(dtype == tl.int16) or tl.constexpr(dtype == tl.int8):
        acc_dtype = tl.int32
    else:
        acc_dtype = dtype
    max_value = get_dtype_max(acc_dtype)
    orig_max = get_dtype_max(dtype)

    for work_id in tl.range(pid, total_work, num_prog, num_stages=num_stages):
        m = work_id // num_k_tiles
        k_tile = work_id % num_k_tiles
        k_offset = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K

        min_vals = tl.full([BLOCK_K], dtype=acc_dtype, value=max_value)
        min_idxs = tl.zeros([BLOCK_K], dtype=tl.int32)

        base = m * N * K
        for n_off in tl.range(0, N, BLOCK_N):
            n_idx = n_off + tl.arange(0, BLOCK_N)
            offset = base + n_idx[:, None] * K + k_offset[None, :]
            mask = (n_idx[:, None] < N) & k_mask[None, :]
            vals = tl.load(inp + offset, mask=mask, other=orig_max).to(acc_dtype)

            local_min = tl.min(vals, axis=0)
            eq = vals == local_min[None, :]
            idx_matrix = tl.where(eq, n_idx[:, None], 2147483647)
            local_argmin = tl.min(idx_matrix, axis=0)

            update = local_min < min_vals
            min_vals = tl.where(update, local_min, min_vals)
            min_idxs = tl.where(update, local_argmin, min_idxs)

        out_offset = m * K + k_offset
        tl.store(out_value + out_offset, min_vals.to(dtype), k_mask)
        tl.store(out_index + out_offset, min_idxs, k_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def min_kernel_inner_batch(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)

    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)

    if tl.constexpr(dtype == tl.float16) or tl.constexpr(dtype == tl.bfloat16):
        acc_dtype = tl.float32
    elif tl.constexpr(dtype == tl.int16) or tl.constexpr(dtype == tl.int8):
        acc_dtype = tl.int32
    else:
        acc_dtype = dtype

    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tiles, step):
        m_off = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_off < M

        min_vals = tl.full([BLOCK_M], value=get_dtype_max(acc_dtype), dtype=acc_dtype)
        min_idxs = tl.zeros([BLOCK_M], dtype=tl.int32)

        for n in tl.range(0, N, 1):
            offsets = m_off * N + n
            vals = tl.load(inp + offsets, mask=m_mask, other=max_value).to(acc_dtype)
            update = vals < min_vals
            min_vals = tl.where(update, vals, min_vals)
            min_idxs = tl.where(update, n, min_idxs)

        tl.store(out_value + m_off, min_vals.to(dtype), m_mask)
        tl.store(out_index + m_off, min_idxs, m_mask)


def min(inp):
    logger.debug("GEMS MIN GCU400")
    return torch.amin(inp)


def min_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS MIN DIM GCU400")

    return_dtype = inp.dtype
    if inp.dtype == torch.int64:
        inp = inp.to(torch.int32)
    if inp.dtype == torch.int16:
        inp = inp.to(torch.int32)
    if inp.dtype == torch.float64:
        inp = inp.to(torch.float32)

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim

    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    inp = inp.contiguous()

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int32, device=inp.device)

    if K == 1:
        BLOCK_N = _min(triton.next_power_of_2(N), 4096)
        if N <= 64 and M > 1024:
            BLOCK_M = 1024 if inp.dtype in (torch.float16, torch.bfloat16) else 512
            grid_m = _min(triton.cdiv(M, BLOCK_M), 48)
            with torch_device_fn.device(inp.device):
                min_kernel_inner_batch[(grid_m,)](
                    inp, out_value, out_index, M, N,
                    BLOCK_M=BLOCK_M,
                    num_warps=1,
                )
        else:
            grid_m = _min(M, 48)
            num_stages = 3 if M > grid_m else 1
            with torch_device_fn.device(inp.device):
                min_kernel_inner_1d[(grid_m,)](
                    inp, out_value, out_index, M, N,
                    BLOCK_N=BLOCK_N,
                    num_stages=num_stages, num_warps=1,
                )
    else:
        BLOCK_K = _min(triton.next_power_of_2(K), 128)
        max_bn = _min(2048, 32768 // BLOCK_K)
        BLOCK_N = _min(triton.next_power_of_2(N), max_bn)

        num_k_tiles = triton.cdiv(K, BLOCK_K)
        total_work = M * num_k_tiles
        grid_size = _min(total_work, 48)

        num_stages = 1

        with torch_device_fn.device(inp.device):
            min_kernel_non_inner[(grid_size,)](
                inp, out_value, out_index, M, N, K, num_k_tiles,
                BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
                num_stages=num_stages, num_warps=1,
            )

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    Min_out = namedtuple("min", ["values", "indices"])
    out = Min_out(values=out_value.to(return_dtype), indices=out_index.to(torch.int64))
    return out
