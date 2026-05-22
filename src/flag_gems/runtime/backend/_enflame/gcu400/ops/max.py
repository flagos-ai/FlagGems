import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def max_kernel_1_simple(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp + offset, mask=mask, other=min_value)
    tl.store(mid + pid, tl.max(inp_val))


@libentry()
@triton.jit(do_not_specialize=["M"])
def max_kernel_1_grid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    dtype = inp.type.element_ty
    min_val = get_dtype_min(dtype)

    acc = tl.full([BLOCK_SIZE], value=min_val, dtype=dtype)

    block_id = pid
    while block_id * BLOCK_SIZE < M:
        offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        vals = tl.load(inp + offset, mask=mask, other=min_val)
        acc = tl.where(vals > acc, vals, acc)
        block_id += num_progs

    tl.store(mid + pid, tl.max(acc))


@libentry()
@triton.jit
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def max_kernel_inner_1d(
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
    min_value = get_dtype_min(dtype)

    if tl.constexpr(dtype == tl.float16) or tl.constexpr(dtype == tl.bfloat16):
        acc_dtype = tl.float32
    elif tl.constexpr(dtype == tl.int16) or tl.constexpr(dtype == tl.int8):
        acc_dtype = tl.int32
    else:
        acc_dtype = dtype

    for row_id in tl.range(pid, M, step, num_stages=num_stages):
        max_vals = tl.full([BLOCK_N], value=get_dtype_min(acc_dtype), dtype=acc_dtype)
        max_idxs = tl.zeros([BLOCK_N], dtype=tl.int32)

        base = row_id * N
        for col_off in tl.range(0, N, BLOCK_N):
            n_offset = col_off + tl.arange(0, BLOCK_N)
            mask = n_offset < N
            x = tl.load(inp + base + n_offset, mask=mask, other=min_value).to(acc_dtype)

            update = x > max_vals
            max_vals = tl.where(update, x, max_vals)
            max_idxs = tl.where(update, n_offset, max_idxs)

        final_max = tl.max(max_vals, axis=0)
        eq_mask = max_vals == final_max
        max_idxs_masked = tl.where(eq_mask, max_idxs, 2147483647)
        final_idx = tl.min(max_idxs_masked, axis=0)

        tl.store(out_value + row_id, final_max.to(dtype))
        tl.store(out_index + row_id, final_idx)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K", "num_k_tiles"])
def max_kernel_non_inner(
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
    min_value = get_dtype_min(acc_dtype)
    orig_min = get_dtype_min(dtype)

    for work_id in tl.range(pid, total_work, num_prog, num_stages=num_stages):
        m = work_id // num_k_tiles
        k_tile = work_id % num_k_tiles
        k_offset = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K

        max_vals = tl.full([BLOCK_K], dtype=acc_dtype, value=min_value)
        max_idxs = tl.zeros([BLOCK_K], dtype=tl.int32)

        base = m * N * K
        for n_off in tl.range(0, N, BLOCK_N):
            n_idx = n_off + tl.arange(0, BLOCK_N)
            offset = base + n_idx[:, None] * K + k_offset[None, :]
            mask = (n_idx[:, None] < N) & k_mask[None, :]
            vals = tl.load(inp + offset, mask=mask, other=orig_min).to(acc_dtype)

            local_max = tl.max(vals, axis=0)
            eq = vals == local_max[None, :]
            idx_matrix = tl.where(eq, n_idx[:, None], 2147483647)
            local_argmax = tl.min(idx_matrix, axis=0)

            update = local_max > max_vals
            max_vals = tl.where(update, local_max, max_vals)
            max_idxs = tl.where(update, local_argmax, max_idxs)

        out_offset = m * K + k_offset
        tl.store(out_value + out_offset, max_vals.to(dtype), k_mask)
        tl.store(out_index + out_offset, max_idxs, k_mask)


def max(inp):
    logger.debug("GEMS MAX")
    return_dtype = inp.dtype
    if inp.dtype == torch.int64:
        inp = inp.to(torch.int32)
    if inp.dtype == torch.float64:
        inp = inp.to(torch.float32)

    inp = inp.contiguous()
    M = inp.numel()
    dtype = inp.dtype

    if M <= 10 * 1024 * 1024:
        bsize = 65536
        num_programs = min(triton.cdiv(M, bsize), 48)
        block_mid = triton.next_power_of_2(num_programs)
        mid = torch.empty((num_programs,), dtype=dtype, device=inp.device)
        out = torch.empty([], dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            max_kernel_1_grid[(num_programs,)](
                inp, mid, M, BLOCK_SIZE=bsize, num_warps=1
            )
            max_kernel_2[(1,)](mid, out, num_programs, block_mid, num_warps=1)
    else:
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        out = torch.empty([], dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            max_kernel_1_simple[(mid_size, 1, 1)](inp, mid, M, block_size, num_warps=1)
            max_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid, num_warps=1)
    return out.to(return_dtype)


def max_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS MAX DIM")

    return_dtype = inp.dtype
    if inp.dtype == torch.int64:
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
        BLOCK_N = min(triton.next_power_of_2(N), 4096)
        grid_m = min(M, 48)
        num_stages = 3 if M > grid_m else 1
        with torch_device_fn.device(inp.device):
            max_kernel_inner_1d[(grid_m,)](
                inp,
                out_value,
                out_index,
                M,
                N,
                BLOCK_N=BLOCK_N,
                num_stages=num_stages,
                num_warps=1,
            )
    else:
        BLOCK_K = min(triton.next_power_of_2(K), 128)
        BLOCK_N = min(triton.next_power_of_2(N), 256)

        num_k_tiles = triton.cdiv(K, BLOCK_K)
        total_work = M * num_k_tiles
        grid_size = min(total_work, 48)

        with torch_device_fn.device(inp.device):
            max_kernel_non_inner[(grid_size,)](
                inp,
                out_value,
                out_index,
                M,
                N,
                K,
                num_k_tiles,
                BLOCK_K=BLOCK_K,
                BLOCK_N=BLOCK_N,
                num_stages=1,
                num_warps=1,
            )

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    Max_out = namedtuple("max", ["values", "indices"])
    out = Max_out(values=out_value.to(return_dtype), indices=out_index.to(torch.int64))
    return out
