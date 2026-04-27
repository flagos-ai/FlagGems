import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def max_kernel_1(inp, mid, M, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr = 1):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_tile = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, num_prog, num_stages=num_stages):
        offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        inp_ptrs = inp + offset
        mask = offset < M
        min_value = get_dtype_min(inp.type.element_ty)
        if inp.type.element_ty == tl.int64:
            min_value = 0
        inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
        max_val = tl.max(inp_val)
        mid_ptr = mid + tile_id
        tl.store(mid_ptr, max_val)


@libentry()
@triton.jit
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid.type.element_ty)
    if mid.type.element_ty == tl.int64:
        min_value = 0
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 2048:
        return False
    if BLOCK_M * BLOCK_N >= 256 * 1024:
        return False
    return True


@libentry()
@libtuner(
    configs=list(filter(keep, runtime.get_tuned_config("naive_reduction"))),
    key=["M", "N"],
)
@triton.jit
def max_kernel_dim_low(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 4096,
    num_stages: tl.constexpr = 2,
):
    # set offset
    pid_m = tl.program_id(0)
    step = tl.num_programs(0)
    # tl.device_print("max_kernel_dim_low")
    num_tile = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id_m in tl.range(pid_m, num_tile, step):
        m_offset = tile_id_m * BLOCK_M + tl.arange(0, BLOCK_M)
        dtype = inp.type.element_ty
        min_value = get_dtype_min(dtype)
        if inp.type.element_ty == tl.int64:
            min_value = 0
        # result_value = tl.full([BLOCK_M], value=min_value, dtype=acc_type)
        # result_index = tl.zeros([BLOCK_M], dtype=tl.int32)
        n_offset_0 = tl.arange(0, BLOCK_N)
        offset_0 = m_offset[:, None] * N + n_offset_0[None, :]
        # set mask
        mask_0 = m_offset[:, None] < M and n_offset_0[None, :] < N
        inp_ptrs_0 = inp + offset_0
        inp_vals_0 = tl.load(inp_ptrs_0, mask=mask_0, other=min_value)
        result_value, result_index = tl.max(inp_vals_0, axis=1, return_indices=True)
        # tl.device_print("test1")
        # for i in tl.range(BLOCK_N, N, BLOCK_N, num_stages=num_stages):
        #     tl.device_print("test")
        if N > BLOCK_N:
            for i in tl.range(BLOCK_N, N, BLOCK_N, num_stages=num_stages):
                n_offset = i + tl.arange(0, BLOCK_N)
                offset = m_offset[:, None] * N + n_offset[None, :]
                # set mask
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                inp_ptrs = inp + offset
                inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
                max_value, max_index = tl.max(inp_vals, axis=1, return_indices=True)
                update_mask = max_value > result_value
                result_value = tl.where(update_mask, max_value, result_value)
                result_index = tl.where(update_mask, i + max_index, result_index)
        mask1 = m_offset < M
        offset_index = m_offset
        out_value_ptrs = out_value + offset_index
        out_index_ptrs = out_index + offset_index

        tl.store(out_value_ptrs, result_value, mask=mask1)
        tl.store(out_index_ptrs, result_index, mask=mask1)


@libentry()
@libtuner(
    configs=list(filter(keep, runtime.get_tuned_config("naive_reduction"))),
    key=["M", "N"],
)
@triton.jit
def max_kernel_dim_high(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    num_stages: tl.constexpr = 3,
):
    # set offset
    pid_n = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (N + BLOCK_N - 1) // BLOCK_N
    for tile_id_n in tl.range(pid_n, num_tile, step):
        n_offset = tile_id_n * BLOCK_N + tl.arange(0, BLOCK_N)

        dtype = inp.type.element_ty
        min_value = get_dtype_min(dtype)
        if inp.type.element_ty == tl.int64:
            min_value = 0
        # result_index = tl.zeros([BLOCK_N], dtype=tl.int32)
        m_offset_0 = tl.arange(0, BLOCK_M)
        offset_0 = m_offset_0[:, None] * N + n_offset[None, :]
        # set mask
        mask_0 = m_offset_0[:, None] < M and n_offset[None, :] < N
        inp_ptrs_0 = inp + offset_0
        inp_vals_0 = tl.load(inp_ptrs_0, mask=mask_0, other=min_value)
        result_value, result_index = tl.max(inp_vals_0, axis=0, return_indices=True)
        if M > BLOCK_M:
            for i in tl.range(BLOCK_M, M, BLOCK_M, num_stages=num_stages):
                # tl.device_print("test22")
                m_offset = i + tl.arange(0, BLOCK_M)
                offset = m_offset[:, None] * N + n_offset[None, :]
                # set mask
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                inp_ptrs = inp + offset
                inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
                max_value, max_index = tl.max(inp_vals, axis=0, return_indices=True)
                update_mask = max_value > result_value
                result_value = tl.where(update_mask, max_value, result_value)
                result_index = tl.where(update_mask, i + max_index, result_index)
        mask1 = n_offset < N
        offset_index = n_offset
        out_value_ptrs = out_value + offset_index
        out_index_ptrs = out_index + offset_index

        tl.store(out_value_ptrs, result_value, mask=mask1)
        tl.store(out_index_ptrs, result_index, mask=mask1)


@libentry()
@libtuner(
    configs=list(filter(keep, runtime.get_tuned_config("naive_reduction"))),
    key=["M", "N"],
)
@triton.jit
def max_kernel_dim_mid(
    inpIn,
    out_value_in,
    out_index_in,
    B,
    M,
    N,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 4,
    num_stages: tl.constexpr = 2,
):
    pid_b = tl.program_id(1)
    pid_n = tl.program_id(0)
    step = tl.num_programs(1)
    for tile_id_b in tl.range(pid_b, B, step):
        b_offset = tile_id_b * M * N
        inp = inpIn + b_offset
        n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        out_value = out_value_in + tile_id_b * N
        out_index = out_index_in + tile_id_b * N
        dtype = inpIn.type.element_ty
        min_value = get_dtype_min(dtype)
        if inpIn.type.element_ty == tl.int64:
            min_value = 0
        # result_index = tl.zeros([BLOCK_N], dtype=tl.int32)
        m_offset_0 = tl.arange(0, BLOCK_M)
        offset_0 = m_offset_0[:, None] * N + n_offset[None, :]
        # set mask
        mask_0 = m_offset_0[:, None] < M and n_offset[None, :] < N
        inp_ptrs_0 = inp + offset_0
        inp_vals_0 = tl.load(inp_ptrs_0, mask=mask_0, other=min_value)
        result_value, result_index = tl.max(inp_vals_0, axis=0, return_indices=True)
        if M > BLOCK_M:
            for i in tl.range(BLOCK_M, M, BLOCK_M, num_stages=num_stages):
                m_offset = i + tl.arange(0, BLOCK_M)
                offset = m_offset[:, None] * N + n_offset[None, :]
                # set mask
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                inp_ptrs = inp + offset
                inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
                max_value, max_index = tl.max(inp_vals, axis=0, return_indices=True)
                update_mask = max_value > result_value
                result_value = tl.where(update_mask, max_value, result_value)
                result_index = tl.where(update_mask, i + max_index, result_index)
        mask1 = n_offset < N
        offset_index = n_offset
        out_value_ptrs = out_value + offset_index
        out_index_ptrs = out_index + offset_index

        tl.store(out_value_ptrs, result_value, mask=mask1)
        tl.store(out_index_ptrs, result_index, mask=mask1)


def max(inp):
    logger.debug("GEMS MAX")
    return_dtype = inp.dtype
    if inp.dtype == torch.int64:
        inp = inp.to(torch.int32)
    if inp.dtype == torch.float64:
        inp = inp.to(torch.float32)

    inp = inp.contiguous()
    M = inp.numel()
    block_size = 32 * 64
    if M < 24 * 16 * 1024:
        block_size = 16 * 1024
    elif M >= 24 * 32 * 1024 and M < 24 * 64 * 1024:
        block_size = 32 * 1024
    elif M >= 24 * 64 * 1024:
        block_size = 64 * 1024
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)
    num_stages = 1
    if mid_size > 4 * 24:
        num_stages = 3
    with torch_device_fn.device(inp.device):
        max_kernel_1[(min(mid_size, 24), 1, 1)](
            inp, mid, M, block_size, num_stages=num_stages, num_warps=1
        )
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
    if inp.shape[dim] == 1:
        out_index = torch.zeros(inp.shape, dtype=torch.int32, device=inp.device)
        if not keepdim:
            inp = torch.squeeze(inp, dim)
            out_index = torch.squeeze(out_index, dim)
        Max_out = namedtuple("max", ["values", "indices"])
        out = Max_out(values=inp.to(return_dtype), indices=out_index.to(torch.int64))
        return out
    shape = list(inp.shape)
    shape[dim] = 1
    out_value = torch.empty(shape, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape, dtype=torch.int32, device=inp.device)
    if dim < 0:
        dim = dim + inp.ndim
    if dim == 0:
        M = inp.shape[0]
        N = inp.numel() // M
        grid = lambda meta: (min(triton.cdiv(N, meta["BLOCK_N"]), 24),)
        with torch_device_fn.device(inp.device):
            max_kernel_dim_high[grid](inp, out_value, out_index, M, N)
    elif dim == inp.ndim - 1:
        N = inp.shape[inp.ndim - 1]
        M = inp.numel() // N
        grid = lambda meta: (min(triton.cdiv(M, meta["BLOCK_M"]), 24),)
        # grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            max_kernel_dim_low[grid](inp, out_value, out_index, M, N)
    else:
        B = 1
        for i in range(0, dim):
            B *= inp.shape[i]
        M = inp.shape[dim]
        N = 1
        for i in range(dim + 1, inp.ndim):
            N *= inp.shape[i]
        if B <= N * 128:
            grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), min(B, 24), 1)
            with torch_device_fn.device(inp.device):
                max_kernel_dim_mid[grid](inp, out_value, out_index, B, M, N)
        else:
            in_reshape = inp.reshape((B, M, N))
            inp_new = dim_compress(in_reshape, {0, 2})
            M = inp_new.shape[0]
            N = inp_new.numel() // M
            grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
            with torch_device_fn.device(inp.device):
                max_kernel_dim_high[grid](inp_new, out_value, out_index, M, N)
    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)
    Max_out = namedtuple("max", ["values", "indices"])
    out = Max_out(values=out_value.to(return_dtype), indices=out_index.to(torch.int64))
    return out
