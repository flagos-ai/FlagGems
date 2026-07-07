import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def mean_kernel_1(inp, mid, M, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr = 1):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_tile = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, num_prog, num_stages=num_stages):
        offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        inp_ptrs = inp + offset
        mask = offset < M
        inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)
        sum_val = tl.sum(inp_val, axis=0)
        mid_ptr = mid + tile_id
        tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def mean_kernel_2(mid, out, M, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0)
    sum_val = tl.sum(mid_val, axis=0) / M
    tl.store(out, sum_val)


def mean(inp, *, dtype=None):
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype

    block_size = 32 * 64
    if M < 24 * 16 * 1024:
        block_size = 16 * 1024
    elif M >= 24 * 32 * 1024 and M < 24 * 64 * 1024:
        block_size = 32 * 1024
    elif M >= 24 * 64 * 1024:
        block_size = 64 * 1024
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)
    num_stages = 1
    if mid_size > 4 * 24:
        num_stages = 3
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        mean_kernel_1[(min(triton.cdiv(M, block_size), 24), 1, 1)](
            inp, mid, M, block_size, num_stages=num_stages, num_warps=1
        )
        mean_kernel_2[(1, 1, 1)](mid, out, M, mid_size, block_mid, num_warps=1)
    return out


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
def mean_kernel_dim_low(
    inp,
    Mean,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    # Map the program id to the row of X it should compute.
    step = tl.num_programs(0)
    pid_m = tl.program_id(0)
    num_tile = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid_m, num_tile, step):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        # # Compute mean
        # _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        n_offset_0 = tl.arange(0, BLOCK_N)
        offset_0 = m_offset[:, None] * N + n_offset_0[None, :]
        # set mask
        mask_0 = m_offset[:, None] < M and n_offset_0[None, :] < N
        inp_ptrs_0 = inp + offset_0
        _mean = tl.load(inp_ptrs_0, mask_0, other=0.0).to(tl.float32)
        if N > BLOCK_N:
            for i in tl.range(BLOCK_N, N, BLOCK_N, num_stages=num_stages):
                n_offset = i + tl.arange(0, BLOCK_N)
                offset = m_offset[:, None] * N + n_offset[None, :]
                # set mask
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                inp_ptrs = inp + offset
                a = tl.load(inp_ptrs, mask, other=0.0).to(tl.float32)
                _mean = a + _mean
        _mean /= N
        mean_row = tl.sum(_mean, axis=1)
        Mean_ptr = Mean + m_offset
        mask_m = m_offset < M
        tl.store(Mean_ptr, mean_row, mask_m)


@libentry()
@libtuner(
    configs=list(filter(keep, runtime.get_tuned_config("naive_reduction"))),
    key=["M", "N"],
)
@triton.jit
def mean_kernel_dim_high(
    inp,
    Mean,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    # Map the program id to the row of X it should compute.
    pid_n = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (N + BLOCK_N - 1) // BLOCK_N
    for tile_id_n in tl.range(pid_n, num_tile, step):
        n_offset = tile_id_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_offset_0 = tl.arange(0, BLOCK_M)
        offset_0 = m_offset_0[:, None] * N + n_offset[None, :]
        mask_0 = m_offset_0[:, None] < M and n_offset[None, :] < N
        inp_ptrs_0 = inp + offset_0
        _mean = tl.load(inp_ptrs_0, mask_0, other=0.0).to(tl.float32)
        if M > BLOCK_M:
            for i in tl.range(BLOCK_M, M, BLOCK_M, num_stages=num_stages):
                m_offset = i + tl.arange(0, BLOCK_M)
                offset = m_offset[:, None] * N + n_offset[None, :]
                # set mask
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                inp_ptrs = inp + offset
                a = tl.load(inp_ptrs, mask, other=0.0).to(tl.float32)
                _mean += a
        _mean /= M
        mean_col = tl.sum(_mean, axis=0)
        Mean_ptr = Mean + n_offset
        n_mask = n_offset < N
        tl.store(Mean_ptr, mean_col, n_mask)


@libentry()
@libtuner(
    configs=list(filter(keep, runtime.get_tuned_config("naive_reduction"))),
    key=["M", "N"],
)
@triton.jit
def mean_kernel_dim_mid(
    inpIn,
    out_value_in,
    B,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    # Map the program id to the row of X it should compute.
    pid_b = tl.program_id(1)
    pid_n = tl.program_id(0)
    step = tl.num_programs(1)
    for tile_id_b in tl.range(pid_b, B, step):
        b_offset = tile_id_b * M * N
        inp = inpIn + b_offset
        out_value = out_value_in + tile_id_b * N
        n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_offset_0 = tl.arange(0, BLOCK_M)
        offset_0 = m_offset_0[:, None] * N + n_offset[None, :]
        mask_0 = m_offset_0[:, None] < M and n_offset[None, :] < N
        inp_ptrs_0 = inp + offset_0
        _mean = tl.load(inp_ptrs_0, mask_0, other=0.0).to(tl.float32)
        if M > BLOCK_M:
            for i in tl.range(BLOCK_M, M, BLOCK_M, num_stages=num_stages):
                m_offset = i + tl.arange(0, BLOCK_M)
                offset = m_offset[:, None] * N + n_offset[None, :]
                # set mask
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                inp_ptrs = inp + offset
                a = tl.load(inp_ptrs, mask, other=0.0).to(tl.float32)
                _mean += a
        _mean /= M
        mean = tl.sum(_mean, axis=0)
        out_value_ptrs = out_value + n_offset
        n_mask = n_offset < N
        tl.store(out_value_ptrs, mean, n_mask)


def mean_dim(x, dim, keepdim=False, *, dtype=None):
    return_dtype = x.dtype
    if x.dtype == torch.int64:
        x.dtype = torch.int32
    if dtype is None:
        dtype = x.dtype
    if dim is None:
        out = mean(x, dtype=dtype)
        if not keepdim:
            out = out.reshape([1] * x.ndim)
        return out
    # if x.shape[dim] == 1:
    #     return x
    # print_x = x
    # print("print_x", print_x)
    # print("inp.shape:", x.shape)
    # print("dim:", dim)
    if len(dim) == 1:
        inp = x
        mean_dim = dim[0]
        shape = list(x.shape)
        if shape[mean_dim] == 1:
            if not keepdim:
                inp = inp.squeeze(dim)
            return inp.to(return_dtype)
        shape[mean_dim] = 1
        out = torch.empty(shape, dtype=dtype, device=x.device)
        if mean_dim == 0:
            M = inp.shape[0]
            N = inp.numel() // M
            grid = lambda meta: (min(triton.cdiv(N, meta["BLOCK_N"]), 24),)
            with torch_device_fn.device(inp.device):
                mean_kernel_dim_high[grid](inp, out, M, N)
        elif mean_dim == inp.ndim - 1:
            N = inp.shape[inp.ndim - 1]
            M = inp.numel() // N
            grid = lambda meta: (min(triton.cdiv(M, meta["BLOCK_M"]), 24),)
            # grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
            with torch_device_fn.device(inp.device):
                mean_kernel_dim_low[grid](inp, out, M, N)
        else:
            B = 1
            for i in range(0, mean_dim):
                B *= inp.shape[i]
            M = inp.shape[mean_dim]
            N = 1
            for i in range(mean_dim + 1, inp.ndim):
                N *= inp.shape[i]
            if B <= N * 128:
                grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), min(B, 24), 1)
                with torch_device_fn.device(inp.device):
                    mean_kernel_dim_mid[grid](inp, out, B, M, N)
            else:
                in_reshape = inp.reshape((B, M, N))
                inp_new = dim_compress(in_reshape, {0, 2})
                M = inp_new.shape[0]
                N = inp_new.numel() // M
                grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
                with torch_device_fn.device(inp.device):
                    mean_kernel_dim_high[grid](inp_new, out, M, N)
        if not keepdim:
            out = out.squeeze(dim)
        return out.to(return_dtype)
    else:
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        x = dim_compress(x, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = x.numel() // N
        if M == 1:
            out = mean(x)
            for i in range(0, x.ndim):
                out = out.unsqueeze(0)
            if not keepdim:
                out = out.squeeze(dim)
            return out.to(return_dtype)
        else:
            out = torch.empty(shape, dtype=dtype, device=x.device)
            grid = lambda META: (min(triton.cdiv(M, META["BLOCK_M"]), 24),)
            with torch_device_fn.device(x.device):
                mean_kernel_dim_low[grid](x, out, M, N)
            if not keepdim:
                out = out.squeeze(dim)
            return out.to(return_dtype)
