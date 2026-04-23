import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["alpha"])
def index_add_row_kernel(
    src, index, out,
    alpha,
    M, N_src, N_out,
    ALPHA_ONE: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    col_range = tl.arange(0, BLOCK_COL)

    for row in tl.range(pid, M, num_progs):
        src_base = row * N_src
        out_base = row * N_out

        for col_off in tl.range(0, N_src, BLOCK_COL):
            cols = col_off + col_range
            col_mask = cols < N_src

            idx = tl.load(index + cols, mask=col_mask, other=0).to(tl.int32)
            src_val = tl.load(src + src_base + cols, mask=col_mask, other=0.0)

            if ALPHA_ONE:
                val = src_val
            else:
                val = src_val * alpha

            out_off = out_base + idx
            tl.atomic_add(out + out_off, val, mask=col_mask, sem='relaxed')


@libentry()
@triton.jit(do_not_specialize=["alpha"])
def index_add_flat_kernel(
    src, index, out,
    alpha,
    N_total, N_src, N_out,
    ALPHA_ONE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N_total

    row = offset // N_src
    col = offset % N_src

    idx = tl.load(index + col, mask=mask, other=0).to(tl.int32)
    src_val = tl.load(src + offset, mask=mask, other=0.0)

    if ALPHA_ONE:
        val = src_val
    else:
        val = src_val * alpha

    out_off = row * N_out + idx
    tl.atomic_add(out + out_off, val, mask=mask, sem='relaxed')


@libentry()
@triton.jit(do_not_specialize=["alpha"])
def index_add_non_inner_kernel(
    src, index, out,
    alpha,
    M, N_src, N_out, K,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_work = M * N_src

    for work_id in tl.range(pid, total_work, num_progs):
        m = work_id // N_src
        j = work_id % N_src

        idx_val = tl.load(index + j).to(tl.int32)

        src_base = (m * N_src + j) * K
        out_base = (m * N_out + idx_val) * K

        for k_off in tl.range(0, K, BLOCK_K):
            k = k_off + tl.arange(0, BLOCK_K)
            k_mask = k < K

            src_val = tl.load(src + src_base + k, mask=k_mask, other=0.0)
            tl.atomic_add(out + out_base + k, src_val * alpha, mask=k_mask, sem='relaxed')


def _select_dispatch(M, N_src, N_total):
    if M >= 256 and N_src >= 64:
        BLOCK_COL = min(triton.next_power_of_2(N_src), 2048)
        num_programs = min(M, 48)
        return 'row', BLOCK_COL, num_programs, 4
    else:
        if N_total <= 4096:
            bs = 128
        elif N_total <= 131072:
            bs = 256
        else:
            bs = 1024
        grid_size = min(triton.cdiv(N_total, bs), 65535)
        nw = 4
        return 'flat', bs, grid_size, nw


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS INDEX ADD")
    if index.dtype == torch.int64:
        index = index.to(torch.int32)

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim

    inp_c = inp.contiguous()
    out = inp_c.clone()
    src = src.contiguous()

    M = math.prod(inp.shape[:dim]) if dim > 0 else 1
    N_src = src.size(dim)
    N_out = inp.size(dim)
    K = math.prod(inp.shape[dim + 1:]) if dim < inp.ndim - 1 else 1

    if N_src == 0:
        return out

    alpha_one = (alpha == 1)

    if K == 1:
        N_total = M * N_src
        mode, block_param, grid_param, nw = _select_dispatch(M, N_src, N_total)

        if mode == 'row':
            with torch_device_fn.device(inp.device):
                index_add_row_kernel[(grid_param,)](
                    src, index, out, alpha,
                    M, N_src, N_out,
                    ALPHA_ONE=alpha_one,
                    BLOCK_COL=block_param,
                    num_warps=nw,
                )
        else:
            with torch_device_fn.device(inp.device):
                index_add_flat_kernel[(grid_param,)](
                    src, index, out, alpha,
                    N_total, N_src, N_out,
                    ALPHA_ONE=alpha_one,
                    BLOCK_SIZE=block_param,
                    num_warps=nw,
                )
    else:
        BLOCK_K = min(triton.next_power_of_2(K), 1024)
        total_work = M * N_src
        num_programs = min(total_work, 48)

        with torch_device_fn.device(inp.device):
            index_add_non_inner_kernel[(num_programs,)](
                src, index, out, alpha,
                M, N_src, N_out, K,
                BLOCK_K=BLOCK_K,
                num_warps=4,
            )

    return out


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS INDEX ADD_")
    if index.dtype == torch.int64:
        index = index.to(torch.int32)

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim

    if not inp.is_contiguous():
        result = index_add(inp, dim, index, src, alpha)
        inp.copy_(result)
        return inp

    src = src.contiguous()

    M = math.prod(inp.shape[:dim]) if dim > 0 else 1
    N_src = src.size(dim)
    N_out = inp.size(dim)
    K = math.prod(inp.shape[dim + 1:]) if dim < inp.ndim - 1 else 1

    if N_src == 0:
        return inp

    alpha_one = (alpha == 1)

    if K == 1:
        N_total = M * N_src
        mode, block_param, grid_param, nw = _select_dispatch(M, N_src, N_total)

        if mode == 'row':
            with torch_device_fn.device(inp.device):
                index_add_row_kernel[(grid_param,)](
                    src, index, inp, alpha,
                    M, N_src, N_out,
                    ALPHA_ONE=alpha_one,
                    BLOCK_COL=block_param,
                    num_warps=nw,
                )
        else:
            with torch_device_fn.device(inp.device):
                index_add_flat_kernel[(grid_param,)](
                    src, index, inp, alpha,
                    N_total, N_src, N_out,
                    ALPHA_ONE=alpha_one,
                    BLOCK_SIZE=block_param,
                    num_warps=nw,
                )
    else:
        BLOCK_K = min(triton.next_power_of_2(K), 1024)
        total_work = M * N_src
        num_programs = min(total_work, 48)

        with torch_device_fn.device(inp.device):
            index_add_non_inner_kernel[(num_programs,)](
                src, index, inp, alpha,
                M, N_src, N_out, K,
                BLOCK_K=BLOCK_K,
                num_warps=4,
            )

    return inp
