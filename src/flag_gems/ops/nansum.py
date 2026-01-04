import logging
import math
from functools import reduce

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils import tl_extra_shim

_isnan = tl_extra_shim.isnan

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def nansum_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    inp_val = tl.load(inp_ptrs, mask=mask, other=0).to(cdtype)
    inp_val = tl.where(_isnan(inp_val.to(tl.float32)), tl.zeros_like(inp_val), inp_val)
    sum_val = tl.sum(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def nansum_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    if tl.constexpr(mid.dtype.element_ty == tl.float16) or tl.constexpr(
        mid.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = mid.dtype.element_ty

    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(cdtype)

    mid_val = tl.where(_isnan(mid_val.to(tl.float32)), tl.zeros_like(mid_val), mid_val)

    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def nansum_dim_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        input_ptrs = input_ptr + inp_offset
        inp = tl.load(input_ptrs, mask=mask, other=0).to(cdtype)
        inp = tl.where(_isnan(inp.to(tl.float32)), tl.zeros_like(inp), inp)
        out = tl.sum(inp, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)
    else:
        acc = tl.zeros([TILE_N, TILE_K], dtype=cdtype)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(cdtype)
            inp = tl.where(_isnan(inp.to(tl.float32)), tl.zeros_like(inp), inp)
            acc += inp
        out = tl.sum(acc, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def nansum_dim_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid_m = tle.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        inp_offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + inp_offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=0).to(cdtype)
        inp = tl.where(_isnan(inp.to(tl.float32)), tl.zeros_like(inp), inp)
        out = tl.sum(inp, axis=0)
        out_offset = pid_m
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out)
    else:
        acc = tl.zeros([TILE_N], dtype=cdtype)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp_offsets = pid_m * N + n_offsets
            mask = n_offsets < N
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(cdtype)
            inp = tl.where(_isnan(inp.to(tl.float32)), tl.zeros_like(inp), inp)
            acc += inp
        out = tl.sum(acc, axis=0)
        out_offset = pid_m
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def nansum_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    # Map the program id to the row of inp it should compute.
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + pid * N
    out = out + pid
    row_mask = pid < M

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        x = tl.load(inp + cols, mask, other=0).to(cdtype)
        x = tl.where(_isnan(x.to(tl.float32)), tl.zeros_like(x), x)
        acc += x

    s = tl.sum(acc, axis=1)[:, None]
    tl.store(out, s, row_mask)

def nansum_comm(inp, *, dtype=None, out=None):
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int64)
            dtype = torch.int64
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    if out is None:
        out = torch.empty([], dtype=dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        nansum_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        nansum_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out

def nansum_dim_comm(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64

    # should we convert bool to int64 first?
    if inp.dtype is torch.bool and dtype is torch.int64:
        inp = inp.to(torch.int64)

    if dim == []:
        if not keepdim:
            return nansum(inp, dtype=dtype)
        else:
            dim_num = inp.ndim
            return torch.reshape(nansum(inp, dtype=dtype), [1] * dim_num)

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]

    if len(dim) == 1:
        dim0 = dim[0]
        N = inp.shape[dim0]
        M = reduce(lambda x, y: x * y, shape[:dim0], 1)
        inp = inp.contiguous()
        K = inp.numel() // M // N

        shape[dim0] = 1
        if out is None:
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        with torch_device_fn.device(inp.device):
            if K > 1:
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                nansum_dim_kernel_non_inner[grid](out, inp, M, N, K)
            else:
                grid = (M, 1, 1)
                nansum_dim_kernel_inner[grid](out, inp, M, N)

        if not keepdim:
            out = out.squeeze(dim=dim0)
        return out
    else:
        inp = dim_compress(inp, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N
        if out is None:
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            nansum_dim_kernel[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out


def nansum(inp, dim=None, keepdim=False, *, dtype=None):
    print("GEMS NANSUM")
    logger.debug("GEMS NANSUM")
    if dim is None:
        return nansum_comm(inp, dtype=dtype)
    else:
        return nansum_dim_comm(inp, dim, keepdim, dtype=dtype)


def nansum_out(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    print("GEMS NANSUM_OUT")
    logger.debug("GEMS NANSUM_OUT")
    if dim is None:
        return nansum_comm(inp, dtype=dtype, out=out)
    else:
        return nansum_dim_comm(inp, dim, keepdim, dtype=dtype, out=out)