import logging
import math
from typing import Sequence

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)


@triton.jit
def reduce_mul(a, b):
    return a * b


NAIVE_REDUCTION_CONFIGS = [
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 64}, num_warps=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 1024}, num_warps=8, num_stages=2),
]


def _prune_reduction_configs(configs, named_args, **meta):
    """Skip oversized tiles to avoid needless autotune on tiny shapes."""
    M = named_args["M"]
    N = named_args["N"]
    max_block_m = max(M, 8)
    min_block_m = 8
    n_cap = 1 << (N - 1).bit_length()
    n_cap = max(64, min(n_cap, 1024))
    filtered = [
        cfg
        for cfg in configs
        if min_block_m <= cfg.kwargs["BLOCK_M"] <= max_block_m
        and cfg.kwargs["BLOCK_N"] <= max(256, n_cap)
    ]
    return filtered or configs


def _flatten_dim(shape: Sequence[int], dim: int):
    dim = dim % len(shape)
    n = shape[dim]
    inner = math.prod(shape[dim + 1 :]) if dim + 1 < len(shape) else 1
    outer = math.prod(shape[:dim]) if dim > 0 else 1
    return dim, n, inner, outer


def _reshape_output(out: torch.Tensor, shape: list[int], dim: int, keepdim: bool):
    out_shape = shape.copy()
    out_shape[dim] = 1
    out_view = out.view(out_shape)
    if not keepdim:
        out_view = torch.squeeze(out_view, dim)
    return out_view


@libentry()
@triton.jit
def prod_kernel_mid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    dtype = inp.type.element_ty
    acc_dtype = tl.float32
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0).to(acc_dtype)
    mid_value = tl.reduce(inp_val, axis=0, combine_fn=reduce_mul).to(dtype)
    mid_ptr = mid + pid
    tl.store(mid_ptr, mid_value)


@libentry()
@triton.jit
def prod_kernel_result(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    dtype = mid.type.element_ty
    acc_dtype = tl.float32
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=1.0).to(acc_dtype)
    prod_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_mul).to(dtype)
    tl.store(out, prod_val)


@triton.jit
def prod_kernel_dim_64(
    inp,
    out,
    M,
    INNER,
    STRIDE_OUTER,
    BLOCK_M: tl.constexpr,
):
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    base_ptr = inp + rows * STRIDE_OUTER
    cols = tl.arange(0, 64)
    vals = tl.load(base_ptr[:, None] + cols[None, :], cache_modifier=".cg")
    prod_vals = tl.reduce(vals.to(tl.float32), axis=1, combine_fn=reduce_mul)
    tl.store(out + rows, prod_vals.to(inp.type.element_ty), mask=row_mask)


@triton.jit
def prod_kernel_dim_contig(
    inp,
    out,
    M,
    INNER,
    STRIDE_OUTER,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    base_ptr = inp + rows * STRIDE_OUTER
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols[None, :] < STRIDE_OUTER
    mask = row_mask[:, None] & col_mask
    vals = tl.load(
        base_ptr[:, None] + cols[None, :],
        mask=mask,
        other=1.0,
        cache_modifier=".cg",
    )
    prod_vals = tl.reduce(vals.to(tl.float32), axis=1, combine_fn=reduce_mul)
    tl.store(out + rows, prod_vals.to(inp.type.element_ty), mask=row_mask)


@triton.jit
def prod_kernel_dim_dense(
    inp,
    out,
    M,
    N,
    INNER,
    STRIDE_OUTER,
    STRIDE_REDUCE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = inp.type.element_ty
    acc_dtype = tl.float32
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    outer_idx = rows // INNER
    inner_idx = rows % INNER
    base_ptr = inp + outer_idx * STRIDE_OUTER + inner_idx

    acc = tl.full((BLOCK_M,), value=1.0, dtype=acc_dtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        vals = tl.load(
            base_ptr[:, None] + cols[None, :] * STRIDE_REDUCE,
            cache_modifier=".cg",
        ).to(acc_dtype)
        chunk_prod = tl.reduce(vals, axis=1, combine_fn=reduce_mul)
        acc *= chunk_prod

    tl.store(out + rows, acc.to(dtype))


@triton.autotune(
    configs=NAIVE_REDUCTION_CONFIGS,
    key=["M", "N"],
    prune_configs_by={"early_config_prune": _prune_reduction_configs},
    warmup=2,
    rep=8,
)
@triton.jit
def prod_kernel_dim(
    inp,
    out,
    M,
    N,
    INNER,
    STRIDE_OUTER,
    STRIDE_REDUCE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = inp.type.element_ty
    acc_dtype = tl.float32
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = rows.to(tl.int64)
    row_mask = rows < M

    outer_idx = rows // INNER
    inner_idx = rows % INNER
    base_ptr = inp + outer_idx * STRIDE_OUTER + inner_idx

    acc = tl.full((BLOCK_M,), value=1.0, dtype=acc_dtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        cols = cols.to(tl.int64)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        vals = tl.load(
            base_ptr[:, None] + cols[None, :] * STRIDE_REDUCE,
            mask=mask,
            other=1.0,
            cache_modifier=".cg",
        ).to(acc_dtype)
        chunk_prod = tl.reduce(vals, axis=1, combine_fn=reduce_mul)
        acc *= chunk_prod

    out_ptrs = out + rows
    tl.store(out_ptrs, acc.to(dtype), mask=row_mask)


def prod(inp, *, dtype=None):
    logger.debug("GEMS_MTHREADS PROD")
    if dtype is None:
        dtype = inp.dtype
    if not inp.is_contiguous():
        inp = inp.contiguous()

    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    block_size = min(block_size * 2, 4096, triton.next_power_of_2(M))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        prod_kernel_mid[(mid_size, 1, 1)](inp, mid, M, block_size)
        prod_kernel_result[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def prod_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_MTHREADS PROD DIM")
    assert dim is not None, "dim must be specified"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim

    if dtype is None:
        dtype = inp.dtype
    if not inp.is_contiguous():
        inp = dim_compress(inp, dim)
        dim = inp.ndim - 1

    shape = list(inp.shape)
    dim, n, inner, outer = _flatten_dim(shape, dim)
    m = outer * inner

    out_flat = torch.empty((m,), dtype=dtype, device=inp.device)

    stride = inp.stride()
    stride_reduce = stride[dim]
    stride_outer = stride_reduce * n

    if n == 64 and stride_reduce == 1 and stride_outer == n:
        grid_64 = (triton.cdiv(m, 8),)
        with torch_device_fn.device(inp.device):
            prod_kernel_dim_64[grid_64](
                inp, out_flat, m, inner, stride_outer, BLOCK_M=8, num_warps=2
            )
        return _reshape_output(out_flat, shape, dim, keepdim)

    key = (m, n, str(dtype), str(out_flat.dtype))
    config = prod_kernel_dim.cache.get(key, None)
    if m * n >= 64 * 1024 * 1024 and config is None:
        if dtype in (torch.float16, torch.bfloat16):
            config = triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=2
            )
        else:
            config = triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=1
            )
        prod_kernel_dim.cache[key] = config

    if config is not None:
        block_m_cfg = config.kwargs["BLOCK_M"]
        block_n_cfg = config.kwargs["BLOCK_N"]
        if m % block_m_cfg == 0 and n % block_n_cfg == 0:
            grid_dense = (m // block_m_cfg,)
            with torch_device_fn.device(inp.device):
                prod_kernel_dim_dense[grid_dense](
                    inp,
                    out_flat,
                    m,
                    n,
                    inner,
                    stride_outer,
                    stride_reduce,
                    BLOCK_M=block_m_cfg,
                    BLOCK_N=block_n_cfg,
                    num_warps=config.num_warps or 4,
                    num_stages=config.num_stages or 1,
                )
            return _reshape_output(out_flat, shape, dim, keepdim)

    if stride_reduce == 1 and stride_outer == n and n <= 1024:
        block_m = 128 if n >= 256 else 64
        block_n = min(512, max(64, 1 << (n - 1).bit_length()))
        grid_contig = (triton.cdiv(m, block_m),)
        with torch_device_fn.device(inp.device):
            prod_kernel_dim_contig[grid_contig](
                inp,
                out_flat,
                m,
                inner,
                stride_outer,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=8 if n >= 256 else 4,
                num_stages=2,
            )
        return _reshape_output(out_flat, shape, dim, keepdim)

    if n <= 64:
        prod_kernel_dim.cache[key] = triton.Config(
            {"BLOCK_M": 8, "BLOCK_N": 64}, num_warps=2, num_stages=1
        )

    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        prod_kernel_dim[grid](
            inp,
            out_flat,
            m,
            n,
            max(inner, 1),
            stride_outer,
            stride_reduce,
        )

    return _reshape_output(out_flat, shape, dim, keepdim)
