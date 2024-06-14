import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry


def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=4) for m in block_m
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def l2_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += a * a
    sum = tl.sum(_sum, axis=1)

    out = tl.sqrt(sum)[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def max_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _max = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _max = tl.maximum(tl.abs(a), _max)

    max = tl.max(_max, axis=1)
    out = max[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def min_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _min = tl.full([BLOCK_M, BLOCK_N], value=float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=float("inf")).to(tl.float32)
        _min = tl.minimum(tl.abs(a), _min)

    min = tl.min(_min, axis=1)
    out = min[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def l0_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0).to(tl.float32)
        _sum += tl.where(a != 0, 1, 0)
    sum = tl.sum(_sum, axis=1)
    out = sum[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def v_norm_kernel(X, Out, M, N, ord, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += tl.math.pow(tl.abs(a), ord)
    sum = tl.sum(_sum, axis=1)
    out = tl.math.pow(sum, 1 / ord)[:, None]
    tl.store(Out, out, row_mask)


def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    logging.debug("GEMS VECTOR NORM")
    if dtype is not None:
        dtype = torch.dtype(dtype)
    else:
        dtype = x.dtype
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise NotImplementedError(f"vector_norm not implemented for {dtype}")
    if dim is None:
        dim = list(range(x.ndim))
    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]
    order = [i for i in range(x.ndim) if i not in dim] + dim
    x = x.permute(order).contiguous()
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N
    out = torch.empty(shape, dtype=dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    if ord == 2:
        l2_norm_kernel[grid](x, out, M, N)
    elif ord == float("inf"):
        max_norm_kernel[grid](x, out, M, N)
    elif ord == -float("inf"):
        min_norm_kernel[grid](x, out, M, N)
    elif ord == 0:
        l0_norm_kernel[grid](x, out, M, N)
    else:
        v_norm_kernel[grid](x, out, M, N, ord)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
