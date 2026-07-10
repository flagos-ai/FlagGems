import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def welford_func(mean_x, count_x, M_x, mean_y, count_y, M_y):
    count = count_x + count_y
    _count = tl.maximum(count, 1)
    mc_x = mean_x * count_x
    mc_y = mean_y * count_y
    mean = (mc_x + mc_y) / _count
    M = M_x + mc_x * mean_x + M_y + mc_y * mean_y - count * mean * mean
    return mean, count, M


@libentry()
@triton.jit(do_not_specialize=["correction", "M", "N"])
def var_mean_welford_kernel(
    X,
    Var,
    Mean,
    M,
    N,
    correction,
    BLOCK_N: tl.constexpr,
):
    # One row per program to avoid autotune correctness issues on some backends.
    pid = ext.program_id(0)
    X = X + pid * N
    Var = Var + pid
    Mean = Mean + pid

    _sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += x
    mean = tl.sum(_sum) / N

    _acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        _acc += diff * diff
    var = tl.sum(_acc) / (N - correction)
    # Write mean / var
    tl.store(Mean, mean)
    tl.store(Var, var)


@libentry()
@triton.jit
def var_mean_kernel_1(
    X,
    Acc,
    Average,
    Count,
    N,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X it should compute.
    pid = ext.program_id(0)
    offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    X = X + offset
    Acc = Acc + pid
    Average = Average + pid
    Count = Count + pid
    mask = offset < N

    x = tl.load(X, mask, other=0.0).to(tl.float32)

    count = tl.sum(mask.to(tl.float32))
    average = tl.sum(x) / count
    acc = tl.sum(x * x) - count * average * average

    tl.store(Average, average)
    tl.store(Acc, acc)
    tl.store(Count, count)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("var_mean"))
@triton.jit(do_not_specialize=["correction"])
def var_mean_kernel_2(
    Acc,
    Average,
    Count,
    Var,
    Mean,
    N,
    correction,
    BLOCK_NUM,
    BLOCK_N: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_N)
    mask = offset < BLOCK_NUM
    Acc = Acc + offset
    Average = Average + offset
    Count = Count + offset
    acc = tl.load(Acc, mask, other=0.0).to(tl.float32)
    average = tl.load(Average, mask, other=0.0).to(tl.float32)
    count = tl.load(Count, mask, other=0.0).to(tl.float32)

    mean, _, nvar = tl.reduce((average, count, acc), axis=0, combine_fn=welford_func)

    var = nvar / (N - correction)
    tl.store(Mean, mean)
    tl.store(Var, var)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit(do_not_specialize=["correction"])
def _var_mean_dim_kernel_inner(
    Var,
    Mean,
    X,
    M,
    N,
    correction,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    # Two-pass mean/variance over the innermost reduced dim (K == 1), avoiding
    # the dim_compress copy. Stores both the variance and the mean.
    pid_m = tl.program_id(0)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(X + pid_m * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
    else:
        sum_acc = tl.zeros((TILE_N,), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            x = tl.load(X + pid_m * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
            sum_acc += x
        mean = tl.sum(sum_acc, axis=0) / N

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(X + pid_m * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        sq_sum = tl.sum(tl.where(mask, diff * diff, 0.0), axis=0)
    else:
        sq_acc = tl.zeros((TILE_N,), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            x = tl.load(X + pid_m * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
            diff = x - mean
            sq_acc += tl.where(mask, diff * diff, 0.0)
        sq_sum = tl.sum(sq_acc, axis=0)

    # torch clamps the divisor at 0, so denom <= 0 yields +inf (or nan when the
    # squared-deviation sum is also 0), matching torch.var_mean.
    denom = tl.maximum(N - correction, 0.0)
    var = sq_sum / denom
    tl.store(Var + pid_m, var.to(Var.dtype.element_ty), mask=pid_m < M)
    tl.store(Mean + pid_m, mean.to(Mean.dtype.element_ty), mask=pid_m < M)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit(do_not_specialize=["correction"])
def _var_mean_dim_kernel_non_inner(
    Var,
    Mean,
    X,
    M,
    N,
    K,
    correction,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        mask = (n_offsets < N) & (k_offsets < K)
        offsets = pid_m * N * K + n_offsets * K + k_offsets
        x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0, keep_dims=True) / N
    else:
        sum_acc = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            mask = (n_offsets < N) & (k_offsets < K)
            offsets = pid_m * N * K + n_offsets * K + k_offsets
            x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
            sum_acc += x
        mean = tl.sum(sum_acc, axis=0, keep_dims=True) / N

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        mask = (n_offsets < N) & (k_offsets < K)
        offsets = pid_m * N * K + n_offsets * K + k_offsets
        x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        sq_sum = tl.sum(tl.where(mask, diff * diff, 0.0), axis=0, keep_dims=True)
    else:
        sq_acc = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            mask = (n_offsets < N) & (k_offsets < K)
            offsets = pid_m * N * K + n_offsets * K + k_offsets
            x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
            diff = x - mean
            sq_acc += tl.where(mask, diff * diff, 0.0)
        sq_sum = tl.sum(sq_acc, axis=0, keep_dims=True)

    denom = tl.maximum(N - correction, 0.0)
    var = sq_sum / denom
    out_offsets = pid_m * K + k_offsets
    tl.store(Var + out_offsets, var.to(Var.dtype.element_ty), mask=k_offsets < K)
    tl.store(Mean + out_offsets, mean.to(Mean.dtype.element_ty), mask=k_offsets < K)


def var_mean(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS VAR MEAN")
    if correction is None:
        correction = 1.0

    if dim is None or len(dim) == x.ndim:
        dim = list(range(x.ndim))
        shape = [1] * x.ndim
        N = x.numel()
        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(shape, dtype=x.dtype, device=x.device)
        BLOCK_N = 1024
        BLOCK_NUM = triton.cdiv(N, BLOCK_N)
        acc = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)
        average = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)
        count = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)

        with torch_device_fn.device(x.device):
            var_mean_kernel_1[(BLOCK_NUM,)](x, acc, average, count, N, BLOCK_N=BLOCK_N)
            var_mean_kernel_2[(1,)](
                acc, average, count, var, mean, N, correction, BLOCK_NUM
            )
    elif len(dim) == 1:
        # Single-dim reduction: split into (M, N, K) and reduce over N with a
        # strided kernel, avoiding the dim_compress copy. K == 1 means the
        # reduced dim is innermost (contiguous); K > 1 means it is a middle or
        # outer dim, where the copy used to dominate the runtime.
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        dim0 = dim[0]
        N = shape[dim0]
        M = 1
        for size in shape[:dim0]:
            M *= size
        K = 1
        for size in shape[dim0 + 1 :]:
            K *= size
        shape[dim0] = 1
        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(shape, dtype=x.dtype, device=x.device)
        if M == 0 or K == 0:
            # A spectator dimension is empty: both outputs are valid empty
            # tensors and there is nothing to reduce. Matches torch.
            if not keepdim:
                var = var.squeeze(dim=dim)
                mean = mean.squeeze(dim=dim)
            return var, mean
        if N == 0:
            # Reducing over an empty dim: mean and variance are both NaN, like
            # torch (mean of nothing is 0/0, variance is undefined).
            var.fill_(float("nan"))
            mean.fill_(float("nan"))
            if not keepdim:
                var = var.squeeze(dim=dim)
                mean = mean.squeeze(dim=dim)
            return var, mean
        x = x.contiguous()
        with torch_device_fn.device(x.device):
            if K == 1:
                grid = (M, 1, 1)
                _var_mean_dim_kernel_inner[grid](var, mean, x, M, N, correction)
            else:
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                _var_mean_dim_kernel_non_inner[grid](var, mean, x, M, N, K, correction)
    else:
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        x = dim_compress(x, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = x.numel() // N
        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(shape, dtype=x.dtype, device=x.device)

        BLOCK_N = 1024
        grid = (M,)
        with torch_device_fn.device(x.device):
            var_mean_welford_kernel[grid](
                x, var, mean, M, N, correction, BLOCK_N=BLOCK_N
            )

    if not keepdim:
        var = var.squeeze(dim=dim)
        mean = mean.squeeze(dim=dim)
    return var, mean
