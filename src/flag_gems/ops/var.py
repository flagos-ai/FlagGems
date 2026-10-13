# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


@triton.jit
def welford_combine(mean_x, count_x, M_x, mean_y, count_y, M_y):
    # Numerically stable parallel-Welford (Chan) merge. Unlike welford_func
    # above, the M2 update is expressed with the difference of means, so it does
    # not lose precision when the data mean is far from zero.
    count = count_x + count_y
    _count = tl.maximum(count, 1.0)
    delta = mean_y - mean_x
    mean = mean_x + delta * count_y / _count
    M = M_x + M_y + delta * delta * count_x * count_y / _count
    return mean, count, M


@libentry()
@triton.jit(do_not_specialize=["correction", "M", "N"])
def var_welford_kernel(
    X,
    Var,
    M,
    N,
    correction,
    BLOCK_N: tl.constexpr,
):
    # One row per program to avoid autotune correctness issues on some backends.
    pid = ext.program_id(0)
    X = X + pid * N
    Var = Var + pid

    # Two-pass approach using tl.sum to avoid tl.reduce correctss issues.
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
    # Write var
    tl.store(Var, var)


@libentry()
@triton.jit
def var_kernel_1(
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
def var_kernel_2(
    Acc,
    Average,
    Count,
    Var,
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
    tl.store(Var, var)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit(do_not_specialize=["correction"])
def _var_dim_kernel_inner(
    Out,
    X,
    M,
    N,
    correction,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    row = X + pid_m * N

    if ONE_TILE_PER_CTA:
        # The whole row fits in one tile: load once, reuse for mean and M2.
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(row + n_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
        diff = x - mean
        m2 = tl.sum(tl.where(mask, diff * diff, 0.0), axis=0)
    else:
        # Single-read Welford. The old code read the whole row twice (once for
        # the mean, once for the squared deviations), doubling global-memory
        # traffic, which dominated the runtime on large rows. Here each lane
        # streams a running (count, mean, M2) over its own elements with no
        # cross-lane reduction inside the loop (that would serialize on the
        # small tile axis), and the lanes are merged once at the end with a
        # numerically stable Chan combine.
        count_acc = tl.zeros((TILE_N,), dtype=tl.float32)
        mean_acc = tl.zeros((TILE_N,), dtype=tl.float32)
        m2_acc = tl.zeros((TILE_N,), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            x = tl.load(row + n_offsets, mask=mask, other=0.0).to(tl.float32)
            new_count = count_acc + mask.to(tl.float32)
            delta = x - mean_acc
            mean_acc += tl.where(mask, delta / tl.maximum(new_count, 1.0), 0.0)
            m2_acc += tl.where(mask, delta * (x - mean_acc), 0.0)
            count_acc = new_count
        _mean, _count, m2 = tl.reduce(
            (mean_acc, count_acc, m2_acc), axis=0, combine_fn=welford_combine
        )

    denom = N - correction
    var = m2 / tl.maximum(denom, 0.0)
    tl.store(Out + pid_m, var.to(Out.dtype.element_ty), mask=pid_m < M)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit(do_not_specialize=["correction"])
def _var_dim_kernel_non_inner(
    Out,
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
        # The whole reduced dim fits in one tile: load once, reuse for mean/M2.
        n_offsets = tl.arange(0, TILE_N)[:, None]
        mask = (n_offsets < N) & (k_offsets < K)
        offsets = pid_m * N * K + n_offsets * K + k_offsets
        x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0, keep_dims=True) / N
        diff = x - mean
        m2 = tl.sum(tl.where(mask, diff * diff, 0.0), axis=0, keep_dims=True)
    else:
        # Single-read Welford (see the inner kernel). Each (n_lane, k) slot
        # streams a running (count, mean, M2) with no cross-lane reduction in
        # the loop; the n lanes are merged once at the end with the stable Chan
        # combine.
        count_acc = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)
        mean_acc = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)
        m2_acc = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            mask = (n_offsets < N) & (k_offsets < K)
            offsets = pid_m * N * K + n_offsets * K + k_offsets
            x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
            new_count = count_acc + mask.to(tl.float32)
            delta = x - mean_acc
            mean_acc += tl.where(mask, delta / tl.maximum(new_count, 1.0), 0.0)
            m2_acc += tl.where(mask, delta * (x - mean_acc), 0.0)
            count_acc = new_count
        _mean, _count, m2 = tl.reduce(
            (mean_acc, count_acc, m2_acc),
            axis=0,
            combine_fn=welford_combine,
            keep_dims=True,
        )

    denom = N - correction
    var = m2 / tl.maximum(denom, 0.0)
    out_offsets = pid_m * K + k_offsets
    tl.store(Out + out_offsets, var.to(Out.dtype.element_ty), mask=k_offsets < K)


def var(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS VAR")
    if correction is None:
        correction = 1.0

    if dim is None or len(dim) == x.ndim:
        dim = list(range(x.ndim))
        shape = [1] * x.ndim
        N = x.numel()
        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        BLOCK_N = 1024
        BLOCK_NUM = triton.cdiv(N, BLOCK_N)
        acc = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)
        average = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)
        count = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)

        with torch_device_fn.device(x.device):
            var_kernel_1[(BLOCK_NUM,)](x, acc, average, count, N, BLOCK_N=BLOCK_N)
            var_kernel_2[(1,)](acc, average, count, var, N, correction, BLOCK_NUM)
        if not keepdim:
            var = var.squeeze(dim=dim)
        return var

    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]

    if len(dim) == 1:
        # Single-dim reduction: split into (M, N, K) and reduce over N with a
        # strided kernel, avoiding the dim_compress copy. K == 1 means the
        # reduced dim is innermost (contiguous); K > 1 means it is a middle or
        # outer dim, where the copy used to dominate the runtime.
        dim0 = dim[0]
        N = shape[dim0]
        M = 1
        for size in shape[:dim0]:
            M *= size
        K = 1
        for size in shape[dim0 + 1 :]:
            K *= size
        shape[dim0] = 1

        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        if M == 0 or K == 0:
            # A spectator dimension is empty: the output is a valid empty
            # tensor and there is nothing to reduce. Matches torch.
            return out.squeeze(dim=dim0) if not keepdim else out
        if N == 0:
            # Reducing over an empty dim: variance is 0/0, i.e. NaN, like torch.
            # (A correction >= N with N > 0 is handled in the kernel, which
            # clamps the divisor at 0 and yields +inf, also matching torch.)
            out.fill_(float("nan"))
            return out.squeeze(dim=dim0) if not keepdim else out

        x_contiguous = x.contiguous()
        with torch_device_fn.device(x.device):
            if K > 1:
                grid = lambda META: (M, triton.cdiv(K, META["TILE_K"]), 1)
                _var_dim_kernel_non_inner[grid](out, x_contiguous, M, N, K, correction)
            else:
                grid = (M, 1, 1)
                _var_dim_kernel_inner[grid](out, x_contiguous, M, N, correction)
        return out.squeeze(dim=dim0) if not keepdim else out

    # Multi-dim reduction keeps the original dim_compress path.
    x = dim_compress(x, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N
    var = torch.empty(shape, dtype=x.dtype, device=x.device)

    BLOCK_N = 1024
    grid = (M,)
    with torch_device_fn.device(x.device):
        var_welford_kernel[grid](x, var, M, N, correction, BLOCK_N=BLOCK_N)

    if not keepdim:
        var = var.squeeze(dim=dim)
    return var


def var_dim(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS VAR_DIM")
    return var(x, dim=dim, correction=correction, keepdim=keepdim)


def var_correction(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS VAR_CORRECTION")
    return var(x, dim=dim, correction=correction, keepdim=keepdim)
