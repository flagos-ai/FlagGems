# SPDX-License-Identifier: Apache-2.0
"""Contiguous-K and wave-quantization-aware Split-K Triton GEMM for MetaX Wave64 GPUs."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _metax_nt_kcontig_part_kernel(
    A,
    W,
    Out,
    Part,
    M,
    N,
    K,
    stride_am,
    stride_wn,
    stride_om,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    if GROUP_M > 1 and grid_m > 1:
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size
    else:
        pid_m = pid // grid_n
        pid_n = pid % grid_n

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    if EVEN_M:
        ram = tl.max_contiguous(tl.multiple_of(rm, BM), BM)
    else:
        ram = tl.minimum(rm, M - 1)
    if EVEN_N:
        rbn = tl.max_contiguous(tl.multiple_of(rn, BN), BN)
    else:
        rbn = tl.minimum(rn, N - 1)

    total_iters = tl.cdiv(K, BK)
    iters_per_split = tl.cdiv(total_iters, SPLIT_K)
    k_start = pid_k * iters_per_split
    k_end = tl.minimum(total_iters, k_start + iters_per_split)

    rk = tl.max_contiguous(tl.multiple_of(tl.arange(0, BK), BK), BK)
    a_ptrs = A + ram[:, None] * stride_am + (k_start * BK + rk)[None, :]
    w_ptrs = W + rbn[:, None] * stride_wn + (k_start * BK + rk)[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k_it in range(k_start, k_end):
        if EVEN_K:
            a = tl.load(a_ptrs)
            w = tl.load(w_ptrs)
        else:
            k_rem = K - k_it * BK
            a = tl.load(a_ptrs, mask=rk[None, :] < k_rem, other=0.0)
            w = tl.load(w_ptrs, mask=rk[None, :] < k_rem, other=0.0)
        acc = tl.dot(a, tl.trans(w), acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BK
        w_ptrs += BK

    if SPLIT_K == 1:
        res = acc.to(Out.dtype.element_ty)
        c_ptrs = Out + rm[:, None] * stride_om + rn[None, :]
        if EVEN_M and EVEN_N:
            tl.store(c_ptrs, res)
        else:
            tl.store(c_ptrs, res, mask=(rm < M)[:, None] & (rn < N)[None, :])
    else:
        p_ptrs = Part + pid_k * M * N + rm[:, None] * N + rn[None, :]
        if EVEN_M and EVEN_N:
            tl.store(p_ptrs, acc)
        else:
            tl.store(p_ptrs, acc, mask=(rm < M)[:, None] & (rn < N)[None, :])


@triton.jit
def _metax_splitk_reduce_kernel(
    Part,
    Out,
    M,
    N,
    stride_om,
    CONTIG_OUT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BS: tl.constexpr,
):
    idx = tl.program_id(0) * BS + tl.arange(0, BS)
    mn = M * N
    mask = idx < mn
    acc = tl.zeros((BS,), dtype=tl.float32)
    for s in tl.static_range(0, SPLIT_K):
        acc += tl.load(Part + s * mn + idx, mask=mask, other=0.0)
    res = acc.to(Out.dtype.element_ty)
    if CONTIG_OUT:
        tl.store(Out + idx, res, mask=mask)
    else:
        r = idx // N
        c = idx % N
        tl.store(Out + r * stride_om + c, res, mask=mask)


# C500 Wave64 tile sizes: (BM, BN, BK, Split-K, waves, stages, GROUP_M, scenario).
def _config(m, n, k):
    if not 1 <= m <= 4096:
        return None
    if (k, n) == (2048, 130560):
        if m <= 16:
            return (16, 64, 64, 1, 4, 3, 1, "")
        if m <= 32:
            return (32, 64, 64, 1, 8, 3, 1, "")
        if m <= 64:
            return (64, 128, 64, 1, 8, 3, 1, "")
        if m <= 256:
            return (64, 128, 64, 1, 8, 3, 4, "")
        if m <= 1024:
            return (128, 256, 32, 1, 8, 3, 8, "")
        return (256, 256, 32, 1, 8, 2, 8, "fullstage")

    if (k, n) not in {(2048, 2560), (2048, 2048), (2048, 12288), (6144, 2048)}:
        return None

    if m <= 64:
        bm = 16 if m <= 16 else (32 if m <= 32 else 64)
        if k == 6144:
            return (bm, 64, 128, 3, 8, 3, 1, "")
        if n == 2560:
            if m <= 16:
                return (16, 64, 64, 2, 4, 3, 1, "")
            if m <= 32:
                return (32, 64, 128, 2, 8, 3, 1, "")
            return (32, 64, 128, 1, 8, 3, 1, "")
        if n == 2048:
            bk = 128 if 16 < m <= 32 else 64
            return (bm, 64, bk, 3, 4 if m <= 16 else 8, 3, 1, "")
        if m <= 16:
            return (16, 64, 64, 1, 4, 3, 1, "")
        if m <= 32:
            return (32, 64, 128, 1, 8, 3, 1, "")
        return (64, 128, 64, 1, 8, 3, 1, "")

    if m < 256:
        if n == 12288:
            return (128, 128, 64, 1, 8, 2, 4, "")
        return (64, 64, 64, 1, 4, 3, 4, "")

    if k == 6144:
        return (128, 128, 64, 3 if m <= 512 else 2, 8, 2, 8, "")
    if n == 2048:
        return (128, 128, 64, 1, 8, 2, 8, "")
    if n == 2560 and m <= 1024:
        return (128, 128, 64, 1, 8, 2, 8, "")
    if m <= 1024:
        return (128, 256, 32, 1, 8, 3, 8, "")
    return (256, 256, 32, 1, 8, 2, 8, "fullstage")


def try_metax_kcontig_nt_mm(a, weight, out):
    """Return whether the C500 K-contiguous projection wrote the entire output."""
    if a.ndim != 2 or weight.ndim != 2 or out.ndim != 2:
        return False
    m, k = a.shape
    n, wk = weight.shape
    cfg = _config(m, n, k)
    if cfg is None or wk != k or out.shape != (m, n):
        return False
    if (
        not a.is_cuda
        or a.device != weight.device
        or a.device != out.device
        or a.dtype not in (torch.bfloat16, torch.float16)
        or a.dtype != weight.dtype
        or a.dtype != out.dtype
    ):
        return False
    for tensor in (a, weight, out):
        rows, cols = tensor.shape
        row_stride, col_stride = tensor.stride()
        if (
            col_stride != 1
            or row_stride < cols
            or (rows - 1) * row_stride + cols >= 2**31
        ):
            return False
    if torch._C._is_alias_of(a, out) or torch._C._is_alias_of(weight, out):
        return False

    bm, bn, bk, splits, waves, stages, group, scenario = cfg
    with torch_device_fn.device(a.device):
        partial = (
            torch.empty((splits, m, n), dtype=torch.float32, device=a.device)
            if splits > 1
            else out
        )
        grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), splits)
        launch_kw = dict(num_warps=waves, num_stages=stages, pipeline="basic")
        if scenario:
            launch_kw["scenario"] = scenario
        _metax_nt_kcontig_part_kernel[grid](
            a,
            weight,
            out,
            partial,
            m,
            n,
            k,
            a.stride(0),
            weight.stride(0),
            out.stride(0),
            BM=bm,
            BN=bn,
            BK=bk,
            SPLIT_K=splits,
            GROUP_M=group,
            EVEN_M=m % bm == 0,
            EVEN_N=n % bn == 0,
            EVEN_K=k % bk == 0,
            **launch_kw,
        )
        if splits > 1:
            mn = m * n
            bs = 1024 if mn >= 262144 else 512
            rwaves = 8 if bs == 1024 else 4
            _metax_splitk_reduce_kernel[(triton.cdiv(mn, bs),)](
                partial,
                out,
                m,
                n,
                out.stride(0),
                CONTIG_OUT=out.stride(0) == n,
                SPLIT_K=splits,
                BS=bs,
                num_warps=rwaves,
                num_stages=1,
            )
    return True
