# Copyright 2023-2026 SGLang Team
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
# Adapted from CherryLemon/sglang, Day0 85f8105f5d2e2e0bdea70ff50cb5449bf9567053.

"""V4.1 predecessor-pre mHC; preserve FP32 mixing weights and K reductions."""

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_combine_kernel(
    x_ptr,
    pre_ptr,
    y_ptr,
    H,
    x_stride_m,
    pre_stride_m,
    pre_stride_k,
    y_stride_m,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(HC):
        pk = tl.load(pre_ptr + pid_m * pre_stride_m + k * pre_stride_k).to(tl.float32)
        xv = tl.load(
            x_ptr + pid_m * x_stride_m + k * H + offs_h, mask=mask, other=0.0
        ).to(tl.float32)
        acc += pk * xv
    tl.store(y_ptr + pid_m * y_stride_m + offs_h, acc, mask=mask)


@triton.jit
def _hc_mix_stats_partial_kernel(
    x_ptr,
    w_ptr,
    part_mix_ptr,
    part_sq_ptr,
    M,
    K,
    x_stride_m,
    w_stride_n,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Mixing dot products and row sum of squares over one K slice; the slicing
    and tiles are compile-time constants, so a row's fp32 operation sequence
    does not depend on the batch size."""
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, MIX_PAD)
    mask_m = offs_m < M
    mask_n = offs_n < MIX
    k_per_slice = K // NUM_SLICES
    k_start = pid_s * k_per_slice
    acc = tl.zeros([BLOCK_M, MIX_PAD], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for kb in range(0, k_per_slice, BLOCK_K):
        offs_k = k_start + kb + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_start + k_per_slice
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * x_stride_m + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        w_tile = tl.load(
            w_ptr + offs_n[None, :] * w_stride_n + offs_k[:, None],
            mask=mask_n[None, :] & mask_k[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(x_tile, w_tile, input_precision=DOT_PRECISION)
        sq += tl.sum(x_tile * x_tile, axis=1)
    tl.store(
        part_mix_ptr + (pid_s * M + offs_m[:, None]) * MIX + offs_n[None, :],
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )
    tl.store(part_sq_ptr + pid_s * M + offs_m, sq, mask=mask_m)


@triton.jit
def _hc_mix_stats_reduce_kernel(
    part_mix_ptr,
    part_sq_ptr,
    mixes_ptr,
    M,
    inv_k,
    eps,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Sum the NUM_SLICES partials in slice order and apply the rms scaling."""
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, MIX_PAD)
    mask_m = offs_m < M
    mask_n = offs_n < MIX
    acc = tl.zeros([BLOCK_M, MIX_PAD], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for s in tl.static_range(NUM_SLICES):
        acc += tl.load(
            part_mix_ptr + (s * M + offs_m[:, None]) * MIX + offs_n[None, :],
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        sq += tl.load(part_sq_ptr + s * M + offs_m, mask=mask_m, other=0.0)
    rsqrt = 1.0 / tl.sqrt(sq * inv_k + eps)
    tl.store(
        mixes_ptr + offs_m[:, None] * MIX + offs_n[None, :],
        acc * rsqrt[:, None],
        mask=mask_m[:, None] & mask_n[None, :],
    )


_HC_MIX_SLICE_CHOICES = (80, 64, 40, 32, 16, 8, 4, 2, 1)

_HC_MIX_BLOCK_M = 32

_HC_MIX_BLOCK_K = 64

_HC_MIX_NUM_WARPS = 4

_HC_MIX_DOT_PRECISION = "tf32x3"

_HC_MIX_NUM_STAGES = 2

_HC_MIX_BLOCK_M_SMALL = 8

_HC_MIX_BLOCK_M_MID = 16

_HC_MIX_MID_MAX_M = 2048


def _block_m_for(m: int) -> int:
    """Row-tile choices preserve each row's arithmetic and may depend on M."""
    if m <= _HC_MIX_BLOCK_M_SMALL:
        return _HC_MIX_BLOCK_M_SMALL
    if m <= _HC_MIX_MID_MAX_M:
        return _HC_MIX_BLOCK_M_MID
    return _HC_MIX_BLOCK_M


def _num_slices_for(k: int) -> int:
    """Slice count depends only on K, never on batch size M."""
    blocks = k // _HC_MIX_BLOCK_K
    assert k % _HC_MIX_BLOCK_K == 0, k
    for n in _HC_MIX_SLICE_CHOICES:
        if blocks % n == 0:
            return n
    return 1


def hc_mix_stats(x_flat: torch.Tensor, hc_fn: torch.Tensor, eps: float) -> torch.Tensor:
    """Batch-invariant F.linear(x_flat.float(), hc_fn) * rsqrt(mean(x_flat^2) + eps).

    x_flat is [M, K] in any float dtype; hc_fn is [MIX, K] fp32; returns [M, MIX] fp32.
    K slicing and reduction order are independent of M, so each row is bitwise
    identical whether computed alone or in a batch.
    """
    assert x_flat.dim() == 2 and hc_fn.dim() == 2
    assert x_flat.stride(1) == 1 and hc_fn.stride(1) == 1
    assert hc_fn.dtype == torch.float32
    m, k = x_flat.shape
    mix = hc_fn.shape[0]
    assert hc_fn.shape[1] == k
    num_slices = _num_slices_for(k)
    mix_pad = max(16, triton.next_power_of_2(mix))
    part_mix = torch.empty(
        (num_slices, m, mix), dtype=torch.float32, device=x_flat.device
    )
    part_sq = torch.empty((num_slices, m), dtype=torch.float32, device=x_flat.device)
    mixes = torch.empty((m, mix), dtype=torch.float32, device=x_flat.device)
    if m == 0:
        return mixes
    block_m = _block_m_for(m)
    grid_m = triton.cdiv(m, block_m)
    _hc_mix_stats_partial_kernel[(grid_m, num_slices)](
        x_flat,
        hc_fn,
        part_mix,
        part_sq,
        m,
        k,
        x_flat.stride(0),
        hc_fn.stride(0),
        MIX=mix,
        MIX_PAD=mix_pad,
        NUM_SLICES=num_slices,
        BLOCK_M=block_m,
        BLOCK_K=_HC_MIX_BLOCK_K,
        DOT_PRECISION=_HC_MIX_DOT_PRECISION,
        num_warps=_HC_MIX_NUM_WARPS,
        num_stages=_HC_MIX_NUM_STAGES,
    )
    _hc_mix_stats_reduce_kernel[(grid_m,)](
        part_mix,
        part_sq,
        mixes,
        m,
        1.0 / k,
        eps,
        MIX=mix,
        MIX_PAD=mix_pad,
        NUM_SLICES=num_slices,
        BLOCK_M=block_m,
        num_warps=4,
    )
    return mixes


@triton.jit
def _hc_mix_reduce_sinkhorn_kernel(
    part_mix_ptr,
    part_sq_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    m,
    inv_k,
    rms_eps,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
):
    """One CTA per row keeps the sinkhorn reductions two-dimensional.
    Per-row arithmetic follows the slice reduction, then the Triton sinkhorn.
    """
    row = tl.program_id(0)
    if row >= m:
        return
    j = tl.arange(0, HC)
    jj = j[:, None]
    kk = j[None, :]

    a_pre = tl.zeros([HC], dtype=tl.float32)
    a_post = tl.zeros([HC], dtype=tl.float32)
    a_comb = tl.zeros([HC, HC], dtype=tl.float32)
    sq = tl.zeros([], dtype=tl.float32)
    for s in tl.static_range(NUM_SLICES):
        off = (s * m + row) * MIX
        a_pre += tl.load(part_mix_ptr + off + j)
        a_post += tl.load(part_mix_ptr + off + HC + j)
        a_comb += tl.load(part_mix_ptr + off + 2 * HC + jj * HC + kk)
        sq += tl.load(part_sq_ptr + s * m + row)
    rsqrt = 1.0 / tl.sqrt(sq * inv_k + rms_eps)

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    pre = tl.sigmoid(a_pre * rsqrt * s0 + tl.load(base_ptr + j)) + EPS
    tl.store(pre_ptr + row * HC + j, pre)
    post = 2.0 * tl.sigmoid(a_post * rsqrt * s1 + tl.load(base_ptr + HC + j))
    tl.store(post_ptr + row * HC + j, post)

    comb = a_comb * rsqrt * s2 + tl.load(base_ptr + 2 * HC + jj * HC + kk)
    comb = tl.exp(comb - tl.max(comb, axis=1)[:, None])
    comb = comb / tl.sum(comb, axis=1)[:, None] + EPS
    comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
    for _ in tl.static_range(ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + EPS)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
    tl.store(comb_ptr + row * HC * HC + jj * HC + kk, comb)


def hc_mix_stats_sinkhorn(
    x_flat: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    rms_eps: float,
    hc_eps: float,
):
    """Fuse the reduce and sinkhorn stages of hc_mix_stats followed by hc_split_sinkhorn.

    The split-K kernel fixes the reduction order and preserves batch invariance.
    Sinkhorn uses the Triton port's transcendental lowering, which differs from TileLang.
    """
    assert x_flat.dim() == 2 and hc_fn.dim() == 2
    assert x_flat.stride(1) == 1 and hc_fn.stride(1) == 1
    assert hc_fn.dtype == torch.float32
    m, k = x_flat.shape
    mix = hc_fn.shape[0]
    assert mix == (2 + hc_mult) * hc_mult and hc_fn.shape[1] == k
    dev = x_flat.device
    pre = torch.empty(m, hc_mult, dtype=torch.float32, device=dev)
    post = torch.empty(m, hc_mult, dtype=torch.float32, device=dev)
    comb = torch.empty(m, hc_mult, hc_mult, dtype=torch.float32, device=dev)
    if m == 0:
        return pre, post, comb

    num_slices = _num_slices_for(k)
    mix_pad = max(16, triton.next_power_of_2(mix))
    part_mix = torch.empty((num_slices, m, mix), dtype=torch.float32, device=dev)
    part_sq = torch.empty((num_slices, m), dtype=torch.float32, device=dev)
    block_m = _block_m_for(m)
    _hc_mix_stats_partial_kernel[(triton.cdiv(m, block_m), num_slices)](
        x_flat,
        hc_fn,
        part_mix,
        part_sq,
        m,
        k,
        x_flat.stride(0),
        hc_fn.stride(0),
        MIX=mix,
        MIX_PAD=mix_pad,
        NUM_SLICES=num_slices,
        BLOCK_M=block_m,
        BLOCK_K=_HC_MIX_BLOCK_K,
        DOT_PRECISION=_HC_MIX_DOT_PRECISION,
        num_warps=_HC_MIX_NUM_WARPS,
        num_stages=_HC_MIX_NUM_STAGES,
    )
    _hc_mix_reduce_sinkhorn_kernel[(m,)](
        part_mix,
        part_sq,
        hc_scale.float().contiguous(),
        hc_base.float().contiguous(),
        pre,
        post,
        comb,
        m,
        1.0 / k,
        rms_eps,
        MIX=mix,
        HC=hc_mult,
        NUM_SLICES=num_slices,
        ITERS=sinkhorn_iters,
        EPS=hc_eps,
        num_warps=1,
    )
    return pre, post, comb


def hc_combine(
    x_flat: torch.Tensor, pre: torch.Tensor, hc: int, out_dtype: torch.dtype
) -> torch.Tensor:
    """Fused y[m, h] = sum_k pre[m, k] * x_flat[m, k*H + h]."""
    m = x_flat.shape[0]
    h = x_flat.shape[1] // hc
    y = torch.empty((m, h), dtype=out_dtype, device=x_flat.device)
    if m == 0 or h == 0:
        return y
    block_h = 1024
    _hc_combine_kernel[(m, triton.cdiv(h, block_h))](
        x_flat,
        pre,
        y,
        h,
        x_flat.stride(0),
        pre.stride(0),
        pre.stride(1),
        y.stride(0),
        HC=hc,
        BLOCK_H=block_h,
    )
    return y
