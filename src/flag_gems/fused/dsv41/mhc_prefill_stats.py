# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact per-call K-major weight preparation; no tensor/weight/workspace cache."""

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_lhs_tf32x3(x, w):
    high = tl.inline_asm_elementwise(
        "cvt.rna.tf32.f32 $0, $1;",
        constraints="=r,r",
        args=[w],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    # Preserve the omitted x_low MMA's nonfinite effects at the original mask.
    low = tl.where(tl.abs(high) == float("inf"), float("nan"), w - high)
    correction_x = tl.where(tl.abs(x) == float("inf"), float("nan"), x)
    correction = tl.dot(correction_x, low, input_precision="tf32")
    correction = tl.where(correction == correction, correction, 0.0)
    return tl.dot(x, high, correction, input_precision="tf32")


@triton.jit
def _packed_bf16_lhs_tf32x3(x, high, low):
    correction_x = tl.where(tl.abs(x) == float("inf"), float("nan"), x)
    correction = tl.dot(correction_x, low, input_precision="tf32")
    correction = tl.where(correction == correction, correction, 0.0)
    return tl.dot(x, high, correction, input_precision="tf32")


@triton.jit
def _pack_weight_kernel(
    W, HIGH, LOW, K, WS, TRANSPOSE: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if TRANSPOSE:
        n = offsets % 32
        k = offsets // 32
        valid = (n < 24) & (k < K)
    else:
        n = offsets // K
        k = offsets % K
        valid = n < 24
    value = tl.load(W + n * WS + k, valid, 0.0)
    high = tl.inline_asm_elementwise(
        "cvt.rna.tf32.f32 $0, $1;",
        constraints="=r,r",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    low = tl.where(tl.abs(high) == float("inf"), float("nan"), value - high)
    bound = K * 32 if TRANSPOSE else K * 24
    tl.store(HIGH + offsets, high, offsets < bound)
    tl.store(LOW + offsets, low, offsets < bound)


@triton.jit
def _partial_kernel(
    X,
    W,
    HIGH,
    LOW,
    PM,
    PS,
    M,
    K,
    XS,
    WS,
    BM: tl.constexpr,
    BN: tl.constexpr,
    LAYOUT: tl.constexpr,
    PACK: tl.constexpr,
    ORDER: tl.constexpr,
):
    if BN == 32:
        mix_group = 0
        pid = tl.program_id(0)
    else:
        mix_group = tl.program_id(0) % (32 // BN)
        pid = tl.program_id(0) // (32 // BN)
    if ORDER == 0:
        pidm = pid
        sid = tl.program_id(1)
    elif ORDER == 1:
        pidm = tl.program_id(1)
        sid = pid
    else:
        row_blocks = tl.cdiv(M, BM)
        group = pid // (8 * 80)
        group_first = group * 8
        group_size = tl.minimum(row_blocks - group_first, 8)
        group_offset = pid % (8 * 80)
        pidm = group_first + group_offset % group_size
        sid = group_offset // group_size
    local_rows = tl.arange(0, BM)
    rows = pidm * BM + local_rows
    cols = mix_group * BN + tl.arange(0, BN)
    kr = tl.arange(0, 64)
    span = K // 80
    start = sid * span
    acc = tl.zeros((BM, BN), tl.float32)
    sq = tl.zeros((BM,), tl.float32)
    # Keep K runtime, as in the accepted baseline. Previous const-K unrolling
    # changed the pinned compiler's reduction behavior and is not reproduced.
    for kb in range(0, span, 64):
        ks = start + kb + kr
        mk = ks < start + span
        x = tl.load(
            X + rows[:, None] * XS + ks[None, :], (rows[:, None] < M) & mk[None, :], 0.0
        ).to(tl.float32)
        if PACK == 0:
            w = tl.load(
                W + cols[None, :] * WS + ks[:, None],
                (cols[None, :] < 24) & mk[:, None],
                0.0,
            ).to(tl.float32)
            acc += _bf16_lhs_tf32x3(x, w)
        else:
            if PACK == 1:
                wp = cols[None, :] * K + ks[:, None]
            else:
                wp = ks[:, None] * 32 + cols[None, :]
            mask = (cols[None, :] < 24) & mk[:, None]
            high = tl.load(HIGH + wp, mask, 0.0)
            low = tl.load(LOW + wp, mask, 0.0)
            acc += _packed_bf16_lhs_tf32x3(x, high, low)
        if BN == 32:
            sq += tl.sum(x * x, 1)
        else:
            # Only one independent N tile owns each square partial. No K
            # regrouping or atomic reduction is introduced by output splitting.
            if mix_group == 0:
                sq += tl.sum(x * x, 1)
    if LAYOUT == 0:
        # Match the accepted baseline's 2-D address expression exactly. An
        # equivalent 1-D expression can steer layout/vectorization differently.
        tl.store(
            PM + (sid * M + rows[:, None]) * 24 + cols[None, :],
            acc,
            (rows[:, None] < M) & (cols[None, :] < 24),
        )
        sq_offset = sid * M + rows
    elif LAYOUT == 1:
        sq_offset = (pidm * 80 + sid) * BM + local_rows
        mix_base = sq_offset * 24
        tl.store(
            PM + mix_base[:, None] + cols[None, :],
            acc,
            (rows[:, None] < M) & (cols[None, :] < 24),
        )
    else:
        sq_offset = rows * 80 + sid
        mix_base = sq_offset * 24
        tl.store(
            PM + mix_base[:, None] + cols[None, :],
            acc,
            (rows[:, None] < M) & (cols[None, :] < 24),
        )
    if BN == 32:
        tl.store(PS + sq_offset, sq, rows < M)
    else:
        if mix_group == 0:
            tl.store(PS + sq_offset, sq, rows < M)


@triton.jit
def _reduce_kernel(PM, PS, Y, M, INV_K, EPS, BM: tl.constexpr, LAYOUT: tl.constexpr):
    pidm = tl.program_id(0)
    local_rows = tl.arange(0, BM)
    rows = pidm * BM + local_rows
    cols = tl.arange(0, 32)
    acc = tl.zeros((BM, 32), tl.float32)
    sq = tl.zeros((BM,), tl.float32)
    for s in tl.static_range(80):
        if LAYOUT == 0:
            acc += tl.load(
                PM + (s * M + rows[:, None]) * 24 + cols[None, :],
                (rows[:, None] < M) & (cols[None, :] < 24),
                0.0,
            )
            sq += tl.load(PS + s * M + rows, rows < M, 0.0)
        elif LAYOUT == 1:
            sq_offset = (pidm * 80 + s) * BM + local_rows
            acc += tl.load(
                PM + sq_offset[:, None] * 24 + cols[None, :],
                (rows[:, None] < M) & (cols[None, :] < 24),
                0.0,
            )
            sq += tl.load(PS + sq_offset, rows < M, 0.0)
        else:
            sq_offset = rows * 80 + s
            acc += tl.load(
                PM + sq_offset[:, None] * 24 + cols[None, :],
                (rows[:, None] < M) & (cols[None, :] < 24),
                0.0,
            )
            sq += tl.load(PS + sq_offset, rows < M, 0.0)
    inv = 1.0 / tl.sqrt(sq * INV_K + EPS)
    tl.store(
        Y + rows[:, None] * 24 + cols[None, :],
        acc * inv[:, None],
        (rows[:, None] < M) & (cols[None, :] < 24),
    )


def run_stats(x, w, eps, *, return_parts=False, return_kernels=False):
    if not (
        isinstance(x, torch.Tensor)
        and isinstance(w, torch.Tensor)
        and x.ndim == w.ndim == 2
        and x.shape[0] in (4096, 8192, 15360, 16384, 32768)
        and x.shape[1] == 20480
        and w.shape == (24, 20480)
        and x.dtype == torch.bfloat16
        and w.dtype == torch.float32
        and x.is_cuda
        and w.device == x.device
        and x.is_contiguous()
        and w.is_contiguous()
        and type(eps) is float
        and eps == 1e-20
        and triton.__version__.split("+")[0] == "3.6.0"
        and torch.cuda.get_device_capability(x.device) == (9, 0)
    ):
        raise ValueError("unsupported H100/FlagTree 3.6 mHC prefill stats contract")
    m, k = x.shape
    width = k * 32
    packed = torch.empty((2, width), device=x.device, dtype=torch.float32)
    high, low = packed[0], packed[1]
    pack = _pack_weight_kernel[(triton.cdiv(width, 256),)](
        w,
        high,
        low,
        k,
        w.stride(0),
        TRANSPOSE=True,
        BLOCK=256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    pm = torch.empty((80, m, 24), device=x.device, dtype=torch.float32)
    ps = torch.empty((80, m), device=x.device, dtype=torch.float32)
    out = torch.empty((m, 24), device=x.device, dtype=torch.float32)
    partial = _partial_kernel[(triton.cdiv(m, 16), 80)](
        x,
        w,
        high,
        low,
        pm,
        ps,
        m,
        k,
        x.stride(0),
        w.stride(0),
        BM=16,
        BN=32,
        LAYOUT=0,
        PACK=2,
        ORDER=0,
        num_warps=2,
        num_stages=2,
    )
    reduce = _reduce_kernel[(triton.cdiv(m, 16),)](
        pm,
        ps,
        out,
        m,
        1.0 / k,
        eps,
        BM=16,
        LAYOUT=0,
        num_warps=4,
    )
    if return_kernels:
        return out, pm, ps, {"pack": pack, "partial": partial, "reduce": reduce}
    return (out, pm, ps) if return_parts else out
