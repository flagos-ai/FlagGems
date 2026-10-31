# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit FP8 Q/K, BF16 head weights and per-query key visibility scoring.

Preserves BF16 round points after each dot, weighted ReLU and head sum.
The eight-query schedule reuses K with the qualified lossless F16 operand
casts. FP8 input storage does not imply native FP8 MMA in that schedule.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _plain_score(
    Q,
    W,
    K,
    Lens,
    Out,
    M: tl.constexpr,
    L: tl.constexpr,
    OUT_L: tl.constexpr,
    BN: tl.constexpr,
    GROUP: tl.constexpr,
    STORE_CACHE: tl.constexpr,
):
    nn = tl.cdiv(OUT_L, BN)
    if GROUP == 0:
        row = tl.program_id(0)
        lb = tl.program_id(1)
    else:
        pid = tl.program_id(0)
        first = pid // (GROUP * nn) * GROUP
        gm = tl.minimum(M - first, GROUP)
        row = first + pid % gm
        lb = (pid % (GROUP * nn)) // gm
    cols = lb * BN + tl.arange(0, BN)
    h = tl.arange(0, 32)
    d = tl.arange(0, 128)
    visible = tl.minimum(tl.load(Lens + row), L)
    valid = (cols < visible) & (cols < OUT_L)
    if lb * BN < visible:
        q = tl.load(Q + row * 4096 + h[:, None] * 128 + d[None, :])
        keys = tl.load(K + cols[:, None] * 128 + d[None, :], valid[:, None], other=0.0)
        acc = tl.dot(q, tl.trans(keys), out_dtype=tl.float32, max_num_imprecise_acc=0)
        s = acc.to(tl.bfloat16).to(tl.float32)
        s = tl.maximum(s, 0.0)
        weights = tl.load(W + row * 32 + h).to(tl.float32)
        s = (s * weights[:, None]).to(tl.bfloat16).to(tl.float32)
        logits = tl.sum(s, 0).to(tl.bfloat16).to(tl.float32)
        logits = tl.where(valid, logits, float("-inf"))
    else:
        logits = tl.full((BN,), float("-inf"), tl.float32)
    tl.store(Out + row * OUT_L + cols, logits, cols < OUT_L, cache_modifier=STORE_CACHE)


@triton.jit
def _score(
    Q,
    W,
    K,
    LENS,
    OUT,
    M: tl.constexpr,
    L: tl.constexpr,
    OL: tl.constexpr,
    BN: tl.constexpr,
    QG: tl.constexpr,
    STORE: tl.constexpr,
    LOOP_PIPE: tl.constexpr,
):
    first = tl.program_id(0) * QG
    tile = tl.program_id(1)
    rr = first + tl.arange(0, QG)
    visible_rows = tl.minimum(tl.load(LENS + rr, rr < M, other=0), L)
    max_visible = tl.max(visible_rows, 0)
    n = tile * BN + tl.arange(0, BN)
    h = tl.arange(0, 32)
    d = tl.arange(0, 128)
    if tile * BN < max_visible:
        # All visible columns of every query see the same source K values.
        # Per-query visibility is applied to final scores. Invisible columns
        # cannot affect another column's independent dot accumulator.
        key = tl.load(
            K + n[:, None] * 128 + d[None, :], (n < max_visible)[:, None], other=0.0
        ).to(tl.float16)
        for offset in tl.range(0, QG, num_stages=LOOP_PIPE, loop_unroll_factor=1):
            row = first + offset
            visible = tl.minimum(tl.load(LENS + row, row < M, other=0), L)
            if (row < M) & (tile * BN < visible):
                q = tl.load(Q + row * 4096 + h[:, None] * 128 + d[None, :]).to(
                    tl.float16
                )
                weights = tl.load(W + row * 32 + h).to(tl.float32)
                acc = tl.dot(
                    q, tl.trans(key), out_dtype=tl.float32, max_num_imprecise_acc=0
                )
                s = tl.maximum(acc.to(tl.bfloat16).to(tl.float32), 0.0)
                s = (s * weights[:, None]).to(tl.bfloat16).to(tl.float32)
                logits = tl.sum(s, 0).to(tl.bfloat16).to(tl.float32)
                logits = tl.where(n < visible, logits, float("-inf"))
            else:
                logits = tl.full((BN,), float("-inf"), tl.float32)
            tl.store(
                OUT + row * OL + n, logits, (row < M) & (n < OL), cache_modifier=STORE
            )
    else:
        for offset in tl.range(0, QG, num_stages=LOOP_PIPE, loop_unroll_factor=1):
            row = first + offset
            tl.store(
                OUT + row * OL + n,
                float("-inf"),
                (row < M) & (n < OL),
                cache_modifier=STORE,
            )


def fp8_prefill_score(q, weights, keys, lens, *, reuse_queries=False):
    if not (
        q.ndim == 3
        and q.shape[1:] == (32, 128)
        and keys.ndim == 2
        and keys.shape[1] == 128
        and weights.shape == (q.shape[0], 32)
        and lens.shape == (q.shape[0],)
        and q.dtype == keys.dtype == torch.float8_e4m3fn
        and weights.dtype == torch.bfloat16
        and lens.dtype in (torch.int32, torch.int64)
        and q.is_cuda
        and all(
            t.device == q.device and t.is_contiguous() for t in (q, weights, keys, lens)
        )
        and q.shape[0] > 0
        and keys.shape[0] > 0
        and torch.cuda.get_device_capability(q.device) == (9, 0)
        and triton.__version__.split("+")[0] == "3.6.0"
    ):
        raise ValueError("unsupported H100/FlagTree 3.6 FP8 score contract")
    m, length = q.shape[0], keys.shape[0]
    width = triton.cdiv(length, 4) * 4
    out = torch.empty((m, width), dtype=torch.float32, device=q.device)
    if reuse_queries:
        _score[(triton.cdiv(m, 8), triton.cdiv(width, 128))](
            q,
            weights,
            keys,
            lens,
            out,
            m,
            length,
            width,
            128,
            8,
            ".cs",
            2,
            num_warps=4,
            num_stages=3,
            enable_fp_fusion=False,
        )
    else:
        _plain_score[(m, triton.cdiv(width, 128))](
            q,
            weights,
            keys,
            lens,
            out,
            m,
            length,
            width,
            128,
            0,
            "",
            num_warps=4,
            num_stages=3,
            enable_fp_fusion=False,
        )
    return out
