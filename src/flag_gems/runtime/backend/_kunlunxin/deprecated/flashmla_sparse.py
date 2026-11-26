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
"""Kunlunxin (TritonXPU) specialization of ``flash_mla_sparse_fwd``.

The generic implementation in ``flag_gems/fused/flashmla_sparse.py`` fuses, inside
one Triton kernel, (a) a data-dependent gather of KV rows through ``indices``,
(b) three ``tl.dot`` calls and (c) ``tl.math.exp``/``tl.max`` softmax reductions.
On this backend that combination is broken:

* ``tl.dot`` + data-dependent gather in the same kernel: hard compile failure
  (``ConvertTritonSDNNToLLVM`` / ``PassManager::run failed``) in the isolated
  reproducer, and silently wrong values in the generic kernel shape.
* ``tl.dot`` + ``tl.max``/``tl.math.exp`` in the same kernel: compiles but
  returns silently wrong values (measured relative error 4.9e-1).
* ``tl.dot`` + softmax + a second ``tl.dot``: hard compile failure.

So the op is split into four kernels, none of which mixes ``tl.dot`` with either
a data-dependent address or a transcendental/reduction:

  A ``_gather_kv``  : gather (no ``tl.dot``)      -> dense KV, both layouts
  B ``_qk_logits``  : dense ``tl.dot`` only       -> unscaled logits
  C ``_softmax_stats``: reductions/exp (no ``tl.dot``) -> max_logits, lse, probs
  D ``_pv_matmul``  : dense ``tl.dot`` only       -> output

All intermediate buffers are over-allocated to whole tile boundaries so that
every store is unmasked (masked stores are known to write past tight
allocations on this backend).
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Tile sizes. 64 is the smallest value that is safe on this backend
# (BLOCK_N == 16 does not compile, 32 silently corrupts pointwise tiles, and 2D
# tiles with a row pitch < 64 silently overwrite following rows).
_BT = 64  # topk tile
_BD = 64  # d tile used by the gather kernel
_BH = 64  # head tile
_BDV = 256  # value-dim tile used by the PV matmul


@triton.jit
def _load_ids(
    indices,
    topk_length,
    i_sq,
    offs_t,
    stride_tm,
    SKV,
    TOPK,
    HAVE_TOPK_LENGTH: tl.constexpr,
):
    limit = TOPK
    if HAVE_TOPK_LENGTH:
        tl_len = tl.load(topk_length + i_sq)
        limit = tl.minimum(limit, tl_len)
    in_range = offs_t < limit
    # clamp the address instead of using ``other=``: a masked load whose fill value
    # carries semantics (an invalid-index sentinel) is not reliable on this backend.
    t_off = tl.minimum(offs_t, TOPK - 1)
    ids = tl.load(indices + i_sq * stride_tm + t_off)
    m = in_range & (ids >= 0) & (ids < SKV)
    # clamp so the gather address is always inside kv (also prevents mul overflow)
    return tl.where(m, ids, 0).to(tl.int64), m


@triton.jit
def _valid_mask(
    indices,
    topk_length,
    valid,  # [SQ, TP] float32
    stride_tm,
    SKV,
    TOPK,
    TP: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    BT: tl.constexpr,
):
    i_sq = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1)
    offs_t = i_t * BT + tl.arange(0, BT)
    _, m = _load_ids(
        indices, topk_length, i_sq, offs_t, stride_tm, SKV, TOPK, HAVE_TOPK_LENGTH
    )
    tl.store(valid + i_sq * TP + offs_t, m.to(tl.float32))


@triton.jit
def _gather_kv_dt(
    kv,
    indices,
    topk_length,
    gkv_dt,  # [SQ, DQK, TP], d-major (topk contiguous)
    stride_kvn,
    stride_tm,
    SKV,
    TOPK,
    TP: tl.constexpr,
    DQK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_sq = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1)
    i_d = tl.program_id(2)
    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = i_d * BD + tl.arange(0, BD)
    ids, _ = _load_ids(
        indices, topk_length, i_sq, offs_t, stride_tm, SKV, TOPK, HAVE_TOPK_LENGTH
    )
    # [BD, BT] tile: outer stride 1 (d contiguous in kv), inner stride stride_kvn
    v = tl.load(kv + offs_d[:, None] + ids[None, :] * stride_kvn)
    tl.store(gkv_dt + i_sq * DQK * TP + offs_d[:, None] * TP + offs_t[None, :], v)


@triton.jit
def _gather_kv_td(
    kv,
    indices,
    topk_length,
    gkv_td,  # [SQ, TP, DQK], t-major (d contiguous)
    stride_kvn,
    stride_tm,
    SKV,
    TOPK,
    TP: tl.constexpr,
    DQK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_sq = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1)
    i_d = tl.program_id(2)
    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = i_d * BD + tl.arange(0, BD)
    ids, _ = _load_ids(
        indices, topk_length, i_sq, offs_t, stride_tm, SKV, TOPK, HAVE_TOPK_LENGTH
    )
    # [BT, BD] tile: rows are gathered kv rows, d contiguous -> no tl.trans needed
    v = tl.load(kv + ids[:, None] * stride_kvn + offs_d[None, :])
    tl.store(gkv_td + i_sq * TP * DQK + offs_t[:, None] * DQK + offs_d[None, :], v)


@triton.jit
def _qk_logits(
    q,
    gkv_dt,
    logits,  # [SQ, HQ, TP] float32
    stride_qm,
    stride_qh,
    TP: tl.constexpr,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    DP: tl.constexpr,  # 512
    TD: tl.constexpr,  # DQK - 512, 0 or 64
    BH: tl.constexpr,
    BT: tl.constexpr,
):
    i_sq = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1)
    i_t = tl.program_id(2)

    offs_h = i_h * BH + tl.arange(0, BH)
    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, DP)

    qb = tl.load(q + i_sq * stride_qm + offs_h[:, None] * stride_qh + offs_d[None, :])
    kb = tl.load(gkv_dt + i_sq * DQK * TP + offs_d[:, None] * TP + offs_t[None, :])
    acc = tl.dot(qb, kb, out_dtype=tl.float32)
    if TD > 0:
        offs_td = DP + tl.arange(0, TD)
        qt = tl.load(
            q + i_sq * stride_qm + offs_h[:, None] * stride_qh + offs_td[None, :]
        )
        kt = tl.load(gkv_dt + i_sq * DQK * TP + offs_td[:, None] * TP + offs_t[None, :])
        acc = tl.dot(qt, kt, acc, out_dtype=tl.float32)

    tl.store(
        logits + i_sq * HQ * TP + offs_h[:, None] * TP + offs_t[None, :],
        acc,
    )


@triton.jit
def _softmax_stats(
    logits,
    valid,
    probs,  # [SQ, HQ, TP] bfloat16
    max_logits,  # [SQ, HQ] float32
    lse,  # [SQ, HQ] float32
    attn_sink,
    sm_scale,
    stride_mm,
    stride_lm,
    TP: tl.constexpr,
    NT: tl.constexpr,
    HQ: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    BT: tl.constexpr,
):
    # One program per (s_q, head). Flat 1D tiles only: mixing a [BH] accumulator
    # with the result of a 2D ``tl.max(..., axis=1)`` makes TritonXPUCoreTiling
    # reject the module ('arith.maxnumf' requires the same encoding).
    i_sq = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1)

    offs_t = tl.arange(0, BT)
    lg_base = logits + i_sq * HQ * TP + i_h * TP
    p_base = probs + i_sq * HQ * TP + i_h * TP
    v_base = valid + i_sq * TP

    # pass 1: max over topk, folded per BT tile (never one wide reduction tile)
    run_max = float("-inf")
    for it in range(NT):
        vv = tl.load(v_base + it * BT + offs_t)
        x = tl.load(lg_base + it * BT + offs_t) * sm_scale
        x = tl.where(vv > 0.0, x, float("-inf"))
        run_max = tl.maximum(run_max, tl.max(x))

    has_valid = run_max != float("-inf")
    safe_max = tl.where(has_valid, run_max, 0.0)

    # pass 2: sum of exp
    run_sum = 0.0
    for it in range(NT):
        vv = tl.load(v_base + it * BT + offs_t)
        x = tl.load(lg_base + it * BT + offs_t) * sm_scale
        x = tl.where(vv > 0.0, x, float("-inf"))
        run_sum += tl.sum(tl.math.exp(x - safe_max))

    orig_lse = tl.where(has_valid, safe_max + tl.math.log(run_sum), float("-inf"))

    lse_for_o = orig_lse
    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink + i_h)
        # logaddexp(orig_lse, sink), inf-safe
        big = tl.maximum(orig_lse, sink)
        small = tl.minimum(orig_lse, sink)
        lse_for_o = tl.where(
            big == float("inf"),
            float("inf"),
            tl.where(
                big == float("-inf"),
                float("-inf"),
                big + tl.math.log(1.0 + tl.math.exp(small - big)),
            ),
        )
    # -inf lse means "no valid key": force +inf so exp(x - lse) == 0
    lse_for_o = tl.where(lse_for_o == float("-inf"), float("inf"), lse_for_o)

    tl.store(max_logits + i_sq * stride_mm + i_h, run_max)
    tl.store(
        lse + i_sq * stride_lm + i_h,
        tl.where(has_valid, orig_lse, float("inf")),
    )

    # pass 3: probabilities, already normalized by lse_for_o
    for it in range(NT):
        vv = tl.load(v_base + it * BT + offs_t)
        x = tl.load(lg_base + it * BT + offs_t) * sm_scale
        x = tl.where(vv > 0.0, x, float("-inf"))
        p = tl.math.exp(x - lse_for_o)
        p = tl.where(vv > 0.0, p, 0.0)
        tl.store(p_base + it * BT + offs_t, p.to(tl.bfloat16))


@triton.jit
def _pv_matmul(
    probs,
    gkv_td,
    out,
    stride_om,
    stride_oh,
    TP: tl.constexpr,
    NT: tl.constexpr,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    DV: tl.constexpr,
    BH: tl.constexpr,
    BT: tl.constexpr,
    BDV: tl.constexpr,
):
    i_sq = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1)
    i_v = tl.program_id(2)

    offs_h = i_h * BH + tl.arange(0, BH)
    offs_t = tl.arange(0, BT)
    offs_v = i_v * BDV + tl.arange(0, BDV)

    p_base = probs + i_sq * HQ * TP + offs_h[:, None] * TP
    k_base = gkv_td + i_sq * TP * DQK

    acc = tl.zeros([BH, BDV], dtype=tl.float32)
    for it in range(NT):
        pb = tl.load(p_base + it * BT + offs_t[None, :])
        kb = tl.load(k_base + (it * BT + offs_t)[:, None] * DQK + offs_v[None, :])
        acc = tl.dot(pb, kb, acc, out_dtype=tl.float32)

    tl.store(
        out + i_sq * stride_om + offs_h[:, None] * stride_oh + offs_v[None, :],
        acc.to(tl.bfloat16),
    )


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse MLA prefill forward. See the generic implementation for the contract."""
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    assert (
        q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and indices.dtype == torch.int32
    )
    SQ, HQ, DQK = q.shape
    SKV, HKV, _ = kv.shape
    assert d_v == 512, "Unsupported d_v"
    DV = d_v
    assert kv.shape[-1] == DQK
    _, _, TOPK = indices.shape
    assert indices.shape == (SQ, HKV, TOPK)
    if attn_sink is not None:
        assert attn_sink.is_contiguous()
        assert attn_sink.dtype == torch.float32
        assert attn_sink.shape == (HQ,), "attn_sink error shape"
    if topk_length is not None:
        assert topk_length.is_contiguous()
        assert topk_length.dtype == torch.int32
        assert topk_length.shape == (SQ,), "topk_length error shape"
    assert HKV == 1, "h_kv is expected to be 1"
    assert HQ == 64 or HQ == 128, "Unsupported h_q"
    assert DQK == 576 or DQK == 512, "Unsupported d_qk"

    output = torch.empty((SQ, HQ, DV), device=q.device, dtype=q.dtype)
    max_logits = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
    lse = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
    if SQ == 0 or TOPK == 0:
        # nothing to attend to: reference semantics are all-zero O, -inf max, +inf lse
        output.zero_()
        max_logits.fill_(float("-inf"))
        lse.fill_(float("inf"))
        return output, max_logits, lse

    DP = 512
    TD = DQK - DP
    # over-allocate topk to a whole tile so every store below is unmasked
    TP = triton.cdiv(TOPK, _BT) * _BT
    NT = TP // _BT

    gkv_dt = torch.empty((SQ, DQK, TP), device=q.device, dtype=q.dtype)
    gkv_td = torch.empty((SQ, TP, DQK), device=q.device, dtype=q.dtype)
    valid = torch.empty((SQ, TP), device=q.device, dtype=torch.float32)
    logits = torch.empty((SQ, HQ, TP), device=q.device, dtype=torch.float32)
    probs = torch.empty((SQ, HQ, TP), device=q.device, dtype=q.dtype)

    _valid_mask[(SQ, NT)](
        indices,
        topk_length,
        valid,
        indices.stride(0),
        SKV,
        TOPK,
        TP,
        topk_length is not None,
        _BT,
    )
    _gather_kv_dt[(SQ, NT, DQK // _BD)](
        kv,
        indices,
        topk_length,
        gkv_dt,
        kv.stride(0),
        indices.stride(0),
        SKV,
        TOPK,
        TP,
        DQK,
        topk_length is not None,
        _BT,
        _BD,
    )
    _gather_kv_td[(SQ, NT, DQK // _BD)](
        kv,
        indices,
        topk_length,
        gkv_td,
        kv.stride(0),
        indices.stride(0),
        SKV,
        TOPK,
        TP,
        DQK,
        topk_length is not None,
        _BT,
        _BD,
    )
    _qk_logits[(SQ, HQ // _BH, NT)](
        q,
        gkv_dt,
        logits,
        q.stride(0),
        q.stride(1),
        TP,
        HQ,
        DQK,
        DP,
        TD,
        _BH,
        _BT,
    )
    _softmax_stats[(SQ, HQ)](
        logits,
        valid,
        probs,
        max_logits,
        lse,
        attn_sink,
        sm_scale,
        max_logits.stride(0),
        lse.stride(0),
        TP,
        NT,
        HQ,
        attn_sink is not None,
        _BT,
    )
    _pv_matmul[(SQ, HQ // _BH, DV // _BDV)](
        probs,
        gkv_td,
        output,
        output.stride(0),
        output.stride(1),
        TP,
        NT,
        HQ,
        DQK,
        DV,
        _BH,
        _BT,
        _BDV,
    )
    return output, max_logits, lse
