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
"""Kunlunxin (TritonXPU) specialization of ``triton_sparse_mla_fwd_interface``.

The generic implementation in ``flag_gems/fused/DSA/sparse_mla.py`` runs one
fused kernel that mixes ``tl.dot`` with a data-dependent gather through
``indices`` and with ``tl.max``/``tl.math.exp2`` softmax reductions.  On this
backend that combination is broken (see ``flashmla_sparse.py`` for the same
finding on the sibling op):

* ``tl.dot`` + data-dependent gather in the same kernel: hard compile failure.
* ``tl.dot`` + ``tl.max``/``tl.math.exp`` in the same kernel: compiles but
  silently returns wrong values.
* the untyped ``qk`` accumulator (fp16 initial value, fp32 redefinition after
  ``* log_scale``) is rejected by the XPU frontend type unification:
  ``initial value for `qk` is of type fp16[...], but the then block redefines
  it as fp32[...]``.

So the op is split into four kernels, none of which mixes ``tl.dot`` with a
data-dependent address or a transcendental/reduction:

  A ``_spmla_gather_dt``/``_spmla_gather_td``: gather (no ``tl.dot``)
  B ``_spmla_qk``   : dense ``tl.dot`` only                    -> logits
  C ``_spmla_softmax``: reductions/exp (no ``tl.dot``)         -> probs, lse
  D ``_spmla_pv``   : dense ``tl.dot`` only                    -> output

The causal mask (keys ``n <= query``) is applied once, in the softmax kernel.
All (b, sq, g) groups of the same ``indices`` gather into the same dense KV
buffers.  All intermediate buffers are over-allocated to whole tile
boundaries (``TP = cdiv(topk, 64) * 64`` topk, ``AH`` head blocks, exact
``BD``/``BDV`` dividers) so that every store is unmasked (masked stores are
known to write past tight allocations on this backend).  When a head block
overruns ``H`` (possible only for the tiny ``G < 16`` edge shapes), the output
is written into a padded buffer and the interface returns a slimmed view of
it.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Tile sizes. 64 is the smallest value that is safe on this backend
# (BLOCK_N == 16 does not compile, 32 silently corrupts pointwise tiles, and 2D
# tiles with a row pitch < 64 silently overwrite following rows).
_BT = 64  # topk tile
_BD = 64  # d tile used by the gather kernel
_BH = 64  # head tile
_BDV = 256  # value-dim tile used by the PV matmul


@triton.jit
def _spmla_gather_dt(
    kv,
    indices,
    gkv_dt,  # [B*SQ*VG, DT, TP], t-contiguous
    stride_kvb,
    stride_kvg,
    stride_kvn,
    SKV,
    SQC,
    VGC,
    TOPK,
    DT: tl.constexpr,
    TP: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    ND: tl.constexpr,
):
    i0 = tl.program_id(0).to(tl.int64)  # B*SQ
    i_g = tl.program_id(1)
    i_z = tl.program_id(2)  # NT*ND
    i_d = i_z % ND
    i_t = i_z // ND
    i_b = i0 // SQC
    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = i_d * BD + tl.arange(0, BD)
    in_range = offs_t < TOPK
    # clamp the address instead of using ``other=``: a masked load whose fill
    # value carries semantics (an invalid-index sentinel) is not reliable here.
    t_off = tl.minimum(offs_t, TOPK - 1)
    ids = tl.load(
        indices + i0 * (VGC * TOPK) + i_g * TOPK + t_off
    ).to(tl.int64)
    m = in_range & (ids >= 0) & (ids < SKV)
    ids_safe = tl.where(m, ids, 0)
    # [BD, BT] tile: outer stride 1 (d contiguous), inner stride stride_kvn
    v = tl.load(
        kv
        + i_b * stride_kvb
        + i_g * stride_kvg
        + ids_safe[None, :] * stride_kvn
        + offs_d[:, None]
    )
    tl.store(
        gkv_dt + (i0 * VGC + i_g) * (DT * TP) + offs_d[:, None] * TP + offs_t[None, :],
        v,
    )


@triton.jit
def _spmla_gather_td(
    kv,
    indices,
    gkv_td,  # [B*SQ*VG, TP, DT], d-contiguous
    stride_kvb,
    stride_kvg,
    stride_kvn,
    SKV,
    SQC,
    VGC,
    TOPK,
    DT: tl.constexpr,
    TP: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    ND: tl.constexpr,
):
    i0 = tl.program_id(0).to(tl.int64)  # B*SQ
    i_g = tl.program_id(1)
    i_z = tl.program_id(2)  # NT*ND
    i_d = i_z % ND
    i_t = i_z // ND
    i_b = i0 // SQC
    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = i_d * BD + tl.arange(0, BD)
    in_range = offs_t < TOPK
    t_off = tl.minimum(offs_t, TOPK - 1)
    ids = tl.load(
        indices + i0 * (VGC * TOPK) + i_g * TOPK + t_off
    ).to(tl.int64)
    m = in_range & (ids >= 0) & (ids < SKV)
    ids_safe = tl.where(m, ids, 0)
    # [BT, BD] tile: rows are gathered kv rows, d contiguous
    v = tl.load(
        kv
        + i_b * stride_kvb
        + i_g * stride_kvg
        + ids_safe[:, None] * stride_kvn
        + offs_d[None, :]
    )
    tl.store(
        gkv_td + (i0 * VGC + i_g) * (TP * DT) + offs_t[:, None] * DT + offs_d[None, :],
        v,
    )


@triton.jit
def _spmla_qk(
    q,
    gkv_dt,
    logits,  # [B*SQ*AH, TP] fp32
    stride_qm,
    stride_qh,
    stride_qd,
    G,
    APP,  # AH, padded number of heads
    VGC,
    DT: tl.constexpr,
    TP: tl.constexpr,
    DP: tl.constexpr,
    TD: tl.constexpr,
    BH: tl.constexpr,
    BT: tl.constexpr,
):
    i0 = tl.program_id(0).to(tl.int64)  # B*SQ
    i_g = tl.program_id(1)
    i_z = tl.program_id(2)  # NH*NT
    i_t = i_z % (TP // BT)
    i_bh = i_z // (TP // BT)

    offs_h = i_g * G + i_bh * BH + tl.arange(0, BH)
    offs_t = i_t * BT + tl.arange(0, BT)
    offs_d = tl.arange(0, DP)
    h_mask = offs_h < (i_g + 1) * G

    qb = tl.load(
        q + i0 * stride_qm + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd,
        h_mask[:, None],
        other=0.0,
    )
    kb = tl.load(
        gkv_dt
        + (i0 * VGC + i_g) * (DT * TP)
        + offs_d[:, None] * TP
        + offs_t[None, :]
    )
    acc = tl.dot(qb, kb, out_dtype=tl.float32)
    if TD > 0:
        offs_td = DP + tl.arange(0, TD)
        qt = tl.load(
            q
            + i0 * stride_qm
            + offs_h[:, None] * stride_qh
            + offs_td[None, :] * stride_qd,
            h_mask[:, None],
            other=0.0,
        )
        kt = tl.load(
            gkv_dt
            + (i0 * VGC + i_g) * (DT * TP)
            + offs_td[:, None] * TP
            + offs_t[None, :]
        )
        acc = tl.dot(qt, kt, acc, out_dtype=tl.float32)

    tl.store(logits + (i0 * APP + offs_h[:, None]) * TP + offs_t[None, :], acc)


@triton.jit
def _spmla_softmax(
    indices,
    logits,
    probs,  # [B*SQ*AH, TP] bf16
    lse,  # [B*SQ*H] bf16
    sm_scale,
    SKV,
    SQC,
    VGC,
    G,
    HEAD_TOTAL,
    TOPK,
    APP,  # AH
    TP: tl.constexpr,
    BT: tl.constexpr,
):
    # One program per (b, sq, head).  Flat 1D tiles only: mixing a [BH]
    # accumulator with a 2D ``tl.max(axis=1)`` makes TritonXPUCoreTiling reject
    # the module.
    i0 = tl.program_id(0).to(tl.int64)  # B*SQ
    i_h = tl.program_id(1)
    i_sq = i0 % SQC
    i_g = i_h // G

    offs_t = tl.arange(0, BT)
    idx_base = indices + i0 * (VGC * TOPK) + i_g * TOPK
    lg_base = logits + (i0 * APP + i_h) * TP
    p_base = probs + (i0 * APP + i_h) * TP

    # pass 1: max over topk, folded per BT tile (never one wide reduction tile)
    run_max = float("-inf")
    for it in range(TP // BT):
        id_off = tl.minimum(it * BT + offs_t, TOPK - 1)
        ids = tl.load(idx_base + id_off)
        m = (it * BT + offs_t < TOPK) & (ids >= 0) & (ids < SKV) & (ids <= i_sq)
        x = tl.load(lg_base + it * BT + offs_t) * sm_scale
        x = tl.where(m, x, float("-inf"))
        run_max = tl.maximum(run_max, tl.max(x))

    has_valid = run_max != float("-inf")
    safe_max = tl.where(has_valid, run_max, 0.0)

    # pass 2: sum of exp
    run_sum = 0.0
    for it in range(TP // BT):
        id_off = tl.minimum(it * BT + offs_t, TOPK - 1)
        ids = tl.load(idx_base + id_off)
        m = (it * BT + offs_t < TOPK) & (ids >= 0) & (ids < SKV) & (ids <= i_sq)
        x = tl.load(lg_base + it * BT + offs_t) * sm_scale
        x = tl.where(m, x, float("-inf"))
        run_sum += tl.sum(tl.math.exp(x - safe_max))

    lse_val = safe_max + tl.math.log(run_sum)
    tl.store(lse + i0 * HEAD_TOTAL + i_h, tl.where(has_valid, lse_val, float("-inf")))

    # pass 3: probabilities, already normalized by lse_val
    lse_p = tl.where(has_valid, lse_val, 0.0)
    for it in range(TP // BT):
        id_off = tl.minimum(it * BT + offs_t, TOPK - 1)
        ids = tl.load(idx_base + id_off)
        m = (it * BT + offs_t < TOPK) & (ids >= 0) & (ids < SKV) & (ids <= i_sq)
        x = tl.load(lg_base + it * BT + offs_t) * sm_scale
        x = tl.where(m, x, float("-inf"))
        p = tl.math.exp(x - lse_p)
        p = tl.where(m, p, 0.0)
        tl.store(p_base + it * BT + offs_t, p.to(tl.bfloat16))


@triton.jit
def _spmla_pv(
    probs,  # [B*SQ*AH, TP] bf16
    gkv_td,  # [B*SQ*VG, TP, DT] bf16
    out,  # [B*SQ*AH, DV] bf16
    G,
    APP,  # AH
    VGC,
    DT: tl.constexpr,
    TP: tl.constexpr,
    DV: tl.constexpr,
    BH: tl.constexpr,
    BT: tl.constexpr,
    BDV: tl.constexpr,
):
    i0 = tl.program_id(0).to(tl.int64)  # B*SQ
    i_g = tl.program_id(1)
    i_z = tl.program_id(2)  # NH*NDV
    i_v = i_z % (DV // BDV)
    i_bh = i_z // (DV // BDV)

    offs_h = i_g * G + i_bh * BH + tl.arange(0, BH)
    offs_t = tl.arange(0, BT)
    offs_v = i_v * BDV + tl.arange(0, BDV)

    p_base = probs + (i0 * APP + offs_h[:, None]) * TP
    v_base = gkv_td + (i0 * VGC + i_g) * (TP * DT)

    acc = tl.zeros([BH, BDV], dtype=tl.float32)
    for it in range(TP // BT):
        pb = tl.load(p_base + it * BT + offs_t[None, :])
        vb = tl.load(v_base + (it * BT + offs_t)[:, None] * DT + offs_v[None, :])
        acc = tl.dot(pb, vb, acc, out_dtype=tl.float32)

    tl.store(out + (i0 * APP + offs_h[:, None]) * DV + offs_v[None, :], acc.to(tl.bfloat16))


def triton_sparse_mla_fwd_interface(
    q, kv, indices, sm_scale=None, return_p_sum: bool = False, d_v=512
):
    logger.debug("GEMS SPARSE_MLA_FWD_INTERFACE (kunlunxin)")
    assert return_p_sum is False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    B, SQ, H, DT = q.shape
    _, SKV, VG, _ = kv.shape
    D = d_v
    assert D <= DT
    assert kv.shape[-1] == DT
    TD = DT - D
    _, _, _, K = indices.shape
    assert indices.shape == (B, SQ, VG, K)
    assert H % VG == 0
    assert K > 0 and H > 0 and B > 0 and SQ > 0
    G = H // VG
    if sm_scale is None:
        sm_scale = DT**-0.5
    sm_scale = float(sm_scale)

    BH = max(16, min(64, triton.next_power_of_2(G)))
    NH = triton.cdiv(G, BH)
    AH = triton.cdiv(H, BH) * BH
    BT = _BT
    BD = _BD
    BDV = _BDV
    TP = triton.cdiv(K, BT) * BT
    ND = triton.cdiv(DT, BD)
    NDV = triton.cdiv(D, BDV)
    SQC = B * SQ

    lse = torch.full(
        (B, SQ, H), float("-inf"), device=q.device, dtype=torch.bfloat16
    )

    gkv_dt = torch.empty((B, SQ, VG, DT, TP), device=q.device, dtype=q.dtype)
    gkv_td = torch.empty((B, SQ, VG, TP, DT), device=q.device, dtype=q.dtype)
    logits = torch.empty((B, SQ, AH, TP), device=q.device, dtype=torch.float32)
    probs = torch.empty((B, SQ, AH, TP), device=q.device, dtype=q.dtype)
    padded_out = torch.empty((B, SQ, AH, D), device=q.device, dtype=q.dtype)

    grid_gather = (SQC, VG, (TP // BT) * ND)
    _spmla_gather_dt[grid_gather](
        kv,
        indices,
        gkv_dt,
        kv.stride(0),
        kv.stride(2),
        kv.stride(1),
        SKV,
        SQC,
        VG,
        K,
        DT,
        TP,
        BT,
        BD,
        ND,
    )
    _spmla_gather_td[grid_gather](
        kv,
        indices,
        gkv_td,
        kv.stride(0),
        kv.stride(2),
        kv.stride(1),
        SKV,
        SQC,
        VG,
        K,
        DT,
        TP,
        BT,
        BD,
        ND,
    )
    _spmla_qk[(SQC, VG, NH * (TP // BT))](
        q,
        gkv_dt,
        logits,
        q.stride(1),
        q.stride(2),
        q.stride(3),
        G,
        AH,
        VG,
        DT,
        TP,
        D,
        TD,
        BH,
        BT,
    )
    _spmla_softmax[(SQC, H)](
        indices,
        logits,
        probs,
        lse,
        sm_scale,
        SKV,
        SQC,
        VG,
        G,
        H,
        K,
        AH,
        TP,
        BT,
    )
    _spmla_pv[(SQC, VG, NH * NDV)](
        probs,
        gkv_td,
        padded_out,
        G,
        AH,
        VG,
        DT,
        TP,
        D,
        BH,
        BT,
        BDV,
    )

    out = padded_out[:, :, :H, :]
    if not out.is_contiguous():
        out = out.contiguous()
    return out, lse