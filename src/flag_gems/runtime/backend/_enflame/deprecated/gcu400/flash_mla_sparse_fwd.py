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
"""Sparse MLA prefill for GCU400.

Two-kernel design (replaces the single fused gather kernel and the upstream
TLE warp-specialize path, which aborts at compile time on GCU - see
compiler_problems/P02, instance flaggems_enflame_flash_mla_sparse_fwd_tle):

1. ``_flash_mla_sparse_gather_kernel`` materializes the gathered KV rows into a
   compact buffer. On GCU a data-dependent gather load is emitted element by
   element (~ns per element, rate limited by per-program emission), so the
   single-kernel attention that gathers inside the dot pipeline runs at a
   structural floor (measured 4.2 s for s_q=4096/topk=2048/h_q=128/d_qk=576).
   Viewing the KV rows as int64 quarters the emission count per byte and a
   tiny per-program tile (one [RPP, 128] i64 tile + narrow tail tile) keeps
   many programs resident per SIP, reaching ~0.8-1.3 TB/s effective r+w.
2. ``_flash_mla_sparse_attn_kernel`` runs the online-softmax attention over the
   compacted rows with fully affine loads (BH=64/BK=256/BDP=256 measured best;
   wider N tiles reach the ~123 TFLOPS bf16 dot ceiling of the device).

Semantics (invalid indices, topk_length, attn_sink incl. +-inf, all-invalid
rows -> max_logits=-inf / lse=+inf / output=0) match the upstream op exactly.
"""

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# K1 launch ladder keyed by gathered-row count (measured on GCU400/L600:
# rpp=128/nw=2 wins for >=3M rows, rpp=64/nw=1 below; nw>=4 collapses the
# elementwise gather emission rate by >5x and must not be used).
_K1_LARGE_NROW = 3 * 1024 * 1024
_K1_RPP_LARGE = 128
_K1_NW_LARGE = 2
_K1_RPP_SMALL = 64
_K1_NW_SMALL = 1

# K2 tile (BH, BK, BDP, num_warps): best of the measured sweep
# (tile reuse across the QK and PV dots halves the g-buffer reads; nw=2 beats
# nw=4 with the persisting tiles, nw=1 fails to link with R_DTU_ADDR16_LO_ICALL).
_K2_BH = 64
_K2_BK = 256
_K2_BDP = 256
_K2_NW = 2


@triton.jit
def _flash_mla_sparse_gather_kernel(
    kv,
    idx,
    g_main,
    g_tail,
    SKV,
    NROW,
    WSRC: tl.constexpr,
    RPP: tl.constexpr,
    EVEN: tl.constexpr,
):
    # kv: [SKV, WSRC] int64 view of the bf16 KV cache (WSRC = DQK // 4).
    # g_main: [NROW, 128] i64 (first 512 bf16 of each gathered row);
    # g_tail: [NROW, WSRC - 128] i64 (bf16 tail columns, DQK == 576 only).
    # Invalid indices are clamped to row 0; the attention kernel masks them out.
    pid = tl.program_id(0)
    pidy = tl.program_id(1)
    pidg = pidy.to(tl.int64) * tl.num_programs(0) + pid
    row0 = pidg * RPP
    if row0 < NROW:
        offs_r = tl.arange(0, RPP)
        rows = row0 + offs_r
        rmask = rows < NROW
        if EVEN:
            ids = tl.load(idx + rows)
        else:
            ids = tl.load(idx + rows, rmask, other=0)
        ids = tl.where((ids >= 0) & (ids < SKV), ids, 0).to(tl.int64)
        offs_w1 = tl.arange(0, 128)
        src1 = kv + ids[:, None] * WSRC + offs_w1[None, :]
        dst1 = g_main + rows[:, None] * 128 + offs_w1[None, :]
        if EVEN:
            v1 = tl.load(src1)
            tl.store(dst1, v1)
        else:
            v1 = tl.load(src1, rmask[:, None], other=0)
            tl.store(dst1, v1, rmask[:, None])
        if WSRC > 128:
            WT: tl.constexpr = WSRC - 128
            offs_w2 = tl.arange(0, WT)
            src2 = kv + ids[:, None] * WSRC + 128 + offs_w2[None, :]
            dst2 = g_tail + rows[:, None] * WT + offs_w2[None, :]
            if EVEN:
                v2 = tl.load(src2)
                tl.store(dst2, v2)
            else:
                v2 = tl.load(src2, rmask[:, None], other=0)
                tl.store(dst2, v2, rmask[:, None])


@triton.jit
def _flash_mla_sparse_attn_kernel(
    q,
    g,
    g_tail,
    indices,
    attn_sink,
    topk_length,
    sm_scale: tl.constexpr,
    output,
    max_logits,
    lse,
    stride_qm,
    stride_tm,
    stride_om,
    stride_mm,
    stride_lm,
    stride_gm,
    stride_gtm,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    SKV,
    TOPK,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    EVEN_K: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    BDP: tl.constexpr,
):
    DP: tl.constexpr = 512
    HAVE_TAIL: tl.constexpr = DQK == 576
    pid = tl.program_id(0)
    pidy = tl.program_id(1)
    pidg = pidy * tl.num_programs(0) + pid
    nhb: tl.constexpr = HQ // BH
    i_sq = pidg // nhb
    i_hb = pidg % nhb
    i_sq64 = i_sq.to(tl.int64)
    h0 = i_hb * BH
    q_base = q + i_sq64 * stride_qm + h0 * DQK
    t_base = indices + i_sq64 * stride_tm
    g_base = g + i_sq64 * stride_gm
    gt_base = g_tail + i_sq64 * stride_gtm
    o_base = output + i_sq64 * stride_om + h0 * DP
    ml_base = max_logits + i_sq64 * stride_mm + h0
    l_base = lse + i_sq64 * stride_lm + h0
    attn_sink_ptr = attn_sink + h0 if HAVE_ATTN_SINK else 0
    topk_len_ptr = topk_length + i_sq64 if HAVE_TOPK_LENGTH else 0

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, BDP)
    offs_k = tl.arange(0, BK)
    offs_td = tl.arange(0, 64)

    # q tiles are loop-invariant; hoisting them out of the KV loop keeps the
    # inner iteration free of q traffic.
    q0 = tl.load(q_base + offs_h[:, None] * DQK + 0 * BDP + offs_d[None, :])
    q1 = tl.load(q_base + offs_h[:, None] * DQK + 1 * BDP + offs_d[None, :])
    if HAVE_TAIL:
        qt = tl.load(q_base + offs_h[:, None] * DQK + DP + offs_td[None, :])

    max_log = tl.full([BH], float("-inf"), tl.float32)
    sum_exp = tl.full([BH], 0.0, tl.float32)
    acc0 = tl.zeros([BH, BDP], tl.float32)
    acc1 = tl.zeros([BH, BDP], tl.float32)

    topk_len = tl.load(topk_len_ptr) if HAVE_TOPK_LENGTH else TOPK
    NK = tl.cdiv(topk_len, BK)
    for ck in range(NK):
        k_offs = BK * ck + offs_k
        k_in = k_offs < topk_len
        if EVEN_K:
            ids = tl.load(t_base + k_offs)
        else:
            ids = tl.load(t_base + k_offs, k_in, other=-1)
        valid = k_in & (ids >= 0) & (ids < SKV)
        pos_msk = k_offs < TOPK

        # load the K/V tiles once per chunk; they stay live across the softmax
        # and feed both the QK and the PV dots (halves the g-buffer traffic).
        if EVEN_K:
            g_t0 = tl.load(g_base + k_offs[:, None] * 512 + 0 * BDP + offs_d[None, :])
            g_t1 = tl.load(g_base + k_offs[:, None] * 512 + 1 * BDP + offs_d[None, :])
        else:
            g_t0 = tl.load(
                g_base + k_offs[:, None] * 512 + 0 * BDP + offs_d[None, :],
                pos_msk[:, None],
                other=0.0,
            )
            g_t1 = tl.load(
                g_base + k_offs[:, None] * 512 + 1 * BDP + offs_d[None, :],
                pos_msk[:, None],
                other=0.0,
            )
        qk = tl.dot(q0, tl.trans(g_t0), out_dtype=tl.float32)
        qk = tl.dot(q1, tl.trans(g_t1), qk, out_dtype=tl.float32)
        if HAVE_TAIL:
            if EVEN_K:
                gt_t = tl.load(gt_base + k_offs[:, None] * 64 + offs_td[None, :])
            else:
                gt_t = tl.load(
                    gt_base + k_offs[:, None] * 64 + offs_td[None, :],
                    pos_msk[:, None],
                    other=0.0,
                )
            qk = tl.dot(qt, tl.trans(gt_t), qk, out_dtype=tl.float32)
        qk = qk * sm_scale

        qk = tl.where(valid[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        p = tl.math.exp(qk - new_max[:, None])
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + tl.sum(p, axis=1)
        p_b = p.to(tl.bfloat16)

        acc0 = tl.dot(p_b, g_t0, acc0 * alpha[:, None], out_dtype=tl.float32)
        acc1 = tl.dot(p_b, g_t1, acc1 * alpha[:, None], out_dtype=tl.float32)
        max_log = new_max

    valid_mask = max_log != float("-inf")
    tl.store(ml_base + offs_h, max_log)
    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    tl.store(l_base + offs_h, lse_out)
    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + offs_h)
        factor = tl.math.exp(max_log) / (tl.math.exp(orig_lse) + tl.math.exp(sink))
    else:
        factor = 1.0 / sum_exp

    out0 = tl.where(valid_mask[:, None], acc0 * factor[:, None], 0.0)
    out1 = tl.where(valid_mask[:, None], acc1 * factor[:, None], 0.0)
    o_ptr = o_base + offs_h[:, None] * DP + offs_d[None, :]
    tl.store(o_ptr, out0.to(tl.bfloat16))
    tl.store(o_ptr + BDP, out1.to(tl.bfloat16))


def _grid_2d(nprog: int) -> Tuple[int, int]:
    gx = min(65535, nprog)
    return gx, triton.cdiv(nprog, gx)


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        attn_sink: optional, [h_q], float32.
            If attn_sink is provided, when computing output, output will be additionally multiplied by
            exp(lse) / (exp(lse) + exp(attn_sink)). +-inf in attn_sink will be handled normally (i.e., -inf has no
            effect, +inf will make corresponding output all zeros).
            This argument has no effect on lse and max_logits.
        topk_length: optional, [s_q], int32.
            If provided, the i-th q token will only attend to k tokens specified by indices[i, :, :topk_length[i]],
            ignoring later k/v tokens (even if provided in indices). In extremely rare cases (topk_length provided,
            there is a valid topk index between topk_length[i] ~ s_kv, and that topk index points to a k token
            containing NaN), operator output will contain NaN, so please avoid this situation.

    Returns:
        (output, max_logits, lse)
        Please refer to tests/ref.py for the precise definitions of these parameters.
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, log-sum-exp of attention scores
    """
    logger.debug("GEMS_ENFLAME FLASH_MLA_SPARSE_FWD")
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

    # check from FlashMLA
    assert HKV == 1, "h_kv is expected to be 1"
    assert HQ == 64 or HQ == 128, "Unsupported h_q"
    assert DQK == 576 or DQK == 512, "Unsupported d_qk"

    output = torch.empty((SQ, HQ, DV), device=q.device, dtype=q.dtype)
    max_logits = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
    lse = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)

    # stage 1: gather the selected KV rows into a compact buffer (int64 view
    # packs 4 bf16 per element; the tail columns of DQK=576 go to a second
    # dense buffer so the main tiles stay power-of-two wide).
    NROW = SQ * TOPK
    WSRC = DQK // 4
    kv_i64 = kv.view(torch.int64).reshape(SKV, WSRC)
    idx_flat = indices.view(-1)
    g_main = torch.empty((NROW, 128), dtype=torch.int64, device=q.device)
    if WSRC > 128:
        g_tail = torch.empty((NROW, WSRC - 128), dtype=torch.int64, device=q.device)
    else:
        g_tail = g_main
    if NROW > _K1_LARGE_NROW:
        rpp, nw1 = _K1_RPP_LARGE, _K1_NW_LARGE
    else:
        rpp, nw1 = _K1_RPP_SMALL, _K1_NW_SMALL
    _flash_mla_sparse_gather_kernel[_grid_2d(triton.cdiv(NROW, rpp))](
        kv_i64,
        idx_flat,
        g_main,
        g_tail,
        SKV,
        NROW,
        WSRC=WSRC,
        RPP=rpp,
        EVEN=NROW % rpp == 0,
        num_warps=nw1,
        num_stages=1,
    )

    # stage 2: online-softmax attention over the compacted rows.
    even_k = topk_length is None and TOPK % _K2_BK == 0
    _flash_mla_sparse_attn_kernel[_grid_2d(SQ * (HQ // _K2_BH))](
        q,
        g_main.view(torch.bfloat16),
        g_tail.view(torch.bfloat16),
        indices,
        attn_sink,
        topk_length,
        sm_scale,
        output,
        max_logits,
        lse,
        q.stride(0),
        indices.stride(0),
        output.stride(0),
        max_logits.stride(0),
        lse.stride(0),
        TOPK * 512,
        TOPK * 64,
        HQ=HQ,
        DQK=DQK,
        SKV=SKV,
        TOPK=TOPK,
        HAVE_ATTN_SINK=attn_sink is not None,
        HAVE_TOPK_LENGTH=topk_length is not None,
        EVEN_K=even_k,
        BK=_K2_BK,
        BH=_K2_BH,
        BDP=_K2_BDP,
        num_warps=_K2_NW,
        num_stages=1,
    )
    return output, max_logits, lse
