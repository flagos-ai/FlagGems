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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

LOG2E = 1.4426950408889634


@triton.jit
def _rha(x):
    """Round an fp32 value to the nearest bf16 grid point, ties away from zero.

    Matches CANN RoundMode::CAST_ROUND (round half away from zero), which the
    native op uses for the exp cast and for the final attention-out cast. The
    result is exactly representable in bfloat16, so a subsequent cast is exact.
    """
    u = ((x.to(tl.int32, bitcast=True) + 32768) >> 16) << 16
    return u.to(tl.float32, bitcast=True)


@triton.jit
def _dequant_kernel(
    kvf_ptr,
    kvb_ptr,
    bt_ptr,
    dense_ptr,
    seq_ptr,
    NROWS,
    table_width,
    pages,
    ROWF: tl.constexpr,
    ROWB: tl.constexpr,
    D: tl.constexpr,
    DR: tl.constexpr,
    BLK: tl.constexpr,
):
    NOPE: tl.constexpr = D - DR
    SCOFF: tl.constexpr = DR + (NOPE >> 1)
    i = tl.program_id(0) * BLK + tl.arange(0, BLK)
    pm = i < NROWS
    seq = tl.load(seq_ptr)
    p = tl.maximum(seq - NROWS, 0) + i
    valid = pm & (p >= 0) & (p < seq) & ((p >> 5) < table_width)
    page = tl.load(bt_ptr + (p >> 5), mask=valid, other=-1)
    valid = valid & (page >= 0) & (page < pages)
    row = page.to(tl.int64) * 32 + (p & 31)
    d = tl.arange(0, D)
    dg = d >> 6
    nmask = (d < NOPE)[None, :]
    nf = tl.load(
        kvf_ptr + row[:, None] * ROWF + (DR * 2 + d)[None, :],
        mask=valid[:, None] & nmask,
        other=0.0,
    )
    scl = tl.load(
        kvb_ptr + row[:, None] * ROWB + (SCOFF + dg)[None, :],
        mask=valid[:, None] & (dg < 7)[None, :],
        other=0.0,
    )
    kd = nf.to(tl.bfloat16) * scl
    dr = tl.arange(0, DR)
    rope = tl.load(
        kvb_ptr + row[:, None] * ROWB + dr[None, :], mask=valid[:, None], other=0.0
    )
    tl.store(dense_ptr + i[:, None] * D + d[None, :], kd, mask=pm[:, None] & nmask)
    tl.store(dense_ptr + i[:, None] * D + (NOPE + dr)[None, :], rope, mask=pm[:, None])


@triton.jit
def _dequant_idx_kernel(
    kvf_ptr,
    kvb_ptr,
    bt_ptr,
    idx_ptr,
    dense_ptr,
    NIDX,
    ROFF,
    seq_ptr,
    table_width,
    pages,
    ROWF: tl.constexpr,
    ROWB: tl.constexpr,
    D: tl.constexpr,
    DR: tl.constexpr,
    BLK: tl.constexpr,
):
    NOPE: tl.constexpr = D - DR
    SCOFF: tl.constexpr = DR + (NOPE >> 1)
    i = tl.program_id(0) * BLK + tl.arange(0, BLK)
    pm = i < NIDX
    c = tl.load(idx_ptr + i, mask=pm, other=0)
    valid = pm & (c >= 0) & (c < tl.load(seq_ptr)) & ((c >> 5) < table_width)
    page = tl.load(bt_ptr + (c >> 5), mask=valid, other=-1)
    valid = valid & (page >= 0) & (page < pages)
    row = page.to(tl.int64) * 32 + (c & 31)
    d = tl.arange(0, D)
    dg = d >> 6
    nmask = (d < NOPE)[None, :]
    nf = tl.load(
        kvf_ptr + row[:, None] * ROWF + (DR * 2 + d)[None, :],
        mask=valid[:, None] & nmask,
        other=0.0,
    )
    scl = tl.load(
        kvb_ptr + row[:, None] * ROWB + (SCOFF + dg)[None, :],
        mask=valid[:, None] & (dg < 7)[None, :],
        other=0.0,
    )
    kd = nf.to(tl.bfloat16) * scl
    dr = tl.arange(0, DR)
    rope = tl.load(
        kvb_ptr + row[:, None] * ROWB + dr[None, :], mask=valid[:, None], other=0.0
    )
    o = ROFF + i
    tl.store(dense_ptr + o[:, None] * D + d[None, :], kd, mask=pm[:, None] & nmask)
    tl.store(dense_ptr + o[:, None] * D + (NOPE + dr)[None, :], rope, mask=pm[:, None])


@triton.jit
def _doc_kernel(
    q_ptr,
    out_ptr,
    uni_ptr,
    cidx_ptr,
    seq_ori_ptr,
    seq_cmp_ptr,
    sinks_ptr,
    H,
    T,
    CMP_TOPK,
    qk_scale,
    NWIN,
    D: tl.constexpr,
    DR: tl.constexpr,
    BLOCK_H: tl.constexpr,
    CH: tl.constexpr,
    FULL_H: tl.constexpr,
):
    NEG: tl.constexpr = float("-inf")
    hb = tl.program_id(0)
    t = tl.program_id(1)
    hs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    d = tl.arange(0, D)
    if FULL_H:
        hmask = hs < H
    else:
        hmask = hs >= 0
    qbase = q_ptr + t * (H * D)
    qn = tl.load(qbase + hs[:, None] * D + d[None, :], mask=hmask[:, None], other=0.0)
    sink = tl.load(sinks_ptr + hs, mask=hmask, other=0.0).to(tl.float32)

    seq_o = tl.load(seq_ori_ptr)
    seq_c = tl.load(seq_cmp_ptr)
    P0 = tl.maximum(seq_o - NWIN, 0)
    hi = seq_o - 1
    nblk = tl.cdiv(NWIN + CMP_TOPK, CH)
    cbase = cidx_ptr + t * CMP_TOPK

    m = sink
    denominator = tl.full((BLOCK_H,), 1.0, tl.float32)
    acc = tl.zeros((BLOCK_H, D), tl.float32)
    for j in range(0, nblk):
        r = j * CH + tl.arange(0, CH)
        kd = tl.load(uni_ptr + r[:, None] * D + d[None, :])
        s = tl.dot(qn, tl.trans(kd)) * qk_scale
        win = r < NWIN
        ci = tl.load(cbase + (r - NWIN), mask=r >= NWIN, other=0)
        ok = tl.where(win, (P0 + r) <= hi, (ci >= 0) & (ci < seq_c))
        s = tl.where(ok[None, :], s, NEG)

        m_new = tl.maximum(m, tl.max(s, 1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        denominator = denominator * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kd)
        m = m_new

    inv = 1.0 / denominator
    tl.store(
        out_ptr + t * (H * D) + hs[:, None] * D + d[None, :],
        (acc * inv[:, None]).to(tl.bfloat16),
        mask=hmask[:, None],
    )


@triton.jit
def _ori64_kernel(
    q_ptr,
    out_ptr,
    dense_ptr,
    seq_ori_ptr,
    sinks_ptr,
    H,
    T,
    qk_scale2,
    ntok,
    D: tl.constexpr,
    DR: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINL: tl.constexpr,
    WINR: tl.constexpr,
    FULL_H: tl.constexpr,
):
    HALF: tl.constexpr = D // 2
    NEG: tl.constexpr = float("-inf")
    hb = tl.program_id(0)
    tb = tl.program_id(1) * ntok
    hs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    d = tl.arange(0, HALF)
    if FULL_H:
        hmask = hs < H
    else:
        hmask = hs >= 0
    sink2 = tl.load(sinks_ptr + hs, mask=hmask, other=0.0).to(tl.float32)
    seq_o = tl.load(seq_ori_ptr)
    P0 = tl.maximum(seq_o - T - WINL, 0)

    for ti in range(0, ntok):
        t = tb + ti
        qbase = q_ptr + t * (H * D)
        obase = out_ptr + t * (H * D)
        m2 = sink2
        denominator = tl.full((BLOCK_H,), 1.0, tl.float32)

        pos = seq_o - T + t
        if WINL < 0:
            lo = 0 * pos
        else:
            lo = pos - WINL
        lo = tl.maximum(lo, 0)
        hi = tl.minimum(pos + WINR, seq_o - 1)
        nb = tl.maximum(tl.cdiv(hi - lo + 1, BLOCK_N), 0)

        for j in range(0, nb):
            p_ = lo + j * BLOCK_N + tl.arange(0, BLOCK_N)
            pm = p_ <= hi
            rs = (p_ - P0)[:, None] * D

            q0 = tl.load(
                qbase + hs[:, None] * D + d[None, :], mask=hmask[:, None], other=0.0
            )
            k0 = tl.load(dense_ptr + rs + d[None, :], mask=pm[:, None], other=0.0)
            s = tl.dot(q0, tl.trans(k0))
            q1 = tl.load(
                qbase + hs[:, None] * D + (HALF + d)[None, :],
                mask=hmask[:, None],
                other=0.0,
            )
            k1 = tl.load(
                dense_ptr + rs + (HALF + d)[None, :], mask=pm[:, None], other=0.0
            )
            s = (s + tl.dot(q1, tl.trans(k1))) * qk_scale2
            s = tl.where(pm[None, :], s, NEG)

            m_new = tl.maximum(m2, tl.max(s, 1))
            alpha = tl.exp(m2 - m_new)
            p = tl.exp(s - m_new[:, None])
            denominator = denominator * alpha + tl.sum(p, 1)
            pb = _rha(p).to(tl.bfloat16)

            inv = 1.0 / denominator
            o0 = _rha(tl.dot(pb, k0) * inv[:, None])
            tl.store(
                obase + hs[:, None] * D + d[None, :],
                o0.to(tl.bfloat16),
                mask=hmask[:, None],
            )
            o1 = _rha(tl.dot(pb, k1) * inv[:, None])
            tl.store(
                obase + hs[:, None] * D + (HALF + d)[None, :],
                o1.to(tl.bfloat16),
                mask=hmask[:, None],
            )
            m2 = m_new


def _launch(
    q,
    ori_kv,
    ori_block_table,
    seqused_ori_kv,
    sinks,
    softmax_scale,
    cmp_kv,
    cmp_block_table,
    cmp_sparse_indices,
    seqused_cmp_kv,
):
    # Each query sees up to 128 original KV rows. Allocate just the union
    # of reachable rows; lengths and page IDs stay on the device.
    tokens, heads, width = q.shape
    scale = width**-0.5 if softmax_scale is None else float(softmax_scale)
    out = torch.empty_like(q)
    lse = torch.empty(0, dtype=torch.float32, device=q.device)
    original_rows = tokens + 127
    compressed_rows = cmp_sparse_indices.numel() if cmp_kv is not None else 0
    dense = torch.empty(
        (original_rows + compressed_rows, width), dtype=q.dtype, device=q.device
    )
    _dequant_kernel[(triton.cdiv(original_rows, 16),)](
        ori_kv,
        ori_kv.view(torch.bfloat16),
        ori_block_table,
        dense,
        seqused_ori_kv,
        original_rows,
        ori_block_table.shape[1],
        ori_kv.shape[0],
        608,
        304,
        512,
        64,
        16,
        num_warps=1,
        num_stages=1,
    )
    if compressed_rows:
        _dequant_idx_kernel[(triton.cdiv(compressed_rows, 16),)](
            cmp_kv,
            cmp_kv.view(torch.bfloat16),
            cmp_block_table,
            cmp_sparse_indices,
            dense,
            compressed_rows,
            original_rows,
            seqused_cmp_kv,
            cmp_block_table.shape[1],
            cmp_kv.shape[0],
            608,
            304,
            512,
            64,
            16,
            num_warps=1,
            num_stages=1,
        )
        _doc_kernel[(triton.cdiv(heads, 16), tokens)](
            q,
            out,
            dense,
            cmp_sparse_indices,
            seqused_ori_kv,
            seqused_cmp_kv,
            sinks,
            heads,
            tokens,
            compressed_rows,
            scale,
            original_rows,
            512,
            64,
            16,
            128,
            True,
            num_warps=1,
            num_stages=1,
        )
    else:
        _ori64_kernel[(triton.cdiv(heads, 32), tokens)](
            q,
            out,
            dense,
            seqused_ori_kv,
            sinks,
            heads,
            tokens,
            scale,
            1,
            512,
            64,
            32,
            128,
            127,
            0,
            True,
            num_warps=1,
            num_stages=1,
        )
    return out, lse


def mixed_quant_sparse_flash_mla(
    q,
    quant_mode,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    rope_head_dim=None,
    softmax_scale=None,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="TND",
    layout_kv="PA_BBND",
    topk_value_mode=1,
    return_softmax_lse=False,
    key_dtype=None,
    value_dtype=None,
):
    """Single-sequence TND mixed-quant sparse MLA prefill and compressed decode."""
    if (
        q.ndim != 3
        or q.shape[1:] != (64, 512)
        or q.dtype != torch.bfloat16
        or quant_mode != 1
        or layout_q != "TND"
        or layout_kv != "PA_BBND"
        or rope_head_dim != 64
        or ori_mask_mode != 4
        or ori_win_left != 127
        or ori_win_right != 0
        or return_softmax_lse
        or topk_value_mode != 1
        or ori_sparse_indices is not None
        or any(
            a is not None
            for a in (
                ori_topk_length,
                cmp_topk_length,
                key_dtype,
                value_dtype,
                cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv,
                seqused_q,
            )
        )
    ):
        raise ValueError(
            "expected a single sequence with a causal 128-token window and 64 rotary channels"
        )
    if (
        ori_kv is None
        or ori_kv.ndim != 4
        or ori_kv.shape[1:] != (32, 1, 608)
        or ori_kv.dtype != torch.float8_e4m3fn
        or ori_block_table is None
        or ori_block_table.ndim != 2
        or ori_block_table.shape[0] != 1
        or ori_block_table.dtype != torch.int32
        or seqused_ori_kv is None
        or seqused_ori_kv.shape != (1,)
        or seqused_ori_kv.dtype != torch.int32
        or sinks is None
        or sinks.shape != (64,)
        or sinks.dtype != torch.float32
        or cu_seqlens_q is None
        or cu_seqlens_q.shape != (2,)
    ):
        raise ValueError(
            "invalid packed original cache, sequence lengths, block table or per-head sinks"
        )
    operands = [q, ori_kv, ori_block_table, seqused_ori_kv, sinks, cu_seqlens_q]
    if cmp_kv is not None:
        if (
            q.shape[0] != 1
            or cmp_ratio != 4
            or cmp_mask_mode != 3
            or cmp_kv.ndim != 4
            or cmp_kv.shape[1:] != (32, 1, 608)
            or cmp_kv.dtype != ori_kv.dtype
            or cmp_block_table is None
            or cmp_block_table.ndim != 2
            or cmp_block_table.shape[0] != 1
            or cmp_block_table.dtype != torch.int32
            or cmp_sparse_indices is None
            or cmp_sparse_indices.shape != (1, 1, 512)
            or cmp_sparse_indices.dtype != torch.int32
            or seqused_cmp_kv is None
            or seqused_cmp_kv.shape != (1,)
            or seqused_cmp_kv.dtype != torch.int32
        ):
            raise ValueError(
                "compressed path requires single-token decode and 512 selected entries"
            )
        operands.extend((cmp_kv, cmp_block_table, cmp_sparse_indices, seqused_cmp_kv))
    elif (
        any(
            a is not None
            for a in (
                cmp_block_table,
                cmp_sparse_indices,
                seqused_cmp_kv,
                cmp_residual_kv,
            )
        )
        or cmp_ratio != 1
        or cmp_mask_mode != 0
    ):
        raise ValueError("incomplete compressed-cache arguments")
    if (
        cu_seqlens_q.dtype != torch.int32
        or not q.shape[0]
        or q.shape[0] > ori_block_table.shape[1] * 32
        or not ori_kv.shape[0]
    ):
        raise ValueError("invalid query lengths or original page table capacity")
    if cmp_kv is not None and (cmp_block_table.shape[1] == 0 or cmp_kv.shape[0] == 0):
        raise ValueError("compressed page table and storage must be nonempty")
    if cmp_residual_kv is not None:
        if cmp_residual_kv.dtype != torch.int32 or cmp_residual_kv.shape != (1,):
            raise ValueError("expected a single compressed residual length")
        operands.append(cmp_residual_kv)
    if metadata is not None:
        if metadata.dtype != torch.int32 or metadata.ndim != 1:
            raise ValueError("invalid native tiling metadata")
        operands.append(metadata)
    if any(not t.is_contiguous() or t.device != q.device for t in operands):
        raise ValueError(
            "all cache and metadata operands must be contiguous on the query device"
        )
    if q.device.type != "npu":
        raise ValueError("expected inputs on the same NPU")
    if q.numel() >= 2**31:
        raise ValueError("query must be addressable with int32 indices")
    if not q.shape[0] <= int(seqused_ori_kv[0]) <= ori_block_table.shape[1] * 32:
        raise ValueError("original sequence length is outside the page table capacity")
    if int(cu_seqlens_q[0]) != 0 or int(cu_seqlens_q[1]) != q.shape[0]:
        raise ValueError("expected the single-sequence query prefix [0, tokens]")
    if (
        cmp_kv is not None
        and not 0 <= int(seqused_cmp_kv[0]) <= cmp_block_table.shape[1] * 32
    ):
        raise ValueError("compressed length is outside the page table capacity")
    with torch_device_fn.device(q.device):
        return _launch(
            q,
            ori_kv,
            ori_block_table,
            seqused_ori_kv,
            sinks,
            softmax_scale,
            cmp_kv,
            cmp_block_table,
            cmp_sparse_indices,
            seqused_cmp_kv,
        )
