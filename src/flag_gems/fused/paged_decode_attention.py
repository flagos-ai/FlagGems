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

"""Paged decode attention (GQA, single query row per sequence).

Meant for the decode step of paged-KV inference: ``q`` has one row per
sequence, ``k_cache``/``v_cache`` are ``[num_blocks, block_size, kv_heads,
head_dim]`` and ``block_table`` maps logical pages to physical ones.

Design (tuned on Iluvatar BI-V150, also validated on generic Triton):

* one CTA = one warp = one ``(seq, kv_head, segment)``.  With ``num_warps=1``
  the ``p`` tile never crosses a warp, so the MMA accumulator -> operand
  layout conversion needs no shared-memory round trip (``num_warps=2`` was
  16% slower, 4/8 worse still);
* ``BLOCK_N=16`` keeps the ``p`` tile inside one MMA shape (32/64 lost
  12-30%);
* base-2 online softmax: the caller's ``scale`` is folded with ``log2(e)`` so
  the loop only uses ``exp2``/``max`` (``exp`` was 10% slower);
* the main loop is mask-free; only the ragged tail block is masked (masking
  every block cost +0.25 ms/layer);
* when ``block_n == block_size`` the segment boundaries are rounded up to
  whole pages, so one loop iteration is exactly one page: ``page`` becomes a
  scalar load and ``pos`` a constant arange, which removes the per-token
  div/mod and the vector page gather from the hot loop (measured 7-10%
  faster per layer with bit-identical output);
* the segment count is a compile-time constant and the segment length is
  derived from ``seq_lens`` on the device, so the launch grid never depends on
  host-side kv length -- that is what makes the op capturable into a CUDA
  graph (the caller must pass preallocated ``seg_buffers`` in that case).

Measured on BI-V150 (B=64, kv=4096, 16 q / 2 kv heads, d=128, block=16, bf16):
0.72 ms/layer vs 1.04 ms/layer for vLLM's Triton unified attention (372 vs
258 GB/s), max|diff| 2.4e-4; at kv=16384: 2.41 vs 3.50 ms/layer.  The
remaining gap to the load-only ceiling (0.53 ms/layer) is the PV ``tl.dot``
layout conversion -- BI-V150 reports compute capability 7.1, i.e. no bf16
tensor cores, so every bf16 dot has to go through fp16 HMMA plus layout
shuffles.
"""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
DEFAULT_NUM_SEGMENTS = 32
DEFAULT_BLOCK_N = 16


@triton.jit
def _segment_kernel(
    Q, K, V, SEG_ACC, SEG_MAX, SEG_SUM, BT, SEQ_LEN,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_btb, stride_btp,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
    GQA: tl.constexpr, BLOCK_N: tl.constexpr,
    NSEG: tl.constexpr, scale,
):
    pid_g = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, GQA)
    kvlen = tl.load(SEQ_LEN + pid_s)
    seg_len = (kvlen + NSEG - 1) // NSEG
    start = pid_g * seg_len
    end = tl.minimum(start + seg_len, kvlen)

    m_i = tl.full((GQA,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((GQA,), dtype=tl.float32)
    acc = tl.zeros((GQA, D), dtype=tl.float32)

    qv = tl.load(Q + pid_s * stride_qb + (pid_h * GQA + offs_g)[:, None] * stride_qh
                 + offs_d[None, :] * stride_qd)

    n_full = (end - start) // BLOCK_N * BLOCK_N
    main_end = start + n_full

    for tok0 in range(start, main_end, BLOCK_N):
        offs_t = tok0 + tl.arange(0, BLOCK_N)
        page = tl.load(BT + pid_s * stride_btb + (offs_t // BS) * stride_btp)
        pos = offs_t % BS
        k_ptrs = (K + page[None, :] * stride_kb + pos[None, :] * stride_ks
                  + pid_h * stride_kh + offs_d[:, None] * stride_kd)
        vt_ptrs = (V + page[:, None] * stride_vb + pos[:, None] * stride_vs
                   + pid_h * stride_vh + offs_d[None, :] * stride_vd)
        s = tl.dot(qv, tl.load(k_ptrs)) * scale
        vt = tl.load(vt_ptrs)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m_i = m_new

    if main_end < end:
        offs_t = main_end + tl.arange(0, BLOCK_N)
        mask_t = offs_t < end
        page = tl.load(BT + pid_s * stride_btb + (offs_t // BS) * stride_btp,
                       mask=mask_t, other=0)
        pos = offs_t % BS
        k_ptrs = (K + page[None, :] * stride_kb + pos[None, :] * stride_ks
                  + pid_h * stride_kh + offs_d[:, None] * stride_kd)
        vt_ptrs = (V + page[:, None] * stride_vb + pos[:, None] * stride_vs
                   + pid_h * stride_vh + offs_d[None, :] * stride_vd)
        s = tl.dot(qv, tl.load(k_ptrs, mask=mask_t[None, :], other=0.0)) * scale
        s = tl.where(mask_t[None, :], s, float("-inf"))
        vt = tl.load(vt_ptrs, mask=mask_t[:, None], other=0.0)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m_i = m_new

    base = (pid_s * HKV + pid_h) * NSEG + pid_g
    tl.store(SEG_MAX + base * GQA + offs_g, m_i)
    tl.store(SEG_SUM + base * GQA + offs_g, l_i)
    tl.store(SEG_ACC + (base * GQA + offs_g)[:, None] * D + offs_d[None, :], acc)


@triton.jit
def _segment_kernel_paged(
    Q, K, V, SEG_ACC, SEG_MAX, SEG_SUM, BT, SEQ_LEN,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_btb, stride_btp,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
    GQA: tl.constexpr, NSEG: tl.constexpr, scale,
):
    """Same math as ``_segment_kernel`` with BLOCK_N == BS (one page/iter)."""
    pid_g = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, GQA)
    kvlen = tl.load(SEQ_LEN + pid_s)
    units = (kvlen + NSEG * BS - 1) // (NSEG * BS)
    start = pid_g * units * BS
    end0 = start + units * BS
    # clamp: a segment entirely past kvlen must not iterate, otherwise m stays
    # -inf and exp2(-inf - (-inf)) would turn the partial into NaN
    end = tl.minimum(end0, kvlen)

    m_i = tl.full((GQA,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((GQA,), dtype=tl.float32)
    acc = tl.zeros((GQA, D), dtype=tl.float32)

    qv = tl.load(Q + pid_s * stride_qb + (pid_h * GQA + offs_g)[:, None] * stride_qh
                 + offs_d[None, :] * stride_qd)
    pos = tl.arange(0, BS)

    for t0 in range(start, end, BS):
        page = tl.load(BT + pid_s * stride_btb + (t0 // BS) * stride_btp)
        if t0 + BS <= kvlen:
            kt = tl.load(K + page * stride_kb + pos[None, :] * stride_ks
                         + pid_h * stride_kh + offs_d[:, None] * stride_kd)
            vt = tl.load(V + page * stride_vb + pos[:, None] * stride_vs
                         + pid_h * stride_vh + offs_d[None, :] * stride_vd)
            s = tl.dot(qv, kt) * scale
        else:
            mask_t = t0 + pos < kvlen
            kt = tl.load(K + page * stride_kb + pos[None, :] * stride_ks
                         + pid_h * stride_kh + offs_d[:, None] * stride_kd,
                         mask=mask_t[None, :], other=0.0)
            vt = tl.load(V + page * stride_vb + pos[:, None] * stride_vs
                         + pid_h * stride_vh + offs_d[None, :] * stride_vd,
                         mask=mask_t[:, None], other=0.0)
            s = tl.where(mask_t[None, :], tl.dot(qv, kt) * scale,
                         float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m_i = m_new

    base = (pid_s * HKV + pid_h) * NSEG + pid_g
    tl.store(SEG_MAX + base * GQA + offs_g, m_i)
    tl.store(SEG_SUM + base * GQA + offs_g, l_i)
    tl.store(SEG_ACC + (base * GQA + offs_g)[:, None] * D + offs_d[None, :], acc)


@triton.jit
def _combine_kernel(
    SEG_ACC, SEG_MAX, SEG_SUM, OUT,
    stride_ob, stride_oh, stride_od,
    HQ: tl.constexpr, HKV: tl.constexpr, GQA: tl.constexpr, D: tl.constexpr,
    NSEG: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_g = tl.arange(0, GQA)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, NSEG)

    base = (pid_s * HKV + pid_h) * NSEG
    mx = tl.load(SEG_MAX + (base + offs_n)[None, :] * GQA + offs_g[:, None])
    sm = tl.load(SEG_SUM + (base + offs_n)[None, :] * GQA + offs_g[:, None])
    m = tl.max(mx, 1)
    w = tl.exp2(mx - m[:, None])
    denom = tl.sum(w * sm, 1)
    accs = tl.load(SEG_ACC + ((base + offs_n)[None, :, None] * GQA + offs_g[:, None, None]) * D
                   + offs_d[None, None, :])
    out = tl.sum(accs * w[:, :, None], 1) / denom[:, None]
    tl.store(OUT + pid_s * stride_ob + (pid_h * GQA + offs_g)[:, None] * stride_oh
             + offs_d[None, :] * stride_od, out.to(OUT.dtype.element_ty))


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    num_segments: int = DEFAULT_NUM_SEGMENTS,
    block_n: int = DEFAULT_BLOCK_N,
    num_warps: int = 1,
    num_stages: int = 1,
    seg_buffers=None,
):
    """Decode-step attention over a paged KV cache.

    Args:
        q: ``[num_seqs, num_q_heads, head_dim]`` (one row per sequence).
        k_cache: ``[num_blocks, block_size, num_kv_heads, head_dim]``.
        v_cache: same shape as ``k_cache``.
        out: ``[num_seqs, num_q_heads, head_dim]`` destination.
        block_table: ``[num_seqs, max_blocks]`` int32 logical -> physical page.
        seq_lens: ``[num_seqs]`` int32 number of visible KV tokens.
        scale: softmax scale (``head_dim ** -0.5``).
        num_segments: per-sequence KV split; must be a power of two.  Fixed at
            compile time so the grid is shape-static (CUDA-graph friendly).
        seg_buffers: ``(acc, max, sum)`` scratch.  Pass preallocated tensors when
            calling from inside a CUDA graph; ``None`` allocates per call.

    Returns:
        The scratch buffers, so callers can reuse them across launches.
    """
    num_seqs, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    gqa = num_q_heads // num_kv_heads
    rows = num_seqs * num_kv_heads * num_segments
    if seg_buffers is None:
        dev = q.device
        seg_buffers = (
            torch.empty((rows, gqa, head_dim), dtype=torch.float32, device=dev),
            torch.empty((rows, gqa), dtype=torch.float32, device=dev),
            torch.empty((rows, gqa), dtype=torch.float32, device=dev),
        )
    seg_acc, seg_max, seg_sum = seg_buffers
    common = dict(
        stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qd=q.stride(2),
        stride_kb=k_cache.stride(0), stride_ks=k_cache.stride(1),
        stride_kh=k_cache.stride(2), stride_kd=k_cache.stride(3),
        stride_vb=v_cache.stride(0), stride_vs=v_cache.stride(1),
        stride_vh=v_cache.stride(2), stride_vd=v_cache.stride(3),
        stride_btb=block_table.stride(0), stride_btp=block_table.stride(1),
        HQ=num_q_heads, HKV=num_kv_heads, D=head_dim, BS=block_size, GQA=gqa,
        scale=scale * LOG2E, num_warps=num_warps, num_stages=num_stages,
    )
    if block_n == block_size:
        _segment_kernel_paged[(num_segments, num_kv_heads, num_seqs)](
            q, k_cache, v_cache, seg_acc, seg_max, seg_sum, block_table,
            seq_lens, NSEG=num_segments, **common)
    else:
        _segment_kernel[(num_segments, num_kv_heads, num_seqs)](
            q, k_cache, v_cache, seg_acc, seg_max, seg_sum, block_table,
            seq_lens, BLOCK_N=block_n, NSEG=num_segments, **common)
    _combine_kernel[(num_seqs, num_kv_heads)](
        seg_acc, seg_max, seg_sum, out,
        out.stride(0), out.stride(1), out.stride(2),
        HQ=num_q_heads, HKV=num_kv_heads, GQA=gqa, D=head_dim,
        NSEG=num_segments, num_warps=4,
    )
    return seg_buffers
