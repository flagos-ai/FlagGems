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

"""
MetaX sparse attention kernels.

Multi-tier dispatch for MetaX (mcTriton) backend:
  Tier 0: NSA-style V-chunked kernel (3D grid, BV=128)
  Tier 1: chain-dot OPT MMA + flashattn-fwd LLVM preset (num_stages=2, BLOCK=32)
  Tier 2: num_stages=2 + BLOCK=32 (without MACAOptions kwargs)
  Tier 3: byte-exact FlagGems _metax config (BLOCK=16, num_warps=8)
  Tier 4: universal kernel with autotune
  Tier 5: simple bf16 fallback

Source attribution: kernel structure adapted from FlagOpen/FlagGems
runtime/backend/_metax/fused/sparse_attention.py (Apache-2.0) and
fla-org/native-sparse-attention parallel.py (NSA V-chunked pattern).
"""

import os

os.environ.setdefault("TRITON_DISABLE_SWIZZLE", "1")
# MetaX (mcTriton) compiler-pass enable flags
os.environ.setdefault("TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP", "1")
os.environ.setdefault("TRITON_ENABLE_MACA_MERGE_CONVERT_LAYOUT", "1")
os.environ.setdefault("TRITON_ENABLE_SMEM_OFFSET_CACHE", "1")
os.environ.setdefault("TRITON_ENABLE_BSM_INDEX_OPT", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


# ===========================================================================
# Kernel 1: SIMPLE bf16 kernel (fallback)
# ===========================================================================
@triton.jit
def _sparse_attn_kernel_bf16(
    Q,
    KV,
    SINK,
    IDX,
    O,  # noqa: E741
    M,
    KV_LEN,
    TOPK,
    SCALE,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    q_base = pid_b * (M * H * D) + pid_m * (H * D)
    kv_base = pid_b * (KV_LEN * D)
    idx_base = pid_b * (M * TOPK) + pid_m * TOPK
    o_base = q_base

    h_idx = tl.arange(0, H)
    d_idx = tl.arange(0, D)

    q_ptrs = Q + q_base + h_idx[:, None] * D + d_idx[None, :]
    q_bf = tl.load(q_ptrs)

    sink = tl.load(SINK + h_idx)

    NEG_INF = float("-inf")
    m_i = tl.full([H], NEG_INF, tl.float32)
    l_i = tl.zeros([H], tl.float32)
    o_i = tl.zeros([H, D], tl.float32)

    for k_start in range(0, TOPK, BLOCK_TOPK):
        k_off = k_start + tl.arange(0, BLOCK_TOPK)
        k_mask = k_off < TOPK

        idx = tl.load(IDX + idx_base + k_off, mask=k_mask, other=0)

        kv_ptrs = KV + kv_base + idx[:, None] * D + d_idx[None, :]
        kv_bf = tl.load(kv_ptrs, mask=k_mask[:, None], other=0.0)

        scores = tl.dot(q_bf, tl.trans(kv_bf), out_dtype=tl.float32) * SCALE
        scores = tl.where(k_mask[None, :], scores, NEG_INF)

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        scores_exp = tl.exp(scores - m_new[:, None])
        scores_exp = tl.where(k_mask[None, :], scores_exp, 0.0)

        l_i = alpha * l_i + tl.sum(scores_exp, axis=1)
        scores_exp_bf = scores_exp.to(tl.bfloat16)
        o_i = alpha[:, None] * o_i + tl.dot(scores_exp_bf, kv_bf, out_dtype=tl.float32)
        m_i = m_new

    m_total = tl.maximum(m_i, sink)
    alpha = tl.exp(m_i - m_total)
    sink_term = tl.exp(sink - m_total)
    l_total = alpha * l_i + sink_term
    o_final = (alpha[:, None] * o_i) / l_total[:, None]

    o_ptrs = O + o_base + h_idx[:, None] * D + d_idx[None, :]
    tl.store(o_ptrs, o_final.to(tl.bfloat16))


# ===========================================================================
# Kernel 2: UNIVERSAL bf16 kernel (FlagGems-style)
# ===========================================================================
@triton.jit
def _sparse_attn_kernel_universal(
    Q,
    KV,
    O,  # noqa: E741
    ATTN_SINK,
    IDX,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_idxb,
    stride_idxm,
    stride_idxk,
    SCALE,
    TOPK,
    H_ACTUAL,
    BLOCK: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    q_base = Q + pid_b * stride_qb + pid_m * stride_qm
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H_ACTUAL
    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)

    kv_base = KV + pid_b * stride_kvb
    idx_base = IDX + pid_b * stride_idxb + pid_m * stride_idxm

    acc_o = tl.zeros([H, D], dtype=tl.float32)
    scores_max = tl.full([H], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([H], dtype=tl.float32)

    num_blocks = (TOPK + BLOCK - 1) // BLOCK
    offs_blk = tl.arange(0, BLOCK)

    for t in range(num_blocks):
        raw_offs = t * BLOCK + offs_blk
        idx_mask = raw_offs < TOPK
        idxs = tl.load(idx_base + raw_offs * stride_idxk, mask=idx_mask, other=-1)
        valid_mask = idxs != -1

        kv_ptrs = kv_base + idxs[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
        kv_block = tl.load(kv_ptrs, mask=valid_mask[:, None], other=0.0)

        acc_s = tl.dot(q_block, tl.trans(kv_block))
        acc_s = acc_s * SCALE
        mask_bias = tl.where(valid_mask, 0.0, float("-inf"))
        acc_s = acc_s + mask_bias[None, :]

        scores_max_prev = scores_max
        block_max = tl.max(acc_s, axis=1)
        scores_max = tl.maximum(scores_max, block_max)

        correction = tl.exp(scores_max_prev - scores_max)
        p = tl.exp(acc_s - scores_max[:, None])

        acc_o = acc_o * correction[:, None]
        acc_o = tl.dot(p.to(tl.bfloat16), kv_block, acc=acc_o)

        scores_sum = tl.sum(p, axis=1)
        sum_exp = sum_exp * correction + scores_sum

    sink_vals = tl.load(ATTN_SINK + offs_h, mask=h_mask, other=0.0)
    sum_exp = sum_exp + tl.exp(sink_vals - scores_max)

    acc_o = acc_o / sum_exp[:, None]

    o_base = O + pid_b * stride_ob + pid_m * stride_om
    o_ptrs = o_base + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=h_mask[:, None])


# ===========================================================================
# Kernel 3: MetaX-exact kernel (byte-identical to FlagGems _metax style)
# Uses += form NOT acc= form; num_warps=8.
# ===========================================================================
@triton.jit
def _sparse_attn_kernel_metax_exact(
    Q,
    KV,
    O,  # noqa: E741
    ATTN_SINK,
    IDX,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_idxb,
    stride_idxm,
    stride_idxk,
    SCALE,
    TOPK,
    H_ACTUAL,
    BLOCK: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    q_base = Q + pid_b * stride_qb + pid_m * stride_qm
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H_ACTUAL
    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)

    kv_base = KV + pid_b * stride_kvb
    idx_base = IDX + pid_b * stride_idxb + pid_m * stride_idxm

    acc_o = tl.zeros([H, D], dtype=tl.float32)
    scores_max = tl.full([H], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([H], dtype=tl.float32)

    num_blocks = (TOPK + BLOCK - 1) // BLOCK
    offs_blk = tl.arange(0, BLOCK)

    for t in range(num_blocks):
        raw_offs = t * BLOCK + offs_blk
        idx_mask = raw_offs < TOPK
        idxs = tl.load(idx_base + raw_offs * stride_idxk, mask=idx_mask, other=-1)
        valid_mask = idxs != -1

        kv_ptrs = kv_base + idxs[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
        kv_block = tl.load(kv_ptrs, mask=valid_mask[:, None], other=0.0)

        acc_s = tl.dot(q_block, tl.trans(kv_block))
        acc_s = acc_s * SCALE
        mask_bias = tl.where(valid_mask, 0.0, float("-inf"))
        acc_s = acc_s + mask_bias[None, :]

        scores_max_prev = scores_max
        block_max = tl.max(acc_s, axis=1)
        scores_max = tl.maximum(scores_max, block_max)

        correction = tl.exp(scores_max_prev - scores_max)
        p = tl.exp(acc_s - scores_max[:, None])

        # FlagGems-metax style: += form (NOT acc= form)
        acc_o = acc_o * correction[:, None]
        acc_o += tl.dot(p.to(tl.bfloat16), kv_block)

        scores_sum = tl.sum(p, axis=1)
        sum_exp = sum_exp * correction + scores_sum

    sink_vals = tl.load(ATTN_SINK + offs_h, mask=h_mask, other=0.0)
    sum_exp = sum_exp + tl.exp(sink_vals - scores_max)

    acc_o = acc_o / sum_exp[:, None]

    o_base = O + pid_b * stride_ob + pid_m * stride_om
    o_ptrs = o_base + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=h_mask[:, None])


# ===========================================================================
# Kernel 4: NSA-style V-chunked kernel
# 3D grid (m, b, NV) with NV = cdiv(D, BV). Each program owns acc_o[H, BV]
# only — at BV=128, 8 KB instead of 32 KB. Redundant QK per chunk is
# accepted trade-off for 4x parallelism + 4x register pressure drop.
# ===========================================================================
@triton.jit
def _sparse_attn_kernel_v_chunked(
    Q,
    KV,
    O,  # noqa: E741
    ATTN_SINK,
    IDX,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_idxb,
    stride_idxm,
    stride_idxk,
    SCALE,
    TOPK,
    H_ACTUAL,
    BLOCK: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
    H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_v = tl.program_id(2)

    q_base = Q + pid_b * stride_qb + pid_m * stride_qm
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    offs_bv = pid_v * BV + tl.arange(0, BV)
    bv_mask = offs_bv < D
    h_mask = offs_h < H_ACTUAL

    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)

    kv_base = KV + pid_b * stride_kvb
    idx_base = IDX + pid_b * stride_idxb + pid_m * stride_idxm

    acc_o = tl.zeros([H, BV], dtype=tl.float32)
    scores_max = tl.full([H], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([H], dtype=tl.float32)

    num_blocks = (TOPK + BLOCK - 1) // BLOCK
    offs_blk = tl.arange(0, BLOCK)

    for t in range(num_blocks):
        raw_offs = t * BLOCK + offs_blk
        idx_mask = raw_offs < TOPK
        idxs = tl.load(idx_base + raw_offs * stride_idxk, mask=idx_mask, other=-1)
        valid_mask = idxs != -1

        k_ptrs = kv_base + idxs[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
        k_block = tl.load(k_ptrs, mask=valid_mask[:, None], other=0.0)

        acc_s = tl.dot(q_block, tl.trans(k_block))
        acc_s = acc_s * SCALE
        mask_bias = tl.where(valid_mask, 0.0, float("-inf"))
        acc_s = acc_s + mask_bias[None, :]

        scores_max_prev = scores_max
        block_max = tl.max(acc_s, axis=1)
        scores_max = tl.maximum(scores_max, block_max)

        correction = tl.exp(scores_max_prev - scores_max)
        p = tl.exp(acc_s - scores_max[:, None])

        v_ptrs = kv_base + idxs[:, None] * stride_kvn + offs_bv[None, :] * stride_kvd
        v_block = tl.load(
            v_ptrs,
            mask=valid_mask[:, None] & bv_mask[None, :],
            other=0.0,
        )

        acc_o = acc_o * correction[:, None]
        acc_o += tl.dot(p.to(tl.bfloat16), v_block)

        scores_sum = tl.sum(p, axis=1)
        sum_exp = sum_exp * correction + scores_sum

    sink_vals = tl.load(ATTN_SINK + offs_h, mask=h_mask, other=0.0)
    sum_exp = sum_exp + tl.exp(sink_vals - scores_max)

    acc_o = acc_o / sum_exp[:, None]

    o_base = O + pid_b * stride_ob + pid_m * stride_om
    o_ptrs = o_base + offs_h[:, None] * stride_oh + offs_bv[None, :] * stride_od
    tl.store(
        o_ptrs,
        acc_o.to(tl.bfloat16),
        mask=h_mask[:, None] & bv_mask[None, :],
    )


# Autotune configs for universal kernel
_METAX_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 16}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 32}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 64}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 128}, num_warps=8, num_stages=1),
]

_sparse_attn_universal_autotuned = triton.autotune(
    configs=_METAX_AUTOTUNE_CONFIGS,
    key=["TOPK", "H_ACTUAL", "D"],
)(_sparse_attn_kernel_universal)


# ===========================================================================
# Python wrapper — multi-tier dispatch
# ===========================================================================
def sparse_attn_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    b, m, h, d = q.shape
    kv_len = kv.shape[1]
    topk = topk_idxs.shape[-1]

    q_c = q.contiguous()
    kv_c = kv.contiguous()
    sink_c = attn_sink.contiguous()
    idx_c = topk_idxs.contiguous()

    o = torch.empty(b, m, h, d, dtype=q.dtype, device=q.device)

    h_padded = max(16, triton.next_power_of_2(h))

    # Tier 0: NSA-style V-chunked kernel
    BV = 128
    try:
        NV = (d + BV - 1) // BV
        grid = (m, b, NV)
        _sparse_attn_kernel_v_chunked[grid](
            q_c,
            kv_c,
            o,
            sink_c,
            idx_c,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            q_c.stride(3),
            kv_c.stride(0),
            kv_c.stride(1),
            kv_c.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            idx_c.stride(0),
            idx_c.stride(1),
            idx_c.stride(2),
            softmax_scale,
            topk,
            h,
            BLOCK=32,
            D=d,
            BV=BV,
            H=h_padded,
            num_warps=4,
        )
        return o
    except Exception:
        pass

    # Tier 1: chain-dot OPT MMA + flashattn-fwd LLVM preset
    try:
        grid = (m, b)
        _sparse_attn_kernel_metax_exact[grid](
            q_c,
            kv_c,
            o,
            sink_c,
            idx_c,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            q_c.stride(3),
            kv_c.stride(0),
            kv_c.stride(1),
            kv_c.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            idx_c.stride(0),
            idx_c.stride(1),
            idx_c.stride(2),
            softmax_scale,
            topk,
            h,
            BLOCK=32,
            D=d,
            H=h_padded,
            num_warps=8,
            num_stages=2,
            pipeline="basic",
            scenario="flashattn-fwd",
        )
        return o
    except Exception:
        pass

    # Tier 2: num_stages=2 + BLOCK=32 (without MACAOptions kwargs)
    try:
        grid = (m, b)
        _sparse_attn_kernel_metax_exact[grid](
            q_c,
            kv_c,
            o,
            sink_c,
            idx_c,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            q_c.stride(3),
            kv_c.stride(0),
            kv_c.stride(1),
            kv_c.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            idx_c.stride(0),
            idx_c.stride(1),
            idx_c.stride(2),
            softmax_scale,
            topk,
            h,
            BLOCK=32,
            D=d,
            H=h_padded,
            num_warps=8,
            num_stages=2,
        )
        return o
    except Exception:
        pass

    # Tier 3: byte-exact FlagGems _metax config (BLOCK=16, num_warps=8)
    try:
        grid = (m, b)
        _sparse_attn_kernel_metax_exact[grid](
            q_c,
            kv_c,
            o,
            sink_c,
            idx_c,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            q_c.stride(3),
            kv_c.stride(0),
            kv_c.stride(1),
            kv_c.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            idx_c.stride(0),
            idx_c.stride(1),
            idx_c.stride(2),
            softmax_scale,
            topk,
            h,
            BLOCK=16,
            D=d,
            H=h_padded,
            num_warps=8,
        )
        return o
    except Exception:
        pass

    # Tier 4: universal kernel with autotune
    try:
        grid = (m, b)
        _sparse_attn_universal_autotuned[grid](
            q_c,
            kv_c,
            o,
            sink_c,
            idx_c,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            q_c.stride(3),
            kv_c.stride(0),
            kv_c.stride(1),
            kv_c.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            idx_c.stride(0),
            idx_c.stride(1),
            idx_c.stride(2),
            softmax_scale,
            topk,
            h,
            D=d,
            H=h_padded,
        )
        return o
    except Exception:
        pass

    # Tier 5: simple bf16 fallback
    grid = (b, m)
    _sparse_attn_kernel_bf16[grid](
        q_c,
        kv_c,
        sink_c,
        idx_c,
        o,
        m,
        kv_len,
        topk,
        softmax_scale,
        H=h,
        D=d,
        BLOCK_TOPK=16,
        num_warps=2,
    )
    return o
