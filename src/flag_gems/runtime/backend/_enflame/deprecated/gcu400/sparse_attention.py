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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Topk tile of the head-packed kernel. Larger tiles (>=128 rows with >=256 cols)
# are data-dependent row gathers that silently miscompile on GCU.
HEADBLOCK_BLOCK = 64
# Split-topk gating: the partial buffers add traffic that only pays off when the
# topk axis is long enough and the (batch, m) grid is too small to fill the device.
# Measured on GCU: b*m=16 with 7 topk blocks gains ~1.23x from 4 splits, while
# b*m=16 with 3 blocks and b*m>=32 lose 10~60% to the extra partial traffic.
SPLIT_MAX_BATCH_M = 16
SPLIT_MIN_BLOCKS = 6
MAX_SPLITS = 4


@libentry()
@triton.jit(do_not_specialize=["scale", "topk", "total_bh"])
def fused_attn_kernel(
    Q,
    KV,
    Out,
    attn_sink,
    topk_idxs,
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
    scale,
    topk,
    total_bh,
    D: tl.constexpr,
    H: tl.constexpr,
    GRID_BH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_d = tl.arange(0, D)

    for bh_idx in tl.range(pid_bh, total_bh, GRID_BH):
        pid_b = bh_idx // H
        pid_h = bh_idx - pid_b * H

        q_base = Q + pid_b * stride_qb + pid_m * stride_qm + pid_h * stride_qh
        q_vec = tl.load(q_base + offs_d * stride_qd)
        q_scaled_bf = (q_vec.to(tl.float32) * scale).to(tl.bfloat16)

        kv_base = KV + pid_b * stride_kvb
        idx_base = topk_idxs + pid_b * stride_idxb + pid_m * stride_idxm

        acc_o = tl.zeros([D], dtype=tl.float32)
        score_max = tl.full([], float("-inf"), dtype=tl.float32)
        sum_exp = tl.zeros([], dtype=tl.float32)

        offs_blk = tl.arange(0, BLOCK)
        num_blocks = (topk + BLOCK - 1) // BLOCK

        for t in range(num_blocks):
            block_start = t * BLOCK
            raw_offs = block_start + offs_blk
            idx_mask = raw_offs < topk

            idxs = tl.load(idx_base + raw_offs * stride_idxk, mask=idx_mask, other=-1)
            gather_mask = idxs != -1
            idxs_safe = tl.where(gather_mask, idxs, 0)

            kv_ptrs = (
                kv_base + idxs_safe[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
            )
            kv_block = tl.load(kv_ptrs, mask=gather_mask[:, None], other=0.0)

            scores = tl.sum(q_scaled_bf[None, :] * kv_block, axis=1).to(tl.float32)
            scores = tl.where(gather_mask, scores, float("-inf"))

            block_max = tl.max(scores)
            score_max_prev = score_max
            score_max = tl.maximum(score_max, block_max)

            correction = tl.exp(score_max_prev - score_max)
            p = tl.exp(scores - score_max)

            acc_o = acc_o * correction
            p_bf = p.to(tl.bfloat16)
            acc_o += tl.sum(p_bf[:, None] * kv_block, axis=0).to(tl.float32)
            sum_exp = sum_exp * correction + tl.sum(p)

        sink_val = tl.load(attn_sink + pid_h)
        sum_exp = sum_exp + tl.exp(sink_val - score_max)
        acc_o = acc_o / sum_exp

        o_base = Out + pid_b * stride_ob + pid_m * stride_om + pid_h * stride_oh
        tl.store(o_base + offs_d * stride_od, acc_o.to(tl.bfloat16))


@libentry()
@triton.jit(do_not_specialize=["scale", "topk"])
def fused_attn_headblock_kernel(
    Q,
    KV,
    Out,
    attn_sink,
    topk_idxs,
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
    scale,
    topk,
    M,
    D: tl.constexpr,
    H: tl.constexpr,
    HP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Version organized by (batch, m) that computes all HP heads in one pass.

    The old kernel used (m, bh) as the program granularity, so the same KV rows were loaded
    once per head; measured time was proportional to "the number of KV rows loaded" (removing
    all computation did not speed it up), so here the head dimension is flattened into the M
    dimension of the matrix multiply:
      scores = Q[HP, D] @ KV^T[D, BLOCK]   (matrix unit)
      acc   += P[HP, BLOCK] @ KV[BLOCK, D]
    Each program loads the KV tile only once, reducing row loads to 1/H, for a measured 5-10x
    speedup.
    """
    pid = tl.program_id(0)
    pid_b = pid // M
    pid_m = pid - pid_b * M

    offs_d = tl.arange(0, D)
    offs_h = tl.arange(0, HP)
    head_mask = offs_h < H

    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_m * stride_qm
        + offs_h[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q_tile = tl.load(q_ptrs, mask=head_mask[:, None], other=0.0)
    q_scaled = (q_tile.to(tl.float32) * scale).to(tl.bfloat16)

    kv_base = KV + pid_b * stride_kvb
    idx_base = topk_idxs + pid_b * stride_idxb + pid_m * stride_idxm

    acc_o = tl.zeros([HP, D], dtype=tl.float32)
    score_max = tl.full([HP], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([HP], dtype=tl.float32)

    offs_blk = tl.arange(0, BLOCK)
    num_blocks = (topk + BLOCK - 1) // BLOCK

    for t in range(num_blocks):
        block_start = t * BLOCK
        raw_offs = block_start + offs_blk
        idx_mask = raw_offs < topk

        idxs = tl.load(idx_base + raw_offs * stride_idxk, mask=idx_mask, other=-1)
        gather_mask = idxs != -1
        idxs_safe = tl.where(gather_mask, idxs, 0)

        kv_ptrs = (
            kv_base + idxs_safe[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
        )
        kv_block = tl.load(kv_ptrs, mask=gather_mask[:, None], other=0.0)

        scores = tl.dot(q_scaled, tl.trans(kv_block))
        scores = tl.where(gather_mask[None, :], scores, float("-inf"))

        block_max = tl.max(scores, axis=1)
        score_max_prev = score_max
        score_max = tl.maximum(score_max, block_max)

        correction = tl.exp(score_max_prev - score_max)
        p = tl.exp(scores - score_max[:, None])

        acc_o = acc_o * correction[:, None]
        acc_o += tl.dot(p.to(tl.bfloat16), kv_block)
        sum_exp = sum_exp * correction + tl.sum(p, axis=1)

    sink_vals = tl.load(attn_sink + offs_h, mask=head_mask, other=0.0)
    sum_exp = sum_exp + tl.exp(sink_vals - score_max)
    acc_o = acc_o / sum_exp[:, None]

    o_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_m * stride_om
        + offs_h[:, None] * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=head_mask[:, None])


@libentry()
@triton.jit(do_not_specialize=["scale", "topk", "blocks_per_split"])
def fused_attn_split_kernel(
    Q,
    KV,
    attn_sink,
    topk_idxs,
    part_acc,
    part_max,
    part_sum,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_idxb,
    stride_idxm,
    stride_idxk,
    scale,
    topk,
    M,
    blocks_per_split,
    D: tl.constexpr,
    H: tl.constexpr,
    HP: tl.constexpr,
    BLOCK: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """Split-topk pass of the head-packed path (flash-decoding style).

    The head-packed kernel launches only (batch, m) programs, which under-fills the
    device when batch * m is small: measured per-batch cost is 1.9x worse at
    batch=16 than at batch=64 while every program streams the same KV tile shape.
    Each program here owns a contiguous chunk of the topk blocks and emits an
    unnormalized partial (acc, running max, running sum) that the combine kernel
    merges with the usual log-sum-exp rescaling.
    """
    pid = tl.program_id(0)
    bm = pid // SPLITS
    split_id = pid - bm * SPLITS
    pid_b = bm // M
    pid_m = bm - pid_b * M

    offs_d = tl.arange(0, D)
    offs_h = tl.arange(0, HP)
    head_mask = offs_h < H

    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_m * stride_qm
        + offs_h[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q_tile = tl.load(q_ptrs, mask=head_mask[:, None], other=0.0)
    q_scaled = (q_tile.to(tl.float32) * scale).to(tl.bfloat16)

    kv_base = KV + pid_b * stride_kvb
    idx_base = topk_idxs + pid_b * stride_idxb + pid_m * stride_idxm

    acc_o = tl.zeros([HP, D], dtype=tl.float32)
    score_max = tl.full([HP], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([HP], dtype=tl.float32)

    offs_blk = tl.arange(0, BLOCK)
    num_blocks = (topk + BLOCK - 1) // BLOCK
    block_begin = split_id * blocks_per_split
    block_end = tl.minimum(block_begin + blocks_per_split, num_blocks)

    for t in range(block_begin, block_end):
        block_start = t * BLOCK
        raw_offs = block_start + offs_blk
        idx_mask = raw_offs < topk

        idxs = tl.load(idx_base + raw_offs * stride_idxk, mask=idx_mask, other=-1)
        gather_mask = idxs != -1
        idxs_safe = tl.where(gather_mask, idxs, 0)

        kv_ptrs = (
            kv_base + idxs_safe[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
        )
        kv_block = tl.load(kv_ptrs, mask=gather_mask[:, None], other=0.0)

        scores = tl.dot(q_scaled, tl.trans(kv_block))
        scores = tl.where(gather_mask[None, :], scores, float("-inf"))

        block_max = tl.max(scores, axis=1)
        score_max_prev = score_max
        score_max = tl.maximum(score_max, block_max)

        correction = tl.exp(score_max_prev - score_max)
        p = tl.exp(scores - score_max[:, None])

        acc_o = acc_o * correction[:, None]
        acc_o += tl.dot(p.to(tl.bfloat16), kv_block)
        sum_exp = sum_exp * correction + tl.sum(p, axis=1)

    acc_off = bm * (SPLITS * HP * D) + split_id * (HP * D)
    tl.store(
        part_acc + acc_off + offs_h[:, None] * D + offs_d[None, :],
        acc_o,
        mask=head_mask[:, None],
    )
    stat_off = bm * (SPLITS * HP) + split_id * HP
    tl.store(part_max + stat_off + offs_h, score_max, mask=head_mask)
    tl.store(part_sum + stat_off + offs_h, sum_exp, mask=head_mask)


@libentry()
@triton.jit
def fused_attn_combine_kernel(
    part_acc,
    part_max,
    part_sum,
    Out,
    attn_sink,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    M,
    D: tl.constexpr,
    H: tl.constexpr,
    HP: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """Merge split-topk partials and apply the attention sink normalization."""
    pid = tl.program_id(0)
    pid_b = pid // M
    pid_m = pid - pid_b * M

    offs_d = tl.arange(0, D)
    offs_h = tl.arange(0, HP)
    head_mask = offs_h < H

    stat_stride = SPLITS * HP
    acc_stride = SPLITS * HP * D

    score_max = tl.full([HP], float("-inf"), dtype=tl.float32)
    for s in range(SPLITS):
        m_s = tl.load(part_max + pid * stat_stride + s * HP + offs_h, mask=head_mask)
        score_max = tl.maximum(score_max, m_s)

    acc_o = tl.zeros([HP, D], dtype=tl.float32)
    sum_exp = tl.zeros([HP], dtype=tl.float32)
    for s in range(SPLITS):
        base = pid * stat_stride + s * HP
        m_s = tl.load(part_max + base + offs_h, mask=head_mask, other=0.0)
        l_s = tl.load(part_sum + base + offs_h, mask=head_mask, other=0.0)
        rescale = tl.exp(m_s - score_max)
        acc_base = part_acc + pid * acc_stride + s * HP * D
        a_s = tl.load(
            acc_base + offs_h[:, None] * D + offs_d[None, :],
            mask=head_mask[:, None],
            other=0.0,
        )
        acc_o += a_s * rescale[:, None]
        sum_exp += l_s * rescale

    sink_vals = tl.load(attn_sink + offs_h, mask=head_mask, other=0.0)
    sum_exp = sum_exp + tl.exp(sink_vals - score_max)
    acc_o = acc_o / sum_exp[:, None]

    o_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_m * stride_om
        + offs_h[:, None] * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=head_mask[:, None])


def sparse_attn_triton(q, kv, attn_sink, topk_idxs, softmax_scale):
    logger.debug("GEMS_ENFLAME SPARSE_ATTN_TRITON")
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    o = torch.empty_like(q)

    # Head-packing path: round the head count up to a power of two as the M dimension of the matmul, with M=8 as the
    # matrix-unit lower bound;
    # when HP>16 the tile is too large (DSM 917504B limit), so still fall back to the old per-head kernel.
    head_pack = triton.next_power_of_2(h)
    if 8 <= head_pack <= 16 and d >= 64 and d % 64 == 0:
        # Parallelism decision: the head-packing path always has b*m programs; when b is small (16/32),
        # measurements show the per-batch time is 1.8~1.9x that of b=64, indicating the device is not fully occupied.
        # Therefore, when b*m falls below MIN_PROGRAMS, split SPLITS ways along the topk axis for parallelism,
        # with a combine kernel doing the log-sum-exp merge; when b*m is already sufficient, keep the single-kernel
        # path.
        num_blocks = (topk + HEADBLOCK_BLOCK - 1) // HEADBLOCK_BLOCK
        batch_m = b * m
        splits = 1
        if batch_m <= SPLIT_MAX_BATCH_M and num_blocks >= SPLIT_MIN_BLOCKS:
            splits = min(num_blocks, MAX_SPLITS)
        if splits > 1:
            blocks_per_split = (num_blocks + splits - 1) // splits
            part_acc = torch.empty(
                (batch_m, splits, head_pack, d),
                dtype=torch.float32,
                device=q.device,
            )
            part_max = torch.empty(
                (batch_m, splits, head_pack), dtype=torch.float32, device=q.device
            )
            part_sum = torch.empty(
                (batch_m, splits, head_pack), dtype=torch.float32, device=q.device
            )
            with torch_device_fn.device(q.device):
                fused_attn_split_kernel[(batch_m * splits,)](
                    q,
                    kv,
                    attn_sink,
                    topk_idxs,
                    part_acc,
                    part_max,
                    part_sum,
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    q.stride(3),
                    kv.stride(0),
                    kv.stride(1),
                    kv.stride(2),
                    topk_idxs.stride(0),
                    topk_idxs.stride(1),
                    topk_idxs.stride(2),
                    softmax_scale,
                    topk,
                    m,
                    blocks_per_split,
                    D=d,
                    H=h,
                    HP=head_pack,
                    BLOCK=HEADBLOCK_BLOCK,
                    SPLITS=splits,
                    num_warps=1,
                )
                fused_attn_combine_kernel[(batch_m,)](
                    part_acc,
                    part_max,
                    part_sum,
                    o,
                    attn_sink,
                    o.stride(0),
                    o.stride(1),
                    o.stride(2),
                    o.stride(3),
                    m,
                    D=d,
                    H=h,
                    HP=head_pack,
                    SPLITS=splits,
                    num_warps=1,
                )
            return o

        grid = (b * m,)
        with torch_device_fn.device(q.device):
            fused_attn_headblock_kernel[grid](
                q,
                kv,
                o,
                attn_sink,
                topk_idxs,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                kv.stride(0),
                kv.stride(1),
                kv.stride(2),
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),
                topk_idxs.stride(0),
                topk_idxs.stride(1),
                topk_idxs.stride(2),
                softmax_scale,
                topk,
                m,
                D=d,
                H=h,
                HP=head_pack,
                BLOCK=HEADBLOCK_BLOCK,
                num_warps=1,
            )
        return o

    # BLOCK must stay <= 64: data-dependent row-gather tiles with >= 128 rows
    # and >= 256 cols miscompile silently on GCU at low num_warps (P11 family).
    BLOCK = 64
    total_bh = b * h
    GRID_BH = min(total_bh, 255)

    grid = (m, GRID_BH)

    with torch_device_fn.device(q.device):
        fused_attn_kernel[grid](
            q,
            kv,
            o,
            attn_sink,
            topk_idxs,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            kv.stride(0),
            kv.stride(1),
            kv.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            topk_idxs.stride(0),
            topk_idxs.stride(1),
            topk_idxs.stride(2),
            softmax_scale,
            topk,
            total_bh,
            D=d,
            H=h,
            GRID_BH=GRID_BH,
            BLOCK=BLOCK,
            num_warps=1,
        )
    return o
