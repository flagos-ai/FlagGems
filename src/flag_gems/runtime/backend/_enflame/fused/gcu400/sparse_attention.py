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


def sparse_attn_triton(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    o = torch.empty_like(q)

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
