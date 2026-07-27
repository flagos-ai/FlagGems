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

logger = logging.getLogger(__name__)


@triton.jit
def _sparse_mla_fwd_kernel(
    q,
    kv,
    indices,
    output,
    lse,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvn,
    stride_kvg,
    stride_kvd,
    stride_ib,
    stride_im,
    stride_ig,
    stride_ik,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lm,
    stride_lh,
    stride_lv,
    sm_scale: tl.constexpr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    DQ: tl.constexpr,
    DV: tl.constexpr,
    K: tl.constexpr,
    BLOCK_DQ: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    NUM_DV_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    value_block = pid % NUM_DV_BLOCKS
    head = (pid // NUM_DV_BLOCKS) % H
    query_pos = (pid // (NUM_DV_BLOCKS * H)) % SQ
    batch = pid // (NUM_DV_BLOCKS * H * SQ)
    group = head // (H // G)

    offs_dq = tl.arange(0, BLOCK_DQ)
    offs_dv = value_block * BLOCK_DV + tl.arange(0, BLOCK_DV)

    max_score = float("-inf")
    normalizer = 0.0
    accumulator = tl.zeros([BLOCK_DV], dtype=tl.float32)

    for topk_offset in range(0, K):
        index_ptr = (
            indices
            + batch * stride_ib
            + query_pos * stride_im
            + group * stride_ig
            + topk_offset * stride_ik
        )
        kv_index = tl.load(index_ptr)
        valid = (kv_index >= 0) & (kv_index < SK) & (kv_index <= query_pos)
        safe_index = tl.where(valid, kv_index, 0)

        score = 0.0
        for d_start in tl.static_range(0, DQ, BLOCK_DQ):
            dim_offsets = d_start + offs_dq
            q_ptrs = (
                q
                + batch * stride_qb
                + query_pos * stride_qm
                + head * stride_qh
                + dim_offsets * stride_qd
            )
            key_ptrs = (
                kv
                + batch * stride_kvb
                + safe_index * stride_kvn
                + group * stride_kvg
                + dim_offsets * stride_kvd
            )
            q_vals = tl.load(q_ptrs, mask=dim_offsets < DQ, other=0.0).to(
                tl.float32
            )
            key_vals = tl.load(
                key_ptrs,
                mask=valid & (dim_offsets < DQ),
                other=0.0,
            ).to(tl.float32)
            score += tl.sum(key_vals * q_vals, axis=0)
        score = tl.where(valid, score * sm_scale, float("-inf"))

        new_max = tl.where(valid, tl.maximum(max_score, score), max_score)
        old_scale = tl.where(valid, tl.exp(max_score - new_max), 1.0)
        weight = tl.where(valid, tl.exp(score - new_max), 0.0)

        value_ptrs = (
            kv
            + batch * stride_kvb
            + safe_index * stride_kvn
            + group * stride_kvg
            + offs_dv * stride_kvd
        )
        values = tl.load(
            value_ptrs,
            mask=valid & (offs_dv < DV),
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * old_scale + values * weight
        normalizer = normalizer * old_scale + weight
        max_score = new_max

    output_ptrs = (
        output
        + batch * stride_ob
        + query_pos * stride_om
        + head * stride_oh
        + offs_dv * stride_od
    )
    tl.store(output_ptrs, accumulator / normalizer, mask=offs_dv < DV)
    lse_ptr = (
        lse
        + batch * stride_lb
        + query_pos * stride_lm
        + head * stride_lh
        + value_block * stride_lv
    )
    tl.store(lse_ptr, max_score + tl.log(normalizer))


def triton_sparse_mla_fwd_interface(
    q, kv, indices, sm_scale=None, return_p_sum: bool = False, d_v=512
):
    logger.debug("GEMS SPARSE_MLA_FWD_INTERFACE")
    assert return_p_sum is False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    B, SQ, H, DQ = q.shape
    _, SK, G, kv_dim = kv.shape
    assert kv.shape[0] == B and kv_dim == DQ
    assert H % G == 0
    assert indices.shape[:3] == (B, SQ, G)
    assert d_v <= DQ

    scale = DQ**-0.5 if sm_scale is None else sm_scale
    output = torch.empty((B, SQ, H, d_v), dtype=q.dtype, device=q.device)
    block_dv = 32
    num_dv_blocks = triton.cdiv(d_v, block_dv)
    lse = torch.empty((B, SQ, H, num_dv_blocks), dtype=q.dtype, device=q.device)
    grid = (B * SQ * H * num_dv_blocks,)
    _sparse_mla_fwd_kernel[grid](
        q,
        kv,
        indices,
        output,
        lse,
        *q.stride(),
        *kv.stride(),
        *indices.stride(),
        *output.stride(),
        *lse.stride(),
        scale,
        SQ,
        SK,
        H,
        G,
        DQ,
        d_v,
        indices.shape[-1],
        BLOCK_DQ=32,
        BLOCK_DV=block_dv,
        NUM_DV_BLOCKS=num_dv_blocks,
        num_warps=1,
        isCloseVectorization=True,
        buffer_size_limit=2048,
    )
    return output, lse[..., 0]
