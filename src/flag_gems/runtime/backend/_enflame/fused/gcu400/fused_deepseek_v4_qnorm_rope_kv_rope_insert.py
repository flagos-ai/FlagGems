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

"""GCU400 specialized fused DeepseekV4 QNorm+RoPE+KV RoPE Insert kernel (BF16).

Fuses Q-side per-head RMSNorm + GPT-J RoPE and KV-side GPT-J RoPE + paged
cache insert into one launch. Compared with the common implementation, the
launch grid is capped by the GCU400 hardware limit (MAX_GRID_DIM) and the
kernel strides over items inside the grid, which also removes the
`out of resource: grid.x` failure on large token counts.
"""

import logging

import torch  # noqa: F401
import triton
import triton.language as tl

from flag_gems.runtime.backend._enflame.gcu400.utils.config_utils import MAX_GRID_DIM
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _fused_qkv_kernel(
    q_ptr,
    kv_ptr,
    k_cache_ptr,
    slot_mapping_ptr,
    position_ids_ptr,
    cos_sin_cache_ptr,
    stride_q_tok,
    stride_q_head,
    stride_kv_tok,
    stride_cache_block,
    stride_cache_token,
    stride_cos_sin_pos,
    eps,
    num_q_items,
    total_items,
    NUM_HEADS: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
):
    item_id = tl.program_id(0)

    while item_id < total_items:
        if item_id < num_q_items:
            # Q path: RMSNorm (no weight) + GPT-J RoPE on the last 64 dims.
            tok_idx = item_id // NUM_HEADS
            head_idx = item_id % NUM_HEADS
            base = tok_idx * stride_q_tok + head_idx * stride_q_head

            offs = tl.arange(0, 512)
            x = tl.load(q_ptr + base + offs).to(tl.float32)
            sq_sum = tl.sum(x * x, axis=0)
            rsqrt_val = tl.math.rsqrt(sq_sum / 512.0 + eps)
            x_normed = x * rsqrt_val

            pos = tl.load(position_ids_ptr + tok_idx)
            half_offs = tl.arange(0, 32)
            cos = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + half_offs)
            sin = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + 32 + half_offs)

            rope_even = 448 + half_offs * 2
            rope_odd = rope_even + 1
            x_re = tl.load(q_ptr + base + rope_even).to(tl.float32) * rsqrt_val
            x_ro = tl.load(q_ptr + base + rope_odd).to(tl.float32) * rsqrt_val

            q_out_e = x_re * cos - x_ro * sin
            q_out_o = x_re * sin + x_ro * cos

            nope_mask = offs < 448
            tl.store(q_ptr + base + offs, x_normed.to(tl.bfloat16), mask=nope_mask)
            tl.store(q_ptr + base + rope_even, q_out_e.to(tl.bfloat16))
            tl.store(q_ptr + base + rope_odd, q_out_o.to(tl.bfloat16))
        else:
            # KV path: GPT-J RoPE + bf16 paged cache insert (skip negative slots).
            kv_idx = item_id - num_q_items
            slot_id = tl.load(slot_mapping_ptr + kv_idx)

            if slot_id >= 0:
                kv_base = kv_idx * stride_kv_tok
                offs = tl.arange(0, 512)
                kv_data = tl.load(kv_ptr + kv_base + offs)

                pos = tl.load(position_ids_ptr + kv_idx)
                half_offs = tl.arange(0, 32)
                cos = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + half_offs)
                sin = tl.load(
                    cos_sin_cache_ptr + pos * stride_cos_sin_pos + 32 + half_offs
                )

                rope_even = 448 + half_offs * 2
                rope_odd = rope_even + 1
                x_e = tl.load(kv_ptr + kv_base + rope_even).to(tl.float32)
                x_o = tl.load(kv_ptr + kv_base + rope_odd).to(tl.float32)
                out_e = x_e * cos - x_o * sin
                out_o = x_e * sin + x_o * cos

                block_idx = slot_id // CACHE_BLOCK_SIZE
                pos_in_block = slot_id % CACHE_BLOCK_SIZE
                cache_off = (
                    block_idx * stride_cache_block + pos_in_block * stride_cache_token
                )

                nope_mask = offs < 448
                tl.store(k_cache_ptr + cache_off + offs, kv_data, mask=nope_mask)
                tl.store(k_cache_ptr + cache_off + rope_even, out_e.to(tl.bfloat16))
                tl.store(k_cache_ptr + cache_off + rope_odd, out_o.to(tl.bfloat16))

        item_id += tl.num_programs(0)


def fused_deepseek_v4_qnorm_rope_kv_rope_insert(
    q,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps=1e-6,
    cache_block_size=16,
):
    """Fused QNorm+RoPE (Q) and RoPE+Insert (KV), BF16, GCU400 variant."""
    logger.debug("GEMS_ENFLAME_GCU400 FUSED_DEEPSEEK_V4_QNORM_ROPE_KV_ROPE_INSERT")

    # 大 workload 才承担 int32 转换的两次设备拷贝；小 workload 保持原 dtype
    if q.shape[0] >= 512:
        if slot_mapping.dtype == torch.int64:
            slot_mapping = slot_mapping.to(torch.int32)
        if position_ids.dtype == torch.int64:
            position_ids = position_ids.to(torch.int32)

    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    num_insert = slot_mapping.shape[0]

    total_q = num_tokens * num_heads
    total = total_q + num_insert

    if total == 0:
        return

    grid = min(total, MAX_GRID_DIM)

    _fused_qkv_kernel[(grid,)](
        q,
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        k_cache.stride(0),
        k_cache.stride(1),
        cos_sin_cache.stride(0),
        eps,
        total_q,
        total,
        num_heads,
        cache_block_size,
        num_warps=2,
        num_stages=4,
    )
