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

"""GCU400 专用 RoPE 实现（persistent launch + 按规模选择访存形态）。

与原通用实现的三处差异：

1. launch 形态：grid 固定为 `min(n_tokens, 2 * NUM_SIPS)` 个 program，每个 program
   在核内用 `tl.range` 轮转处理多个 token。原实现「一个 token 一个 program」，
   batch=4/seq=512 时有 2048 个 program，仅调度开销就约 126us；persistent 化后
   同一 case 从 219us 降到 155us。

2. 访存形态（`apply_rotary_pos_emb_kernel`）：把 q/k 的 `[num_heads, head_dim]`
   摊平成一维 lane 空间 `[BLOCK_Q]`，一次 load 覆盖全部 head。seq=512 时 load/store
   次数从 18/9 降到 4/2，同一 case 的 kernel 时间 153us -> 67us。`BLOCK_Q`/`BLOCK_K`
   取 `next_power_of_2(num_heads * padded_head_dim)`（`tl.arange` 要求 2 的幂），
   多出来的 lane 由 `dim_index < HEAD_DIM` 与 `head_index < num_heads` 一起 mask 掉。

3. 双路径切换（`FUSE_HEADS_TOKEN_THRESHOLD`）：head 合并把每个 program 的索引
   预计算从 head_dim 条 lane 放大到 `BLOCK_Q` 条 lane。n_tokens 较小（每 program
   轮到的 token 少，摊不薄这份固定成本）时端到端反而慢约 7us（实测 seq=128 时
   110us -> 117us）；n_tokens 大时则大幅领先（seq=512 时 152us -> 117us）。因此
   小规模走 `apply_rotary_pos_emb_perhead_kernel`（逐 head），大规模走合并版本。
   另外合并宽度受 `FUSE_HEADS_MAX_BLOCK` 限制：head_dim=256 且 8 个 head 时 BLOCK 会到
   2048 lane，寄存器溢出后反而更慢，这类形状同样回退到逐 head。两条路径各是一个
   独立 kernel：写进同一个 kernel 的 constexpr 分支会让编译时间整体变长。
"""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Number of SIPs on gcu400; the grid cap for the persistent kernel is 2 * NUM_SIPS.
NUM_SIPS = 24

# With grid = 2*NUM_SIPS, the per-head path degrades noticeably once each program handles more than 32 tokens,
# which corresponds to n_tokens ≈ 1536; beyond that, switching to the merged-head path is more cost-effective.
FUSE_HEADS_TOKEN_THRESHOLD = 1536

# Upper limit on the single-load width of the merged-head path. Increasing BLOCK further (e.g., head_dim=256 with 8
# heads
# gives 2048 lanes) overflows the register footprint of each program and is slower than the per-head path, so above this
# limit, still use the per-head path.
FUSE_HEADS_MAX_BLOCK = 1024


@libentry()
@triton.jit(do_not_specialize=["n_tokens", "grid_dim"])
def apply_rotary_pos_emb_kernel(
    oq_ptr,
    ok_ptr,
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    oq_stride_s,
    oq_stride_h,
    oq_stride_d,
    ok_stride_s,
    ok_stride_h,
    ok_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len,
    n_tokens,
    grid_dim,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    pid = tl.program_id(0)

    # After flattening [num_heads, head_dim] into a 1D lane space, the head/dim decomposition and the rotation indices
    # and cos-sin indices
    # needed by RoPE are all token-independent, so compute them once outside the token loop.
    q_lanes = tl.arange(0, BLOCK_Q)
    q_head = q_lanes // PADDED_HEAD_DIM
    q_dim = q_lanes % PADDED_HEAD_DIM
    q_mask = (q_dim < HEAD_DIM) & (q_head < NUM_Q_HEADS)

    k_lanes = tl.arange(0, BLOCK_K)
    k_head = k_lanes // PADDED_HEAD_DIM
    k_dim = k_lanes % PADDED_HEAD_DIM
    k_mask = (k_dim < HEAD_DIM) & (k_head < NUM_K_HEADS)

    if ROTARY_INTERLEAVED:
        # Interleaved layout: even/odd lanes are conjugates of each other, with cos/sin taken from the same dim//2
        # position.
        q_sin_neg = q_dim % 2 == 0
        q_rot_dim = tl.where(q_sin_neg, q_dim + 1, q_dim - 1)
        q_sincos_dim = q_dim // 2
        k_sin_neg = k_dim % 2 == 0
        k_rot_dim = tl.where(k_sin_neg, k_dim + 1, k_dim - 1)
        k_sincos_dim = k_dim // 2
    else:
        # Front/back half layout. NOTE: deliberately avoid using `%` with a non-power-of-two modulus here — on gcu400
        # that would cause
        # the masked gather below to read out of bounds when HEAD_DIM == 96, manifesting as incorrect
        # q/k results for dim 48..63.
        half: tl.constexpr = HEAD_DIM // 2
        q_sin_neg = q_dim < half
        q_rot_dim = tl.where(q_sin_neg, q_dim + half, q_dim - half)
        q_sincos_dim = tl.where(q_sin_neg, q_dim, q_dim - half)
        k_sin_neg = k_dim < half
        k_rot_dim = tl.where(k_sin_neg, k_dim + half, k_dim - half)
        k_sincos_dim = tl.where(k_sin_neg, k_dim, k_dim - half)

    # Memory-access offsets independent of the token are likewise computed only once.
    q_cols = q_head * q_stride_h + q_dim * q_stride_d
    q_rot_cols = q_head * q_stride_h + q_rot_dim * q_stride_d
    q_out_cols = q_head * oq_stride_h + q_dim * oq_stride_d

    k_cols = k_head * k_stride_h + k_dim * k_stride_d
    k_rot_cols = k_head * k_stride_h + k_rot_dim * k_stride_d
    k_out_cols = k_head * ok_stride_h + k_dim * ok_stride_d

    # persistent launch: the program count is capped at <= 2*NUM_SIPS, with multiple tokens rotated through inside the
    # kernel.
    for s_id in tl.range(pid, n_tokens, grid_dim):
        if pos_ptr is None:
            pos_id = s_id % seq_len
        else:
            pos_id = tl.load(pos_ptr + s_id * p_stride_s)
        tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

        cos_q = tl.load(
            cos_ptr + pos_id * cos_stride_s + q_sincos_dim, mask=q_mask, other=0.0
        ).to(tl.float32)
        sin_q = tl.load(
            sin_ptr + pos_id * sin_stride_s + q_sincos_dim, mask=q_mask, other=0.0
        ).to(tl.float32)
        sin_q = tl.where(q_sin_neg, -sin_q, sin_q)

        cos_k = tl.load(
            cos_ptr + pos_id * cos_stride_s + k_sincos_dim, mask=k_mask, other=0.0
        ).to(tl.float32)
        sin_k = tl.load(
            sin_ptr + pos_id * sin_stride_s + k_sincos_dim, mask=k_mask, other=0.0
        ).to(tl.float32)
        sin_k = tl.where(k_sin_neg, -sin_k, sin_k)

        q_token = q_ptr + s_id * q_stride_s
        q = tl.load(q_token + q_cols, mask=q_mask, other=0.0)
        rotated_q = tl.load(q_token + q_rot_cols, mask=q_mask, other=0.0)
        tl.store(
            oq_ptr + s_id * oq_stride_s + q_out_cols,
            q * cos_q + rotated_q * sin_q,
            mask=q_mask,
        )

        k_token = k_ptr + s_id * k_stride_s
        k = tl.load(k_token + k_cols, mask=k_mask, other=0.0)
        rotated_k = tl.load(k_token + k_rot_cols, mask=k_mask, other=0.0)
        tl.store(
            ok_ptr + s_id * ok_stride_s + k_out_cols,
            k * cos_k + rotated_k * sin_k,
            mask=k_mask,
        )


@libentry()
@triton.jit(do_not_specialize=["n_tokens", "grid_dim"])
def apply_rotary_pos_emb_perhead_kernel(
    oq_ptr,
    ok_ptr,
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    oq_stride_s,
    oq_stride_h,
    oq_stride_d,
    ok_stride_s,
    ok_stride_h,
    ok_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len,
    n_tokens,
    grid_dim,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    """逐 head 路径：每次只覆盖一个 head 的 head_dim 条 lane，寄存器脚印更小。"""

    pid = tl.program_id(0)

    ordered_block = tl.arange(0, PADDED_HEAD_DIM)
    mask = ordered_block < HEAD_DIM
    if ROTARY_INTERLEAVED:
        even_lane = ordered_block % 2 == 0
        rotated_block = tl.where(even_lane, ordered_block + 1, ordered_block - 1)
        sin_cos_block = ordered_block // 2
    else:
        half: tl.constexpr = HEAD_DIM // 2
        is_low_half = ordered_block < half
        rotated_block = tl.where(
            is_low_half, ordered_block + half, ordered_block - half
        )
        sin_cos_block = tl.where(is_low_half, ordered_block, ordered_block - half)

    for s_id in tl.range(pid, n_tokens, grid_dim):
        if pos_ptr is None:
            pos_id = s_id % seq_len
        else:
            pos_id = tl.load(pos_ptr + s_id * p_stride_s)
        tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

        cos = tl.load(
            cos_ptr + pos_id * cos_stride_s + sin_cos_block, mask=mask, other=0.0
        ).to(tl.float32)
        sin = tl.load(
            sin_ptr + pos_id * sin_stride_s + sin_cos_block, mask=mask, other=0.0
        ).to(tl.float32)
        if ROTARY_INTERLEAVED:
            sin = tl.where(even_lane, -sin, sin)
        else:
            sin = tl.where(is_low_half, -sin, sin)

        q_token = q_ptr + s_id * q_stride_s
        oq_token = oq_ptr + s_id * oq_stride_s
        for off_h in range(0, NUM_Q_HEADS):
            ordered_cols = off_h * q_stride_h + (ordered_block * q_stride_d)
            rotated_cols = off_h * q_stride_h + (rotated_block * q_stride_d)
            output_offs = off_h * oq_stride_h + (ordered_block * oq_stride_d)

            q = tl.load(q_token + ordered_cols, mask=mask, other=0.0)
            rotated_q = tl.load(q_token + rotated_cols, mask=mask, other=0.0)
            tl.store(oq_token + output_offs, q * cos + rotated_q * sin, mask=mask)

        k_token = k_ptr + s_id * k_stride_s
        ok_token = ok_ptr + s_id * ok_stride_s
        for off_h in range(0, NUM_K_HEADS):
            ordered_cols = off_h * k_stride_h + (ordered_block * k_stride_d)
            rotated_cols = off_h * k_stride_h + (rotated_block * k_stride_d)
            output_offs = off_h * ok_stride_h + (ordered_block * ok_stride_d)

            k = tl.load(k_token + ordered_cols, mask=mask, other=0.0)
            rotated_k = tl.load(k_token + rotated_cols, mask=mask, other=0.0)
            tl.store(ok_token + output_offs, k * cos + rotated_k * sin, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_tokens", "grid_dim"])
def apply_rotary_pos_emb_inplace_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len,
    n_tokens,
    grid_dim,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    pid = tl.program_id(0)

    ordered_block = tl.arange(0, PADDED_HEAD_DIM)
    mask = ordered_block < HEAD_DIM
    if ROTARY_INTERLEAVED:
        even_lane = ordered_block % 2 == 0
        rotated_block = tl.where(even_lane, ordered_block + 1, ordered_block - 1)
        sin_cos_block = ordered_block // 2
    else:
        half: tl.constexpr = HEAD_DIM // 2
        is_low_half = ordered_block < half
        rotated_block = tl.where(
            is_low_half, ordered_block + half, ordered_block - half
        )
        sin_cos_block = tl.where(is_low_half, ordered_block, ordered_block - half)

    for s_id in tl.range(pid, n_tokens, grid_dim):
        if pos_ptr is None:
            pos_id = s_id % seq_len
        else:
            pos_id = tl.load(pos_ptr + s_id * p_stride_s)
        tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

        cos = tl.load(
            cos_ptr + pos_id * cos_stride_s + sin_cos_block, mask=mask, other=0.0
        ).to(tl.float32)
        sin = tl.load(
            sin_ptr + pos_id * sin_stride_s + sin_cos_block, mask=mask, other=0.0
        ).to(tl.float32)
        if ROTARY_INTERLEAVED:
            sin = tl.where(even_lane, -sin, sin)
        else:
            sin = tl.where(is_low_half, -sin, sin)

        q_token = q_ptr + s_id * q_stride_s
        for off_h in range(0, NUM_Q_HEADS):
            ordered_cols = off_h * q_stride_h + (ordered_block * q_stride_d)
            rotated_cols = off_h * q_stride_h + (rotated_block * q_stride_d)

            q = tl.load(q_token + ordered_cols, mask=mask, other=0.0)
            rotated_q = tl.load(q_token + rotated_cols, mask=mask, other=0.0)
            tl.store(q_token + ordered_cols, q * cos + rotated_q * sin, mask=mask)

        k_token = k_ptr + s_id * k_stride_s
        for off_h in range(0, NUM_K_HEADS):
            ordered_cols = off_h * k_stride_h + (ordered_block * k_stride_d)
            rotated_cols = off_h * k_stride_h + (rotated_block * k_stride_d)

            k = tl.load(k_token + ordered_cols, mask=mask, other=0.0)
            rotated_k = tl.load(k_token + rotated_cols, mask=mask, other=0.0)
            tl.store(k_token + ordered_cols, k * cos + rotated_k * sin, mask=mask)


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.IntTensor] = None,
    rotary_interleaved: bool = False,
    inplace: bool = False,
):
    logger.debug("GEMS_ENFLAME APPLY_ROTARY_POS_EMB")
    assert k.shape[-1] == q.shape[-1]
    assert cos.shape[-1] == sin.shape[-1]
    assert cos.shape[-1] * 2 == q.shape[-1]
    assert cos.stride(-1) == 1
    assert sin.stride(-1) == 1

    q_shape = q.shape
    k_shape = k.shape

    assert q.shape[:-2] == k.shape[:-2]
    if position_ids is None:
        assert len(q.shape) == 4
        seq_len = q.shape[-3]
    else:
        assert position_ids.shape == q.shape[:-2]
        position_ids = position_ids.view(-1)
        seq_len = None

    if q.dtype == torch.int64:
        q = q.to(torch.int32)
    if k.dtype == torch.int64:
        k = k.to(torch.int32)
    if position_ids is not None and position_ids.dtype == torch.int64:
        position_ids = position_ids.to(torch.int32)

    q = q.view(-1, q.shape[-2], q.shape[-1])
    k = k.view(-1, k.shape[-2], k.shape[-1])

    n_tokens, q_heads, head_dim = q.shape
    k_heads = k.shape[-2]

    padded_head_dim = max(triton.next_power_of_2(head_dim), 16)
    # This path selection is hot code on the host (a single call spends about 100us entirely in Python overhead),
    # so block_q/block_k are computed only when the merged path is actually taken.
    fuse_heads = n_tokens > FUSE_HEADS_TOKEN_THRESHOLD
    if fuse_heads:
        block_q = triton.next_power_of_2(q_heads * padded_head_dim)
        block_k = triton.next_power_of_2(k_heads * padded_head_dim)
        fuse_heads = max(block_q, block_k) <= FUSE_HEADS_MAX_BLOCK

    p_stride_s = position_ids.stride(0) if position_ids is not None else 0
    grid_dim = max(1, min(n_tokens, NUM_SIPS * 2))
    grid = (grid_dim,)

    if inplace:
        with torch_device_fn.device(q.device):
            apply_rotary_pos_emb_inplace_kernel[grid](
                q,
                k,
                cos,
                sin,
                position_ids,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                p_stride_s,
                cos.stride(0),
                sin.stride(0),
                seq_len,
                n_tokens,
                grid_dim,
                q_heads,
                k_heads,
                head_dim,
                padded_head_dim,
                rotary_interleaved,
                MAX_POSITION_EMBEDDINGS=cos.shape[0],
            )
        return q.view(q_shape), k.view(k_shape)

    q_embed = torch.empty_like(q)
    k_embed = torch.empty_like(k)

    with torch_device_fn.device(q_embed.device):
        if fuse_heads:
            apply_rotary_pos_emb_kernel[grid](
                q_embed,
                k_embed,
                q,
                k,
                cos,
                sin,
                position_ids,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                q_embed.stride(0),
                q_embed.stride(1),
                q_embed.stride(2),
                k_embed.stride(0),
                k_embed.stride(1),
                k_embed.stride(2),
                p_stride_s,
                cos.stride(0),
                sin.stride(0),
                seq_len,
                n_tokens,
                grid_dim,
                q_heads,
                k_heads,
                head_dim,
                padded_head_dim,
                block_q,
                block_k,
                rotary_interleaved,
                MAX_POSITION_EMBEDDINGS=cos.shape[0],
            )
        else:
            apply_rotary_pos_emb_perhead_kernel[grid](
                q_embed,
                k_embed,
                q,
                k,
                cos,
                sin,
                position_ids,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                q_embed.stride(0),
                q_embed.stride(1),
                q_embed.stride(2),
                k_embed.stride(0),
                k_embed.stride(1),
                k_embed.stride(2),
                p_stride_s,
                cos.stride(0),
                sin.stride(0),
                seq_len,
                n_tokens,
                grid_dim,
                q_heads,
                k_heads,
                head_dim,
                padded_head_dim,
                rotary_interleaved,
                MAX_POSITION_EMBEDDINGS=cos.shape[0],
            )
    q_embed = q_embed.view(q_shape)
    k_embed = k_embed.view(k_shape)
    return q_embed, k_embed
