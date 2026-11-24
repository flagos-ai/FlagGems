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
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    MIN_TILE_ROWS,
    tensorview_call,
    tile_rows_for_ub,
)
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle import tile_load, tile_store


@triton.jit
def rotary_embedding_rw_kernel(
    state_out,
    state,
    cos,
    sin,
    stride_state_n,
    stride_state_h,
    stride_state_d,
    stride_cos_n,
    stride_cos_d,
    num_tokens,
    num_heads,
    token_range,
    head_range,
    dim_range_x,
    dim_range_y,
    rotary_interleaved: tl.constexpr,
):
    if rotary_interleaved:
        # interleaved: dim_range_x/y are scalars (d*2 and d*2+1)
        # offsets and masks are 2-D: [BLOCK_N, BLOCK_H]
        state_x_offset = (
            token_range[:, None] * stride_state_n
            + head_range[None, :] * stride_state_h
            + dim_range_x * stride_state_d
        )
        state_y_offset = (
            token_range[:, None] * stride_state_n
            + head_range[None, :] * stride_state_h
            + dim_range_y * stride_state_d
        )
        # cos/sin index for pair (2i, 2i+1) is i = dim_range_x // 2
        cos_sin_idx = dim_range_x // 2
        cos_offset = token_range[:, None] * stride_cos_n + cos_sin_idx * stride_cos_d
        sin_offset = cos_offset
        state_mask = (token_range[:, None] < num_tokens) & (
            head_range[None, :] < num_heads
        )
        cos_mask = token_range[:, None] < num_tokens
    else:
        # non-interleaved: dim_range_x/y are vectors [0..D/2) and [D/2..D)
        # offsets and masks are 3-D: [BLOCK_N, BLOCK_H, BLOCK_D//2]
        state_x_offset = (
            token_range[:, None, None] * stride_state_n
            + head_range[None, :, None] * stride_state_h
            + dim_range_x[None, None, :] * stride_state_d
        )
        state_y_offset = (
            token_range[:, None, None] * stride_state_n
            + head_range[None, :, None] * stride_state_h
            + dim_range_y[None, None, :] * stride_state_d
        )
        # cos/sin both indexed by [0..D/2), using dim_range_x (front half indices)
        cos_offset = (
            token_range[:, None, None] * stride_cos_n
            + dim_range_x[None, None, :] * stride_cos_d
        )
        sin_offset = cos_offset
        state_mask = (token_range[:, None, None] < num_tokens) & (
            head_range[None, :, None] < num_heads
        )
        cos_mask = token_range[:, None, None] < num_tokens

    state_x = tl.load(state + state_x_offset, mask=state_mask, other=0.0)
    state_y = tl.load(state + state_y_offset, mask=state_mask, other=0.0)
    cos_loaded = tl.load(cos + cos_offset, mask=cos_mask, other=0.0).to(tl.float32)
    sin_loaded = tl.load(sin + sin_offset, mask=cos_mask, other=0.0).to(tl.float32)

    # Standard RoPE rotation:
    #   out_x = x * cos - y * sin  (front half / even positions)
    #   out_y = x * sin + y * cos  (back half / odd positions)
    out_x = state_x * cos_loaded - state_y * sin_loaded
    out_y = state_x * sin_loaded + state_y * cos_loaded

    tl.store(state_out + state_x_offset, out_x, mask=state_mask)
    tl.store(state_out + state_y_offset, out_y, mask=state_mask)


@libentry()
@triton.jit
def rotary_embedding_siso_kernel(
    state_out,  # [num_tokens, num_heads, head_dim]
    state,  # [num_tokens, num_heads, head_dim]
    cos,  # [num_tokens, head_dim // 2]  (already indexed by position_ids)
    sin,  # [num_tokens, head_dim // 2]  (already indexed by position_ids)
    stride_state_n,
    stride_state_h,
    stride_state_d,
    stride_cos_n,
    stride_cos_d,
    num_tokens,
    num_heads,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    rotary_interleaved: tl.constexpr,
):
    token_index = tl.program_id(0)
    token_range = token_index * BLOCK_N + tl.arange(0, BLOCK_N)
    head_index = tl.program_id(1)
    head_range = head_index * BLOCK_H + tl.arange(0, BLOCK_H)

    if rotary_interleaved:
        # interleaved: process each (2i, 2i+1) pair one at a time
        for d in range(0, BLOCK_D // 2):
            dim_range_x = tl.full([], d * 2, dtype=tl.int32)
            dim_range_y = tl.full([], d * 2 + 1, dtype=tl.int32)

            rotary_embedding_rw_kernel(
                state_out,
                state,
                cos,
                sin,
                stride_state_n,
                stride_state_h,
                stride_state_d,
                stride_cos_n,
                stride_cos_d,
                num_tokens,
                num_heads,
                token_range,
                head_range,
                dim_range_x,
                dim_range_y,
                rotary_interleaved,
            )
    else:
        # non-interleaved: front half [0..D/2) paired with back half [D/2..D)
        dim_range_x = tl.arange(0, BLOCK_D // 2)
        dim_range_y = tl.arange(BLOCK_D // 2, BLOCK_D)
        rotary_embedding_rw_kernel(
            state_out,
            state,
            cos,
            sin,
            stride_state_n,
            stride_state_h,
            stride_state_d,
            stride_cos_n,
            stride_cos_d,
            num_tokens,
            num_heads,
            token_range,
            head_range,
            dim_range_x,
            dim_range_y,
            rotary_interleaved,
        )


def _apply_rotary_pos_emb_generic(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.IntTensor] = None,
    rotary_interleaved: bool = False,
    inplace: bool = False,
):
    """
    Apply rotary position embedding to q and k.

    Args:
        q: (*, q_heads, head_dim)
        k: (*, k_heads, head_dim)
        cos: (max_seq_len, head_dim // 2)
        sin: (max_seq_len, head_dim // 2)
        position_ids: (*, ) optional; if None, positions are taken as 0..seq_len-1
        rotary_interleaved: whether head_dim is rotated in an interleaved (GPT-NeoX) style
        inplace: if True, modify q and k in place

    Returns:
        q_embed: (*, q_heads, head_dim)
        k_embed: (*, k_heads, head_dim)
    """
    logger.debug("GEMS_ASCEND ROTARY_POS_EMBEDDING")
    assert (
        k.shape[-1] == q.shape[-1]
    ), f"q and k must have the same last dimension, got {q.shape} and {k.shape}"
    assert (
        cos.shape[-1] == sin.shape[-1]
    ), f"cos and sin must have the same last dimension, got {cos.shape} and {sin.shape}"
    assert (
        cos.shape[-1] * 2 == q.shape[-1]
    ), f"cos/sin dim must be half of q/k dim, got {cos.shape} and {q.shape}"
    assert cos.stride(-1) == 1, "cos must be contiguous at the last dimension"
    assert sin.stride(-1) == 1, "sin must be contiguous at the last dimension"
    assert (
        q.shape[:-2] == k.shape[:-2]
    ), f"q and k must have the same batch/seq shape, got {q.shape[:-2]} and {k.shape[:-2]}"

    q_shape = q.shape
    k_shape = k.shape

    if position_ids is None:
        assert (
            len(q.shape) == 4
        ), f"q must be 4-D when position_ids is not provided, got {q.shape}"
        seq_len = q.shape[-3]
        # Match flattened q/k token order: [pos0..posN] for every batch.
        cos_sel = cos[:seq_len].repeat(q.shape[0], 1)
        sin_sel = sin[:seq_len].repeat(q.shape[0], 1)
    else:
        assert (
            position_ids.shape == q.shape[:-2]
        ), f"position_ids shape {position_ids.shape} must match q batch/seq shape {q.shape[:-2]}"
        position_ids = position_ids.view(-1)
        # gather rows by position: shape [n_tokens, D/2]
        cos_sel = cos[position_ids]
        sin_sel = sin[position_ids]

    q = q.view(-1, q.shape[-2], q.shape[-1])
    k = k.view(-1, k.shape[-2], k.shape[-1])

    num_tokens = q.shape[0]
    num_q_heads = q.shape[1]
    num_k_heads = k.shape[1]

    BLOCK_N = 8
    BLOCK_H = 4
    head_dim = q.shape[-1]

    def _launch(state_out, state, num_heads):
        grid = (
            triton.cdiv(num_tokens, BLOCK_N),
            triton.cdiv(num_heads, BLOCK_H),
        )
        with torch_device_fn.device(state_out.device):
            rotary_embedding_siso_kernel[grid](
                state_out,
                state,
                cos_sel,
                sin_sel,
                state.stride(0),
                state.stride(1),
                state.stride(2),
                cos_sel.stride(0),
                cos_sel.stride(1),
                num_tokens,
                num_heads,
                BLOCK_N=BLOCK_N,
                BLOCK_H=BLOCK_H,
                BLOCK_D=head_dim,
                rotary_interleaved=rotary_interleaved,
            )

    if inplace:
        _launch(q, q, num_q_heads)
        _launch(k, k, num_k_heads)
        return q.view(q_shape), k.view(k_shape)
    else:
        q_embed = torch.empty_like(q)
        k_embed = torch.empty_like(k)
        _launch(q_embed, q, num_q_heads)
        _launch(k_embed, k, num_k_heads)
        return q_embed.view(q_shape), k_embed.view(k_shape)


# ---------------------------------------------------------------------------
# TensorView fast path
#
# The generic path materializes cos[position_ids] and sin[position_ids] as
# [tokens, head_dim // 2] tensors before launching, then runs one program per
# (token block, head block). Here each program takes BLOCK_T tokens with all of
# their heads: q, k and both outputs are 3-D PartitionViews over
# [token, head, half], so a whole tile of head vectors moves as one block DMA
# and the cos/sin rows are gathered inside the kernel instead of being
# materialized first. Tail tokens and the heads past the head count fall
# outside the views and are dropped on store.
# ---------------------------------------------------------------------------
TV_MAX_BLOCK_T = 16

if HAS_TENSORVIEW:

    @triton.jit
    def rotary_tv_kernel(
        q_out,
        k_out,
        q,
        k,
        cos,
        sin,
        position_ids,
        num_tokens,
        cos_stride,
        NUM_Q: tl.constexpr,
        NUM_K: tl.constexpr,
        Q_PAD: tl.constexpr,
        K_PAD: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        tile = tl.program_id(0)
        half: tl.constexpr = HEAD_DIM // 2
        rows = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        live = rows < num_tokens
        # The cos/sin rows are the only data-dependent access, so they stay
        # masked loads rather than going through a view.
        pos = tl.load(position_ids + rows, mask=live, other=0)
        table = pos[:, None] * cos_stride + tl.arange(0, half)[None, :]
        c = tl.load(cos + table, mask=live[:, None], other=0).to(tl.float32)[:, None, :]
        s = tl.load(sin + table, mask=live[:, None], other=0).to(tl.float32)[:, None, :]

        q_view = tle_exp.make_partition_view(
            q,
            shape=[num_tokens, NUM_Q, HEAD_DIM],
            strides=[NUM_Q * HEAD_DIM, HEAD_DIM, 1],
            tile=[BLOCK_T, Q_PAD, half],
        )
        qo_view = tle_exp.make_partition_view(
            q_out,
            shape=[num_tokens, NUM_Q, HEAD_DIM],
            strides=[NUM_Q * HEAD_DIM, HEAD_DIM, 1],
            tile=[BLOCK_T, Q_PAD, half],
        )
        q1 = tile_load(
            q_view, index=(tile, 0, 0), dst_space=tle_exp.language.dsa.ascend.UB
        ).to(tl.float32)
        q2 = tile_load(
            q_view, index=(tile, 0, 1), dst_space=tle_exp.language.dsa.ascend.UB
        ).to(tl.float32)
        tile_store(
            qo_view,
            value=(q1 * c - q2 * s).to(q_out.type.element_ty),
            index=(tile, 0, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )
        tile_store(
            qo_view,
            value=(q2 * c + q1 * s).to(q_out.type.element_ty),
            index=(tile, 0, 1),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

        k_view = tle_exp.make_partition_view(
            k,
            shape=[num_tokens, NUM_K, HEAD_DIM],
            strides=[NUM_K * HEAD_DIM, HEAD_DIM, 1],
            tile=[BLOCK_T, K_PAD, half],
        )
        ko_view = tle_exp.make_partition_view(
            k_out,
            shape=[num_tokens, NUM_K, HEAD_DIM],
            strides=[NUM_K * HEAD_DIM, HEAD_DIM, 1],
            tile=[BLOCK_T, K_PAD, half],
        )
        k1 = tile_load(
            k_view, index=(tile, 0, 0), dst_space=tle_exp.language.dsa.ascend.UB
        ).to(tl.float32)
        k2 = tile_load(
            k_view, index=(tile, 0, 1), dst_space=tle_exp.language.dsa.ascend.UB
        ).to(tl.float32)
        tile_store(
            ko_view,
            value=(k1 * c - k2 * s).to(k_out.type.element_ty),
            index=(tile, 0, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )
        tile_store(
            ko_view,
            value=(k2 * c + k1 * s).to(k_out.type.element_ty),
            index=(tile, 0, 1),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

    def _tv_rows_that_fit(q_pad, k_pad, head_dim, element_size):
        """Tokens per tile UB can hold. The widest tile is max(q_pad, k_pad)
        head vectors of half the head dim; a half tile and its fp32 promotion
        are live for both halves at once, so budget for four."""
        row_bytes = max(q_pad, k_pad) * (head_dim // 2) * element_size
        return tile_rows_for_ub(row_bytes, 4, TV_MAX_BLOCK_T)

    def _tv_block_t(q_pad, k_pad, head_dim, element_size):
        rows = _tv_rows_that_fit(q_pad, k_pad, head_dim, element_size)
        return max(rows, MIN_TILE_ROWS)

    def _apply_rotary_pos_emb_tensorview(
        q,
        k,
        cos,
        sin,
        position_ids=None,
        rotary_interleaved=False,
        inplace=False,
    ):
        logger.debug("GEMS_ASCEND ROTARY_POS_EMBEDDING(TENSORVIEW)")
        q_shape, k_shape = q.shape, k.shape
        q = q.contiguous().view(-1, q.shape[-2], q.shape[-1])
        k = k.contiguous().view(-1, k.shape[-2], k.shape[-1])
        num_tokens, num_q, head_dim = q.shape
        num_k = k.shape[1]
        q_pad = triton.next_power_of_2(num_q)
        k_pad = triton.next_power_of_2(num_k)
        block_t = _tv_block_t(q_pad, k_pad, head_dim, q.element_size())

        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        grid = (triton.cdiv(num_tokens, block_t),)
        with torch_device_fn.device(q.device):
            rotary_tv_kernel[grid](
                q_out,
                k_out,
                q,
                k,
                cos,
                sin,
                position_ids.view(-1),
                num_tokens,
                cos.stride(0),
                NUM_Q=num_q,
                NUM_K=num_k,
                Q_PAD=q_pad,
                K_PAD=k_pad,
                HEAD_DIM=head_dim,
                BLOCK_T=block_t,
            )
        return q_out.view(q_shape), k_out.view(k_shape)

    def _tv_eligible(q, k, cos, sin, position_ids, rotary_interleaved, inplace):
        # The tile covers the half-split layout only, and writing into the
        # inputs would race with the second half's read of them.
        if rotary_interleaved or inplace:
            return False
        # Without position_ids the positions are implicit per batch row, which
        # the kernel would have to materialize; leave that to the generic path.
        if position_ids is None:
            return False
        if q.dim() < 3 or k.dim() < 3 or q.numel() == 0 or k.numel() == 0:
            return False
        if q.shape[:-2] != k.shape[:-2] or q.shape[-1] != k.shape[-1]:
            return False
        # The kernel reads it as one flat run of tokens.
        if position_ids.shape != q.shape[:-2] or not position_ids.is_contiguous():
            return False
        head_dim = q.shape[-1]
        half = head_dim // 2
        if head_dim % 2 != 0:
            return False
        # The tile's last extent is `half` and the cos/sin gather indexes it
        # with tl.arange, so it has to be a power of two.
        if triton.next_power_of_2(half) != half:
            return False
        if cos.shape != sin.shape or cos.dim() != 2 or cos.shape[-1] != half:
            return False
        if cos.stride(-1) != 1 or sin.stride(-1) != 1:
            return False
        # One row stride is passed for both tables, so they have to agree on it.
        if cos.stride(0) != sin.stride(0):
            return False
        if q.dtype != k.dtype or cos.dtype != sin.dtype:
            return False
        q_pad = triton.next_power_of_2(q.shape[-2])
        k_pad = triton.next_power_of_2(k.shape[-2])
        return (
            _tv_rows_that_fit(q_pad, k_pad, head_dim, q.element_size()) >= MIN_TILE_ROWS
        )

    def apply_rotary_pos_emb(
        q,
        k,
        cos,
        sin,
        position_ids: Optional[torch.IntTensor] = None,
        rotary_interleaved: bool = False,
        inplace: bool = False,
    ):
        if not _tv_eligible(q, k, cos, sin, position_ids, rotary_interleaved, inplace):
            return _apply_rotary_pos_emb_generic(
                q, k, cos, sin, position_ids, rotary_interleaved, inplace
            )
        return tensorview_call(
            "ROTARY_POS_EMBEDDING",
            _apply_rotary_pos_emb_tensorview,
            _apply_rotary_pos_emb_generic,
            q,
            k,
            cos,
            sin,
            position_ids,
            rotary_interleaved,
            inplace,
        )

else:
    apply_rotary_pos_emb = _apply_rotary_pos_emb_generic
