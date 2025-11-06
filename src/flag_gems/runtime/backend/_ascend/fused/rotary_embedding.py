import logging
from typing import Optional

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


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

    cos_sim_offset = (
        token_range[:, None, None] * stride_cos_n
        + dim_range_x[None, None, :] * stride_cos_d
    )
    if rotary_interleaved:
        sin_sim_offset = (
            token_range[:, None, None] * stride_cos_n
            + dim_range_y[None, None, :] * stride_cos_d
        )
    else:
        sin_sim_offset = cos_sim_offset

    state_x = tl.load(
        state + state_x_offset,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
        other=0.0,
    )
    state_y = tl.load(
        state + state_y_offset,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
        other=0.0,
    )

    cos_loaded = tl.load(
        cos + cos_sim_offset,
        mask=token_range[:, None, None] < num_tokens,
        other=0.0,
    ).to(tl.float32)
    sin_loaded = tl.load(
        sin + sin_sim_offset,
        mask=token_range[:, None, None] < num_tokens,
        other=0.0,
    ).to(tl.float32)

    out_x = state_x * cos_loaded - state_y * sin_loaded
    out_y = state_x * sin_loaded + state_y * cos_loaded

    tl.store(
        state_out + state_x_offset,
        out_x,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
    )
    tl.store(
        state_out + state_y_offset,
        out_y,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
    )


@libentry()
@triton.jit
def rotary_embedding_siso_kernel(
    state_out,  # [num_tokens, head_num, head_dim]
    state,  # [num_tokens, head_num, head_dim]
    cos,  # [num_tokens, 1, head_dim // 2]
    sin,  # [num_tokens, 1, head_dim // 2]
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
        for d in range(0, BLOCK_D // 2):
            dim_range_x = d * 2
            dim_range_y = d * 2 + 1

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


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.IntTensor] = None,
    rotary_interleaved: bool = False,
):
    """
    Apply rotary position embedding to q and k

    Args:
        q: (*, q_heads, head_dim)
        k: (*, k_heads, head_dim)
        cos: (max_seq_len, head_dim // 2)
        sin: (max_seq_len, head_dim // 2)
        position_ids: (*, ), optional, position ids for each token
        rotary_interleaved: whether the head_dim is rotated in an interleaved way

    Returns:
        q_embed: (*, q_heads, head_dim)
        k_embed: (*, k_heads, head_dim)
    """
    logging.debug("GEMS_ASCEND ROTARY POS EMBEDDING")
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

    q_shape = q.shape
    k_shape = k.shape

    assert (
        q.shape[:-2] == k.shape[:-2]
    ), f"q and k must have the same length, got {q.shape[:-2]} and {k.shape[:-2]}"
    if position_ids is None:
        assert (
            len(q.shape) == 4
        ), f"q must have 4 dimensions if position_ids is not provided, got {q.shape}"
    else:
        assert (
            position_ids.shape == q.shape[:-2]
        ), f"position_ids must have the same length as q, got {position_ids.shape} and {q.shape[:-2]}"

        position_ids = position_ids.view(-1)

    q = q.view(-1, q.shape[-2], q.shape[-1])
    k = k.view(-1, k.shape[-2], k.shape[-1])

    q_embed = torch.empty_like(q)
    k_embed = torch.empty_like(k)

    def torch_rotary_embedding(state_out, state, cos, sin):
        num_tokens = state.shape[0]
        num_heads = state.shape[1]
        head_dim = state.shape[-1]

        BLOCK_N = 8
        BLOCK_H = 4
        grid = (
            triton.cdiv(num_tokens, BLOCK_N),
            triton.cdiv(num_heads, BLOCK_H),
        )
        with torch_device_fn.device(state_out.device):
            with flag_gems.use_gems():
                if position_ids is None:
                    cos = cos[: q_shape[-3], None, :]
                    sin = sin[: q_shape[-3], None, :]
                else:
                    cos = cos[position_ids, None, :]
                    sin = sin[position_ids, None, :]

                if rotary_interleaved:
                    cos = torch.repeat_interleave(cos, 2, dim=-1)
                    sin = torch.repeat_interleave(sin, 2, dim=-1)
                orig_cos = cos
                orig_sin = sin
                for _ in range(q_shape[0] - 1):
                    cos = torch.cat((cos, orig_cos), dim=0)
                    sin = torch.cat((sin, orig_sin), dim=0)
            rotary_embedding_siso_kernel[grid](
                state_out,
                state,
                cos,
                sin,
                state.stride(0),
                state.stride(1),
                state.stride(2),
                cos.stride(0),
                cos.stride(2),
                num_tokens,
                num_heads,
                BLOCK_N=BLOCK_N,
                BLOCK_H=BLOCK_H,
                BLOCK_D=head_dim,
                rotary_interleaved=rotary_interleaved,
            )

    torch_rotary_embedding(q_embed, q, cos, sin)
    torch_rotary_embedding(k_embed, k, cos, sin)

    q_embed = q_embed.view(q_shape)
    k_embed = k_embed.view(k_shape)
    return q_embed, k_embed
