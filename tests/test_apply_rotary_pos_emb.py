import random
import time
from typing import Optional

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)


# Copied from transformers.models.llama.modeling_llama.rotate_half
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.cohere.modeling_cohere.rotate_half
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere/modeling_cohere.py
def rotate_interleave(x):
    """Rotates interleave the hidden dims of the input."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _torch_apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.Tensor] = None,
    rotary_interleaved: bool = False,
):
    q = q.float()
    k = k.float()
    if position_ids is None:
        cos = cos[None, : q.size(-3), None, :]
        sin = sin[None, : q.size(-3), None, :]
    else:
        cos = cos[position_ids].unsqueeze(-2)  # [bs, seq_len, 1, dim/2]
        sin = sin[position_ids].unsqueeze(-2)  # [bs, seq_len, 1, dim/2]
    if rotary_interleaved:
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # [bs, seq_len, 1, dim]
        sin = torch.repeat_interleave(sin, 2, dim=-1)  # [bs, seq_len, 1, dim]
        rotate_fn = rotate_interleave
    else:
        cos = torch.cat([cos, cos], dim=-1)  # [bs, seq_len, 1, dim]
        sin = torch.cat([sin, sin], dim=-1)  # [bs, seq_len, 1, dim]
        rotate_fn = rotate_half

    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)

    return q_embed, k_embed


def _get_rope_cos_sin(max_seq_len, dim, dtype, base=10000, device=flag_gems.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)

    return cos, sin


@pytest.mark.apply_rotary_pos_emb
@pytest.mark.parametrize("batch_size", [2] if cfg.TO_CPU else [4, 8])
@pytest.mark.parametrize("max_seq_len", [16] if cfg.TO_CPU else [512, 2048])
@pytest.mark.parametrize("q_heads,k_heads", [(8, 1), (6, 2), (1, 1), (8, 8)])
@pytest.mark.parametrize("head_dim", [8] if cfg.TO_CPU else [64, 96, 128, 256])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("rotary_interleaved", [True, False])
@pytest.mark.parametrize("has_pos_id", [True, False])
def test_apply_rotary_pos_emb(
    batch_size,
    max_seq_len,
    q_heads,
    k_heads,
    head_dim,
    dtype,
    has_pos_id,
    rotary_interleaved,
):
    seq_len = torch.randint(1, max_seq_len, (1,)).item()
    q = torch.randn(
        (batch_size, seq_len, q_heads, head_dim), dtype=dtype, device=flag_gems.device
    )
    k = torch.randn(
        (batch_size, seq_len, k_heads, head_dim), dtype=dtype, device=flag_gems.device
    )

    position_ids = torch.randint(
        0, max_seq_len, (batch_size, seq_len), device=flag_gems.device
    )
    cos, sin = _get_rope_cos_sin(max_seq_len, head_dim, dtype, device=flag_gems.device)

    ref_q = utils.to_reference(q, True)
    ref_k = utils.to_reference(k, True)
    ref_cos = utils.to_reference(cos, True)
    ref_sin = utils.to_reference(sin, True)
    ref_position_ids = utils.to_reference(position_ids)

    q_embed_ref, k_embed_ref = _torch_apply_rotary_pos_emb(
        q=ref_q,
        k=ref_k,
        cos=ref_cos,
        sin=ref_sin,
        position_ids=ref_position_ids if has_pos_id else None,
        rotary_interleaved=rotary_interleaved,
    )
    q_embed_out, k_embed_out = flag_gems.apply_rotary_pos_emb(
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        position_ids=position_ids if has_pos_id else None,
        rotary_interleaved=rotary_interleaved,
    )

    utils.gems_assert_close(q_embed_out, q_embed_ref, dtype)
    utils.gems_assert_close(k_embed_out, k_embed_ref, dtype)
