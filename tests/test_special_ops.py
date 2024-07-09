from typing import Optional

import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_close,
    to_reference,
)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("p", [0.3, 0.6, 0.9])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_dropout(shape, p, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    ref_inp = to_reference(inp)

    ref_out = torch.nn.functional.dropout(ref_inp, p, True)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.dropout(inp, p, True)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)

    res_out = to_reference(res_out)
    res_in_grad = to_reference(res_in_grad)

    exp_equal = (p * p + (1 - p) * (1 - p)) * inp.numel()
    num_equal = torch.sum(torch.isclose(ref_out, res_out)).item()
    assert (
        abs(num_equal - exp_equal) / exp_equal <= 0.05
    ), f"num_equal: {num_equal}, exp_equal: {exp_equal}, num_total: {inp.numel()}"

    num_equal = torch.sum(torch.isclose(ref_in_grad, res_in_grad)).item()
    assert (
        abs(num_equal - exp_equal) / exp_equal <= 0.05
    ), f"num_equal: {num_equal}, exp_equal: {exp_equal}, num_total: {inp.numel()}"


def get_rope_cos_sin(max_seq_len, dim, dtype, base=10000, device="cuda"):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


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


def torch_apply_rotary_pos_emb(
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


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("max_seq_len", [512, 2048])
@pytest.mark.parametrize("q_heads,k_heads", [(8, 1), (6, 2), (1, 1), (8, 8)])
@pytest.mark.parametrize("head_dim", [64, 96, 128, 256])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
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
        (batch_size, seq_len, q_heads, head_dim), dtype=dtype, device="cuda"
    )
    k = torch.randn(
        (batch_size, seq_len, k_heads, head_dim), dtype=dtype, device="cuda"
    )

    position_ids = torch.randint(0, max_seq_len, (batch_size, seq_len), device="cuda")
    cos, sin = get_rope_cos_sin(max_seq_len, head_dim, dtype, device="cuda")

    ref_q = to_reference(q, True)
    ref_k = to_reference(k, True)
    ref_cos = to_reference(cos, True)
    ref_sin = to_reference(sin, True)
    ref_position_ids = to_reference(position_ids)

    q_embed_ref, k_embed_ref = torch_apply_rotary_pos_emb(
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

    gems_assert_close(q_embed_out, q_embed_ref, dtype)
    gems_assert_close(k_embed_out, k_embed_ref, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rand(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.rand(shape, dtype=dtype, device="cuda")
    assert (res_out <= 1.0).all()
    assert (res_out >= 0.0).all()


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_randn(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.randn(shape, dtype=dtype, device="cuda")
    mean = torch.mean(res_out)
    std = torch.std(res_out)
    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rand_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        res_out = torch.rand_like(x)
    assert (res_out <= 1.0).all()
    assert (res_out >= 0.0).all()
