from typing import Optional

import numpy as np
import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_SHAPES,
    RESOLUTION,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)
from .conftest import TO_CPU


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("p", [0.3, 0.6, 0.9])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_dropout(shape, p, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    ref_inp = to_reference(inp)

    # NOTE: ensure that scalars are float32(instead of float64)
    # in some cases, casting up then casting down have different result
    p = np.float32(p)
    one_minus_p = np.float32(1.0) - p

    ref_out = torch.nn.functional.dropout(ref_inp, p, True)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.dropout(inp, p, True)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)

    res_out = to_reference(res_out)
    res_in_grad = to_reference(res_in_grad)

    exp_equal = (p * p + one_minus_p * one_minus_p) * inp.numel()
    num_equal = torch.sum(torch.isclose(ref_out, res_out)).item()
    if TO_CPU:
        zero_equal = torch.eq(res_out, torch.zeros_like(res_out))
        num_zero = torch.sum(zero_equal).item()
        assert abs(num_zero / inp.numel() - p) <= 0.05
        scale_equal = torch.isclose(
            res_out, ref_inp / one_minus_p, rtol=RESOLUTION[dtype]
        )
        assert torch.all(torch.logical_or(zero_equal, scale_equal))
    else:
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


@pytest.mark.parametrize("EmbeddingSize", [4096])
@pytest.mark.parametrize("Batch", [2, 4])
@pytest.mark.parametrize("M", [4, 8])
@pytest.mark.parametrize("N", [128, 256, 4096])
@pytest.mark.parametrize("padding_idx", [None, -1, 1, 2])
@pytest.mark.parametrize("scale_grad_by_freq", [True, False])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32]
)  # triton.atomic_add still not support bf16
def test_embedding(EmbeddingSize, Batch, M, N, padding_idx, scale_grad_by_freq, dtype):
    indices = torch.randint(
        0, EmbeddingSize, (Batch, M), device="cuda", requires_grad=False
    )
    embedding = torch.randn(
        (EmbeddingSize, N), device="cuda", dtype=dtype, requires_grad=True
    )
    ref_embedding = to_reference(embedding)

    res_out = torch.nn.functional.embedding(
        indices, embedding, padding_idx, scale_grad_by_freq=scale_grad_by_freq
    )
    with flag_gems.use_gems():
        ref_out = torch.nn.functional.embedding(
            indices, ref_embedding, padding_idx, scale_grad_by_freq=scale_grad_by_freq
        )
    out_grad = torch.randn_like(ref_out)
    ref_grad = to_reference(out_grad)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_embedding, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, embedding, out_grad)

    gems_assert_close(ref_out, res_out, dtype)
    gems_assert_close(ref_in_grad, res_in_grad, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.cfloat])
def test_accuracy_resolve_neg(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cuda")
    y = x.conj()
    z = y.imag
    assert z.is_neg()
    with flag_gems.use_gems():
        out = z.resolve_neg()
    assert not out.is_neg()


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("hiddensize", [128, 256])
@pytest.mark.parametrize("topk", [5])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_topk(
    batch_size,
    hiddensize,
    topk,
    largest,
    dtype,
):
    x = torch.arange(hiddensize, dtype=dtype, device="cuda")
    x = x.repeat(batch_size).reshape(batch_size, hiddensize)

    # Each row use different shuffled index.
    for bsz in range(batch_size):
        col_indices = torch.randperm(x.size(1))
        x[bsz, :] = x[bsz, col_indices]

    ref_value, ref_index = torch.topk(x, topk, largest=largest)

    with flag_gems.use_gems():
        res_value, res_index = torch.topk(x, topk, largest=largest)

    gems_assert_close(ref_value, res_value, dtype)
    gems_assert_equal(ref_index, res_index)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.cfloat])
def test_accuracy_resolve_conj(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cuda")
    y = x.conj()
    assert y.is_conj()
    with flag_gems.use_gems():
        z = y.resolve_conj()
    assert not z.is_conj()


@pytest.mark.parametrize("shape", POINTWISE_SHAPES + [(8191,), (8192, 73739)])
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("sorted", [True])
@pytest.mark.parametrize("return_inverse", [True, False])
@pytest.mark.parametrize("return_counts", [False, True])
def test_accuracy_unique(shape, dtype, sorted, return_inverse, return_counts):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(-10, 10, shape, device="cuda").to(dtype)
    ref_inp = to_reference(inp, False)

    if return_counts:
        if return_inverse:
            with flag_gems.use_gems():
                res_out, res_unique_order, res_counts = torch.unique(
                    inp,
                    sorted=sorted,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
            ref_out, ref_unique_order, ref_counts = torch.unique(
                ref_inp,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )
            assert res_out.numel() == ref_out.numel()
            gems_assert_equal(res_unique_order, ref_unique_order)
        else:
            with flag_gems.use_gems():
                res_out, res_counts = torch.unique(
                    inp,
                    sorted=sorted,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
            ref_out, ref_counts = torch.unique(
                ref_inp,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )
            assert res_out.numel() == ref_out.numel()
        gems_assert_equal(res_counts, ref_counts)
    else:
        if return_inverse:
            with flag_gems.use_gems():
                res_out, res_unique_order = torch.unique(
                    inp,
                    sorted=sorted,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
            ref_out, ref_unique_order = torch.unique(
                ref_inp,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )
            assert res_out.numel() == ref_out.numel()
            gems_assert_equal(res_unique_order, ref_unique_order)
        else:
            with flag_gems.use_gems():
                res_out = torch.unique(
                    inp,
                    sorted=sorted,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
            ref_out = torch.unique(
                ref_inp,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )
            assert res_out.numel() == ref_out.numel()
    gems_assert_equal(res_out, ref_out)
