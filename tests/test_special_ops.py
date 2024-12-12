import itertools
from typing import Optional

import numpy as np
import pytest
import torch

import flag_gems

from .accuracy_utils import (
    ALL_INT_DTYPES,
    BOOL_TYPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    SPECIAL_SHAPES,
    STACK_DIM_LIST,
    STACK_SHAPES,
    UPSAMPLE_SHAPES,
    UT_SHAPES_1D,
    UT_SHAPES_2D,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)
from .conftest import TO_CPU


# TODO: sometimes failed at (8192,), 0.6, bfloat16
@pytest.mark.dropout
@pytest.mark.native_dropout
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("p", [0.3, 0.6, 0.9])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_dropout(shape, p, dtype):
    if TO_CPU or shape == (1,):
        shape = (32768,)
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
        from flag_gems.testing import RESOLUTION

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


@pytest.mark.apply_rotary_pos_emb
@pytest.mark.parametrize("batch_size", [2] if TO_CPU else [4, 8])
@pytest.mark.parametrize("max_seq_len", [16] if TO_CPU else [512, 2048])
@pytest.mark.parametrize("q_heads,k_heads", [(8, 1), (6, 2), (1, 1), (8, 8)])
@pytest.mark.parametrize("head_dim", [8] if TO_CPU else [64, 96, 128, 256])
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


# TODO: failed when EmbeddingSize is small
@pytest.mark.embedding
@pytest.mark.parametrize("EmbeddingSize", [1024] if TO_CPU else [4096])
@pytest.mark.parametrize("Batch", [2] if TO_CPU else [2, 4])
@pytest.mark.parametrize("M", [4] if TO_CPU else [4, 8])
@pytest.mark.parametrize("N", [8] if TO_CPU else [128, 256, 4096])
@pytest.mark.parametrize("padding_idx", [None, -1, 1, 2])
@pytest.mark.parametrize("scale_grad_by_freq", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_embedding(EmbeddingSize, Batch, M, N, padding_idx, scale_grad_by_freq, dtype):
    indices = torch.randint(
        0, EmbeddingSize, (Batch, M), device="cuda", requires_grad=False
    )
    embedding = torch.randn(
        (EmbeddingSize, N), device="cuda", dtype=dtype, requires_grad=True
    )
    ref_embedding = to_reference(embedding)
    ref_indices = to_reference(indices)

    ref_out = torch.nn.functional.embedding(
        ref_indices, ref_embedding, padding_idx, scale_grad_by_freq=scale_grad_by_freq
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.embedding(
            indices, embedding, padding_idx, scale_grad_by_freq=scale_grad_by_freq
        )
    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_embedding, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, embedding, out_grad)

    gems_assert_close(res_out, ref_out, dtype)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.resolve_neg
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.cfloat])
def test_accuracy_resolve_neg(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cuda")
    y = x.conj()
    z = y.imag
    assert z.is_neg()
    with flag_gems.use_gems():
        out = z.resolve_neg()
    assert not out.is_neg()


@pytest.mark.topk
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
    ref_x = to_reference(x)
    ref_value, ref_index = torch.topk(ref_x, topk, largest=largest)

    with flag_gems.use_gems():
        res_value, res_index = torch.topk(x, topk, largest=largest)

    gems_assert_close(res_value, ref_value, dtype)
    gems_assert_equal(res_index, ref_index)


@pytest.mark.resolve_conj
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.cfloat])
def test_accuracy_resolve_conj(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cuda")
    y = x.conj()
    assert y.is_conj()
    with flag_gems.use_gems():
        z = y.resolve_conj()
    assert not z.is_conj()


@pytest.mark.unique
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
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


@pytest.mark.multinomial
@pytest.mark.parametrize("shape", UT_SHAPES_1D + UT_SHAPES_2D)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("n_samples", [1000])
def test_accuracy_multinomial_with_replacement(shape, dtype, n_samples):
    if shape[-1] == 1:
        dist = torch.rand(size=shape, dtype=dtype, device="cuda")
        with flag_gems.use_gems():
            res_out = torch.multinomial(dist, n_samples, True)
        assert torch.all(res_out == 0)
    else:
        # Mask p% off of the categories and test the sampling results fall in the rest
        for p in (0.1, 0.5, 0.9):
            dist = torch.rand(size=shape, dtype=dtype, device="cuda")
            dist[torch.rand(shape) < p] = 0
            # Make sure there's at least one non-zero probability
            dist[..., -1] = 0.5
            with flag_gems.use_gems():
                res_out = torch.multinomial(dist, n_samples, True)
            res_dist = torch.gather(dist, -1, res_out)
            # assert torch.all(res_dist)
            assert torch.sum(res_dist == 0) / res_dist.numel() < 0.001


@pytest.mark.multinomial
@pytest.mark.parametrize("pool", UT_SHAPES_2D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_multinomial_without_replacement(pool, dtype):
    dist = torch.rand(size=pool, dtype=dtype, device="cuda")
    k = pool[-1]
    if k > 1:
        ns = [k // 2, k]
    else:
        ns = [1]
    for n in ns:
        with flag_gems.use_gems():
            out = torch.multinomial(dist, n, False)
        # Verifies uniqueness
        idx_cnt = torch.nn.functional.one_hot(out).sum(1)
        assert torch.all(idx_cnt <= 1)


@pytest.mark.pad
@pytest.mark.parametrize("shape", [[1024, 1024], [64, 64, 64, 64]])
@pytest.mark.parametrize("dtype", [torch.float32] if TO_CPU else FLOAT_DTYPES)
@pytest.mark.parametrize("pad_mode", ["constant", "reflect", "replicate", "circular"])
@pytest.mark.parametrize("contiguous", [True, False])
def test_pad(shape, dtype, pad_mode, contiguous):
    x = torch.randn(size=shape, dtype=dtype, device="cuda")
    if not contiguous:
        x = x[::2, ::2]

    ref_x = to_reference(x)

    rank = x.ndim
    pad_params = list(
        torch.randint(0, 10, (rank * 2,), dtype=torch.int32, device="cpu")
        if pad_mode == "constant"
        else torch.randint(0, 10, (rank,), dtype=torch.int32, device="cpu")
    )
    pad_value = float(torch.randint(0, 1024, (1,), dtype=torch.int32, device="cpu"))

    if pad_mode != "constant":
        pad_params = [(pad_val + 2 - 1) // 2 * 2 for pad_val in pad_params]
        pad_value = None

    ref_pad_params = [to_reference(pad_param) for pad_param in pad_params]

    ref_out = torch.nn.functional.pad(ref_x, ref_pad_params, pad_mode, pad_value)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.pad(x, pad_params, pad_mode, pad_value)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.upsample_bicubic2d_aa
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.7)])
@pytest.mark.parametrize(
    "shape",
    [
        (32, 16, 128, 128),
        (15, 37, 256, 256),
        (3, 5, 127, 127),
        (128, 192, 42, 51),
        (3, 7, 1023, 1025),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_upsample_bicubic2d_aa(dtype, shape, scale, align_corners):
    input = torch.rand(shape, dtype=dtype, device="cuda")
    ref_i = to_reference(input, True)
    output_size = tuple([int(input.shape[i + 2] * scale[i]) for i in range(2)])
    ref_out = torch._C._nn._upsample_bicubic2d_aa(
        ref_i, output_size=output_size, align_corners=align_corners
    )
    with flag_gems.use_gems():
        res_out = torch._C._nn._upsample_bicubic2d_aa(
            input, output_size=output_size, align_corners=align_corners
        )

    def span(scale):
        support = 2 if (scale >= 1.0) else 2.0 / scale
        interpolate_range = int(support + 0.5) * 2 + 1
        return interpolate_range

    reduce_dim = span(scale[0]) * span(scale[1])
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.5)])
@pytest.mark.parametrize("shape", UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_upsample_nearest2d(dtype, shape, scale):
    input = torch.randn(shape, dtype=dtype, device="cuda")
    ref_i = to_reference(input).to(torch.float32)
    output_size = [int(input.shape[i + 2] * scale[i]) for i in range(2)]
    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.arange
@pytest.mark.parametrize("start", [0, 1, 3])
@pytest.mark.parametrize("step", [1, 2, 5])
@pytest.mark.parametrize("end", [128, 256, 1024])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES + [None])
@pytest.mark.parametrize("device", ["cuda", None])
@pytest.mark.parametrize(
    "pin_memory", [False, None]
)  # Since triton only target to GPU, pin_memory only used in CPU tensors.
def test_arange(start, step, end, dtype, device, pin_memory):
    if TO_CPU:
        return
    ref_out = torch.arange(
        start, end, step, dtype=dtype, device=device, pin_memory=pin_memory
    )
    with flag_gems.use_gems():
        res_out = torch.arange(
            start, end, step, dtype=dtype, device=device, pin_memory=pin_memory
        )

    gems_assert_equal(res_out, ref_out)


@pytest.mark.isin
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("assume_unique", [False, True])
@pytest.mark.parametrize("invert", [False, True])
def test_accuracy_isin(shape, dtype, assume_unique, invert):
    inp1 = torch.randint(-100, 100, shape, device="cuda").to(dtype)
    test_numel = inp1.numel() // 2 if inp1.numel() > 1 else 1
    test_shape = (test_numel,)
    inp2 = torch.randint(-10, 10, test_shape, device="cuda").to(dtype)
    inp1.ravel()[-1] = 0
    if assume_unique:
        inp1 = torch.unique(inp1)
        inp2 = torch.unique(inp2)
    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)

    with flag_gems.use_gems():
        res_out = torch.isin(inp1, inp2, assume_unique=assume_unique, invert=invert)
    ref_out = torch.isin(ref_inp1, ref_inp2, assume_unique=assume_unique, invert=invert)
    gems_assert_equal(res_out, ref_out)

    inp1_s = inp1.ravel()[0].item()
    with flag_gems.use_gems():
        res1_out = torch.isin(inp1_s, inp2, assume_unique=assume_unique, invert=invert)
    ref1_out = torch.isin(inp1_s, ref_inp2, assume_unique=assume_unique, invert=invert)
    gems_assert_equal(res1_out, ref1_out)

    inp2_s = inp2.ravel()[0].item()
    with flag_gems.use_gems():
        res2_out = torch.isin(inp1, inp2_s, assume_unique=assume_unique, invert=invert)
    ref2_out = torch.isin(ref_inp1, inp2_s, assume_unique=assume_unique, invert=invert)
    gems_assert_equal(res2_out, ref2_out)

    inp0 = torch.tensor([], device="cuda")
    ref_inp0 = to_reference(inp0, False)
    with flag_gems.use_gems():
        res0_out = torch.isin(inp0, inp2, assume_unique=assume_unique, invert=invert)
    ref0_out = torch.isin(
        ref_inp0, ref_inp2, assume_unique=assume_unique, invert=invert
    )
    gems_assert_equal(res0_out, ref0_out)


@pytest.mark.fill
@pytest.mark.parametrize("value", [0, 1, 9])
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fill(value, shape, dtype):
    # Test fill.Scalar
    x = torch.ones(shape, device="cuda", dtype=dtype)
    ref_x = to_reference(x, False)

    ref_out = torch.fill(ref_x, value)
    with flag_gems.use_gems():
        res_out = torch.fill(x, value)

    gems_assert_equal(res_out, ref_out)

    # Test fill.Tensor
    value_tensor = torch.tensor(value, device="cuda", dtype=dtype)
    ref_out_tensor = torch.fill(ref_x, value_tensor)
    with flag_gems.use_gems():
        res_out_tensor = torch.fill(x, value_tensor)

    gems_assert_equal(res_out_tensor, ref_out_tensor)


@pytest.mark.stack
@pytest.mark.parametrize("shape", STACK_SHAPES)
@pytest.mark.parametrize("dim", STACK_DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_stack(shape, dim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device="cuda") for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cuda").to(
                dtype
            )
            for s in shape
        ]
    ref_inp = [to_reference(_) for _ in inp]
    ref_out = torch.stack(ref_inp, dim)

    with flag_gems.use_gems():
        res_out = torch.stack(inp, dim)
    gems_assert_equal(res_out, ref_out)


HSTACK_SHAPES = [
    [(8,), (16,)],
    [(16, 256), (16, 128)],
    [(20, 320, 15), (20, 160, 15), (20, 80, 15)],
]


@pytest.mark.hstack
@pytest.mark.parametrize("shape", HSTACK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_hstack(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device="cuda") for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cuda").to(
                dtype
            )
            for s in shape
        ]
    ref_inp = [to_reference(_) for _ in inp]
    ref_out = torch.hstack(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.hstack(inp)
    gems_assert_equal(res_out, ref_out)


HSTACK_EXCEPTION_SHAPES = [
    [(16, 256), (16,)],
    [(16, 256), (8, 128)],
]


@pytest.mark.hstack
@pytest.mark.parametrize("shape", HSTACK_EXCEPTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_exception_hstack(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device="cuda") for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cuda").to(
                dtype
            )
            for s in shape
        ]

    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            _ = torch.hstack(inp)


CAT_SHAPES = [
    [(1, 32), (8, 32)],
    [(16, 128), (32, 128)],
    [(1024, 1024), (1024, 1024)],
    [(1, 1024, 256), (8, 1024, 256), (16, 1024, 256)],
    [(16, 320, 15), (32, 320, 15), (64, 320, 15)],
    [(16, 128, 64, 64), (16, 128, 64, 64), (24, 128, 64, 64), (32, 128, 64, 64)],
]


def gen_cat_shapes_dim(shapes):
    results = []
    for tensor_shapes in shapes:
        assert all(
            [len(s) == len(tensor_shapes[0]) for s in tensor_shapes]
        ), "All tensor rank must agree."
        assert all(
            [s[-1] == tensor_shapes[0][-1] for s in tensor_shapes]
        ), "All tensor must have same shape except cat dim."
        rank = len(tensor_shapes[0])
        results.append([tensor_shapes, 0])
        for dim in range(1, rank):
            results.append(
                [[(s[dim], *s[1:dim], s[0], *s[dim + 1 :]) for s in tensor_shapes], dim]
            )
            results.append(
                [
                    [(s[dim], *s[1:dim], s[0], *s[dim + 1 :]) for s in tensor_shapes],
                    dim - rank,
                ]
            )
    return results


@pytest.mark.cat
@pytest.mark.parametrize("shape, dim", gen_cat_shapes_dim(CAT_SHAPES))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_cat(shape, dim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device="cuda") for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cuda").to(
                dtype
            )
            for s in shape
        ]
    ref_inp = [to_reference(_) for _ in inp]
    ref_out = torch.cat(ref_inp, dim)

    with flag_gems.use_gems():
        res_out = torch.cat(inp, dim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.cat
@pytest.mark.parametrize(
    "shape, dim",
    [
        (((0, 3), (2, 3)), 0),
        (((0, 3), (0, 3)), 0),
        (((0,), (0,)), 0),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_cat_empty_tensor(shape, dim, dtype):
    inp = [torch.randn(s, dtype=dtype, device="cuda") for s in shape]
    ref_inp = [to_reference(_) for _ in inp]
    ref_out = torch.cat(ref_inp, dim)

    with flag_gems.use_gems():
        res_out = torch.cat(inp, dim)
    gems_assert_equal(res_out, ref_out)


VSTACK_SHAPES = [
    [(3,), (3,)],
    [(3, 33), (7, 33)],
    [(13, 3, 333), (17, 3, 333), (7, 3, 333)],
    [
        (13, 3, 64, 5, 2),
        (16, 3, 64, 5, 2),
        (7, 3, 64, 5, 2),
        (4, 3, 64, 5, 2),
        (1, 3, 64, 5, 2),
    ],
]


@pytest.mark.vstack
@pytest.mark.parametrize("shape", VSTACK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_vstack(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device="cuda") for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cuda").to(
                dtype
            )
            for s in shape
        ]
    ref_inp = [to_reference(_) for _ in inp]
    ref_out = torch.vstack(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.vstack(inp)
    gems_assert_equal(res_out, ref_out)


REPEAT_INTERLEAVE_SHAPES = [
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]
REPEAT_INTERLEAVE_REPEATS = [2]
REPEAT_INTERLEAVE_DIM = [-1, 0, None]


@pytest.mark.repeat_interleave
@pytest.mark.parametrize("shape", REPEAT_INTERLEAVE_SHAPES + [(1,)])
@pytest.mark.parametrize("dim", REPEAT_INTERLEAVE_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_repeat_interleave_self_int(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    repeats = 2
    ref_inp = to_reference(inp)

    ref_out = torch.repeat_interleave(ref_inp, repeats, dim)
    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(ref_inp, repeats, dim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.repeat_interleave
@pytest.mark.parametrize("shape", REPEAT_INTERLEAVE_SHAPES)
@pytest.mark.parametrize("dim", REPEAT_INTERLEAVE_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_repeat_interleave_self_int_non_contiguous(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")[::2]
    repeats = 2
    ref_inp = to_reference(inp)

    ref_out = torch.repeat_interleave(ref_inp, repeats, dim)
    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(ref_inp, repeats, dim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.repeat_interleave
@pytest.mark.parametrize("shape", UT_SHAPES_1D)
@pytest.mark.parametrize("dtype", [torch.int32])
def test_accuracy_repeat_interleave_tensor(shape, dtype):
    repeats = torch.randint(0, 30, shape, dtype=dtype, device="cuda")
    ref_repeats = to_reference(repeats)
    ref_out = torch.repeat_interleave(ref_repeats)

    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(repeats)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.repeat_interleave
@pytest.mark.parametrize("shape", REPEAT_INTERLEAVE_SHAPES)
@pytest.mark.parametrize("dim", [-1, 0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_repeat_interleave_self_tensor(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    repeats = torch.randint(0, 30, (shape[dim],), device="cuda")
    ref_inp = to_reference(inp)
    ref_repeats = to_reference(repeats)

    ref_out = torch.repeat_interleave(ref_inp, ref_repeats, dim)
    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(inp, repeats, dim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.diag
@pytest.mark.parametrize("shape", UT_SHAPES_1D + UT_SHAPES_2D)
@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + BOOL_TYPES)
def test_accuracy_diag(shape, diagonal, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device="cuda")
    elif dtype in BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(0, 0x7FFF, size=shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.diag(ref_inp, diagonal)
    with flag_gems.use_gems():
        res_out = torch.diag(inp, diagonal)
    gems_assert_equal(res_out, ref_out)


def get_dim1_dim2(o_rank):
    dims = list(range(-o_rank, o_rank))
    return [
        p for p in itertools.permutations(dims, 2) if (p[0] % o_rank) != (p[1] % o_rank)
    ]


def get_diag_embed_shape_and_dims():
    shapes = [
        (1024,),
        (1024, 1024),
    ]
    # [(shape, dim1, dim2)]
    result = []

    for s in shapes:
        dim_pairs = get_dim1_dim2(len(s) + 1)
        result.extend([(s, dim1, dim2) for dim1, dim2 in dim_pairs])

    return result


@pytest.mark.diag_embed
@pytest.mark.parametrize("shape, dim1, dim2", get_diag_embed_shape_and_dims())
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + BOOL_TYPES)
def test_accuracy_diag_embed(shape, dtype, offset, dim1, dim2):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device="cuda")
    elif dtype in INT_DTYPES:
        inp = torch.randint(low=0, high=0x7FFF, size=shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(low=0, high=2, size=shape, dtype=dtype, device="cuda")

    ref_inp = to_reference(inp)

    ref_out = torch.diag_embed(ref_inp, offset, dim1, dim2)
    with flag_gems.use_gems():
        res_out = torch.diag_embed(inp, offset, dim1, dim2)
    gems_assert_equal(res_out, ref_out)


def get_diagonal_backward_shape_and_dims():
    shapes = SPECIAL_SHAPES
    result = []

    for s in shapes:
        dim_pairs = get_dim1_dim2(len(s))
        result.extend([(s, dim1, dim2) for dim1, dim2 in dim_pairs])

    return result


@pytest.mark.diagonal_backward
@pytest.mark.parametrize("shape, dim1, dim2", get_diagonal_backward_shape_and_dims())
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_diagonal_backward(shape, dtype, dim1, dim2, offset):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    ref_inp = to_reference(inp)

    ref_out = torch.diagonal(ref_inp, offset, dim1, dim2)
    res_out = torch.diagonal(inp, offset, dim1, dim2)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    res_out = to_reference(res_out)
    res_in_grad = to_reference(res_in_grad)
    gems_assert_equal(res_out, ref_out)
    gems_assert_equal(res_in_grad, ref_in_grad)


@pytest.mark.sort
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("hiddensize", [1, 256, 2048, 9333, 65536])
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
@pytest.mark.parametrize("dim", [0, -1])
def test_sort(batch_size, hiddensize, descending, dtype, dim):
    if dtype in FLOAT_DTYPES:
        x = torch.empty((hiddensize,), dtype=dtype, device="cuda")
        tmp = torch.tensor(0, dtype=dtype)
        inf = torch.tensor(float("inf"), dtype=dtype)
        for i in range(0, hiddensize):
            x[i] = tmp.item()
            tmp = torch.nextafter(tmp, inf)
            if tmp.item() == inf.item():
                hiddensize = i
                x = x[:hiddensize]
                break
    else:
        x = torch.arange(hiddensize, dtype=dtype, device="cuda")
    y = torch.empty((batch_size, hiddensize), dtype=dtype, device="cuda")

    # Each row use different shuffled index.
    col_indices = torch.randperm(x.size(0))
    for bsz in range(batch_size):
        col_indices = torch.randperm(x.size(0))
        y[bsz, :] = x[col_indices]
    if dim == 0:
        y = torch.movedim(y, dim, -1)
    ref_y = to_reference(y)
    ref_value, ref_index = torch.sort(ref_y, dim=dim, descending=descending)

    with flag_gems.use_gems():
        res_value, res_index = torch.sort(y, dim=dim, descending=descending)

    gems_assert_close(res_value, ref_value, dtype)
    gems_assert_equal(res_index, ref_index)
