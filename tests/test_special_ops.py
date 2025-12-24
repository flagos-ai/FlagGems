import itertools
import random
import time
from typing import Optional

import numpy as np
import pytest
import torch

import flag_gems

from .accuracy_utils import (
    ALL_INT_DTYPES,
    ARANGE_START,
    BOOL_TYPES,
    FLOAT_DTYPES,
    FP8_QUANT_SHAPES,
    INT_DTYPES,
    KRON_SHAPES,
    SPECIAL_SHAPES,
    STACK_DIM_LIST,
    STACK_SHAPES,
    UPSAMPLE_SHAPES,
    UT_SHAPES_1D,
    UT_SHAPES_2D,
    gems_assert_close,
    gems_assert_equal,
    to_cpu,
    to_reference,
)
from .conftest import TO_CPU

# Make sure every thread has same seed.
random.seed(time.time() // 100)

device = flag_gems.device


@pytest.mark.dropout
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("p", [0.3, 0.6, 0.9])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_dropout(shape, p, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if TO_CPU or shape == (1,):
        shape = (32768,)
    res_inp = torch.randn(
        shape,
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = to_reference(res_inp)

    # NOTE: ensure that scalars are float32(instead of float64)
    # in some cases, casting up then casting down have different result
    p = np.float32(p)
    one_minus_p = np.float32(1.0) - p

    ref_out = torch.nn.functional.dropout(ref_inp, p, True)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.dropout(res_inp, p, True)

    res_out = to_reference(res_out)
    exp_equal = (p * p + one_minus_p * one_minus_p) * res_inp.numel()
    num_equal = torch.sum(torch.isclose(ref_out, res_out)).item()
    if TO_CPU:
        from flag_gems.testing import RESOLUTION

        zero_equal = torch.eq(res_out, torch.zeros_like(res_out))
        num_zero = torch.sum(zero_equal).item()
        assert abs(num_zero / res_inp.numel() - p) <= 0.05
        scale_equal = torch.isclose(
            res_out, ref_inp / one_minus_p, rtol=RESOLUTION[dtype]
        )
        assert torch.all(torch.logical_or(zero_equal, scale_equal))
    else:
        assert (
            abs(num_equal - exp_equal) / exp_equal <= 0.05
        ), f"num_equal: {num_equal}, exp_equal: {exp_equal}, num_total: {res_inp.numel()}"


@pytest.mark.dropout
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("p", [0.3, 0.6, 0.9])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_dropout_backward(shape, p, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_mask = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)
    ref_grad = to_reference(res_grad)
    ref_mask = to_reference(res_mask)

    scale = 1.0 / (1.0 - p)

    ref_in_grad = torch.ops.aten.native_dropout_backward(ref_grad, ref_mask, scale)
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.native_dropout_backward(res_grad, res_mask, scale)

    gems_assert_close(res_in_grad, ref_in_grad, dtype)


def get_rope_cos_sin(max_seq_len, dim, dtype, base=10000, device=flag_gems.device):
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
        (batch_size, seq_len, q_heads, head_dim), dtype=dtype, device=flag_gems.device
    )
    k = torch.randn(
        (batch_size, seq_len, k_heads, head_dim), dtype=dtype, device=flag_gems.device
    )

    position_ids = torch.randint(
        0, max_seq_len, (batch_size, seq_len), device=flag_gems.device
    )
    cos, sin = get_rope_cos_sin(max_seq_len, head_dim, dtype, device=flag_gems.device)

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
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_indices = torch.randint(
        0, EmbeddingSize, (Batch, M), device=flag_gems.device, requires_grad=False
    )
    res_embedding = torch.randn(
        (EmbeddingSize, N), device=flag_gems.device, dtype=dtype, requires_grad=True
    )
    ref_embedding = to_reference(res_embedding)
    ref_indices = to_reference(res_indices)

    ref_out = torch.nn.functional.embedding(
        ref_indices, ref_embedding, padding_idx, scale_grad_by_freq=scale_grad_by_freq
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.embedding(
            res_indices,
            res_embedding,
            padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
        )
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.embedding
@pytest.mark.parametrize("EmbeddingSize", [1024] if TO_CPU else [4096])
@pytest.mark.parametrize("Batch", [2] if TO_CPU else [2, 4])
@pytest.mark.parametrize("M", [4] if TO_CPU else [4, 8])
@pytest.mark.parametrize("N", [8] if TO_CPU else [128, 256, 4096])
@pytest.mark.parametrize("padding_idx", [-1, 1, 2])
@pytest.mark.parametrize("scale_grad_by_freq", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_embedding_backward(
    EmbeddingSize, Batch, M, N, padding_idx, scale_grad_by_freq, dtype
):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_grad = torch.randn((Batch, M, N), device=flag_gems.device, dtype=dtype)
    res_indices = torch.randint(0, EmbeddingSize, (Batch, M), device=flag_gems.device)
    num_weights = EmbeddingSize
    sparse = False

    ref_grad = to_reference(res_grad)
    ref_indices = to_reference(res_indices)

    ref_in_grad = torch.ops.aten.embedding_backward(
        ref_grad, ref_indices, num_weights, padding_idx, scale_grad_by_freq, sparse
    )
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.embedding_backward(
            res_grad, res_indices, num_weights, padding_idx, scale_grad_by_freq, sparse
        )

    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.resolve_neg
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.cfloat])
def test_accuracy_resolve_neg(shape, dtype):
    if flag_gems.vendor_name == "ascend":
        x = torch.randn(size=shape, dtype=dtype).to(device=flag_gems.device)
    else:
        x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
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
    x = torch.arange(hiddensize, dtype=dtype, device=flag_gems.device)
    x = x.repeat(batch_size).reshape(batch_size, hiddensize)

    # Each row use different shuffled index.
    for bsz in range(batch_size):
        col_indices = torch.randperm(x.size(1))
        x[bsz, :] = x[bsz, col_indices]
    ref_x = to_reference(x)

    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float16:
        ref_x = ref_x.cuda()

    ref_value, ref_index = torch.topk(ref_x, topk, largest=largest)

    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float16:
        if TO_CPU:
            ref_value = ref_value.cpu()
            ref_index = ref_index.cpu()

    with flag_gems.use_gems():
        res_value, res_index = torch.topk(x, topk, largest=largest)

    gems_assert_close(res_value, ref_value, dtype)
    gems_assert_equal(res_index, ref_index)


@pytest.mark.resolve_conj
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.cfloat])
def test_accuracy_resolve_conj(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cpu")
    y = x.conj()
    assert y.is_conj()
    with flag_gems.use_gems():
        res_y = y.to(device=flag_gems.device)
        z = res_y.resolve_conj()
    assert not z.is_conj()


# @pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="AssertionError")
@pytest.mark.unique
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("sorted", [True])
@pytest.mark.parametrize("return_inverse", [True, False])
@pytest.mark.parametrize("return_counts", [False, True])
def test_accuracy_unique(shape, dtype, sorted, return_inverse, return_counts):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10, 10, shape, device=flag_gems.device).to(dtype)
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
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    if shape[-1] == 1:
        dist = torch.rand(size=shape, dtype=dtype, device=flag_gems.device)
        with flag_gems.use_gems():
            res_out = torch.multinomial(dist, n_samples, True)
        assert torch.all(res_out == 0)
    else:
        # Mask p% off of the categories and test the sampling results fall in the rest
        for p in (0.1, 0.5, 0.9):
            dist = torch.rand(size=shape, dtype=dtype, device=flag_gems.device)
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
    dist = torch.rand(size=pool, dtype=dtype, device=flag_gems.device)
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
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("pad_mode", ["constant", "reflect", "replicate", "circular"])
@pytest.mark.parametrize("contiguous", [True, False])
def test_pad(shape, dtype, pad_mode, contiguous):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    if not contiguous:
        if flag_gems.vendor_name == "kunlunxin":
            x = x.cpu()[::2, ::2].to(flag_gems.device)
        else:
            x = x[::2, ::2]

    ref_x = to_reference(x)
    if ref_x.dtype == torch.float16:
        ref_x = ref_x.to(torch.float32)

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

    if ref_out.dtype != res_out.dtype:
        ref_out = ref_out.to(res_out.dtype)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(flag_gems.vendor_name == "cambricon", reason="fix")
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
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_upsample_bicubic2d_aa(dtype, shape, scale, align_corners):
    input = torch.rand(shape, dtype=dtype, device=flag_gems.device)
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

    if ref_out.dtype != res_out.dtype:
        ref_out = ref_out.to(res_out.dtype)

    reduce_dim = span(scale[0]) * span(scale[1])
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.5)])
@pytest.mark.parametrize("shape", UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_upsample_nearest2d(dtype, shape, scale):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_i = to_reference(input).to(torch.float32)
    output_size = [int(input.shape[i + 2] * scale[i]) for i in range(2)]
    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.arange
@pytest.mark.parametrize("start", ARANGE_START)
@pytest.mark.parametrize("step", [1, 2, 5])
@pytest.mark.parametrize("end", [128, 256, 1024])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES + [None])
@pytest.mark.parametrize("device", [device, None])
@pytest.mark.parametrize(
    "pin_memory", [False, None]
)  # Since triton only target to GPU, pin_memory only used in CPU tensors.
def test_arange(start, step, end, dtype, device, pin_memory):
    ref_out = torch.arange(
        start,
        end,
        step,
        dtype=dtype,
        device="cpu" if TO_CPU else device,
        pin_memory=pin_memory,
    )
    with flag_gems.use_gems():
        res_out = torch.arange(
            start, end, step, dtype=dtype, device=device, pin_memory=pin_memory
        )

    gems_assert_equal(res_out, ref_out)


@pytest.mark.linspace
@pytest.mark.parametrize("start", [0, 2, 4])
@pytest.mark.parametrize("end", [256, 2048, 4096])
@pytest.mark.parametrize("steps", [1, 256, 512])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES + [None])
@pytest.mark.parametrize("device", [device, None])
@pytest.mark.parametrize("pin_memory", [False, None])
def test_linspace(start, end, steps, dtype, device, pin_memory):
    ref_out = torch.linspace(
        start,
        end,
        steps,
        dtype=dtype,
        layout=None,
        device="cpu" if TO_CPU else device,
        pin_memory=pin_memory,
    )
    with flag_gems.use_gems():
        res_out = torch.linspace(
            start,
            end,
            steps,
            dtype=dtype,
            layout=None,
            device=device,
            pin_memory=pin_memory,
        )
    if dtype in [torch.float16, torch.bfloat16, torch.float32, None]:
        gems_assert_close(res_out, ref_out, dtype=dtype)
    else:
        gems_assert_equal(res_out, ref_out)


@pytest.mark.logspace
@pytest.mark.parametrize("start", [0, 2, 4])
@pytest.mark.parametrize("end", [32, 40])
@pytest.mark.parametrize("steps", [0, 1, 8, 17])
@pytest.mark.parametrize("base", [1.2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES + [None])
@pytest.mark.parametrize("device", [device])
@pytest.mark.parametrize("pin_memory", [False])
def test_logspace(start, end, steps, base, dtype, device, pin_memory):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and dtype is torch.half
        and torch.__version__ < "2.5"
    ):
        pytest.skip("wait lerp cpu half impl")

    ref_out = torch.logspace(
        start,
        end,
        steps,
        base,
        dtype=dtype,
        layout=None,
        device="cpu",
        pin_memory=pin_memory,
    ).to(
        "cpu" if TO_CPU else device
    )  # compute on cpu and move back to device
    with flag_gems.use_gems():
        res_out = torch.logspace(
            start,
            end,
            steps,
            base,
            dtype=dtype,
            layout=None,
            device=device,
            pin_memory=pin_memory,
        )
    if dtype in [torch.float16, torch.bfloat16, torch.float32, None]:
        gems_assert_close(res_out, ref_out, dtype=dtype)
    else:
        gems_assert_equal(res_out, ref_out)


# @pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RESULT TODOFIX")
@pytest.mark.isin
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("assume_unique", [False, True])
@pytest.mark.parametrize("invert", [False, True])
def test_accuracy_isin(shape, dtype, assume_unique, invert):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp1 = torch.randint(-100, 100, shape, device=flag_gems.device).to(dtype)
    test_numel = inp1.numel() // 2 if inp1.numel() > 1 else 1
    test_shape = (test_numel,)
    inp2 = torch.randint(-10, 10, test_shape, device=flag_gems.device).to(dtype)
    inp1.ravel()[-1] = 0
    if assume_unique:
        inp1 = torch.unique(inp1.cpu()).to(device)
        inp2 = torch.unique(inp2.cpu()).to(device)
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

    inp0 = torch.tensor([], device=flag_gems.device)
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
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)
    ref_x = to_reference(x, False)

    ref_out = torch.fill(ref_x, value)
    with flag_gems.use_gems():
        res_out = torch.fill(x, value)

    gems_assert_equal(res_out, ref_out)

    # Test fill.Tensor
    value_tensor = torch.tensor(value, device=flag_gems.device, dtype=dtype)
    ref_value_tensor = to_reference(value_tensor, False)
    ref_out_tensor = torch.fill(ref_x, ref_value_tensor)
    with flag_gems.use_gems():
        res_out_tensor = torch.fill(x, value_tensor)

    gems_assert_equal(res_out_tensor, ref_out_tensor)


CAMBRICON_STACK_SHAPES = [
    [
        (8, 8, 128),
        (8, 8, 128),
        (8, 8, 128),
    ],
    [
        (32, 64, 128, 8),
        (32, 64, 128, 8),
        (32, 64, 128, 8),
        (32, 64, 128, 8),
    ],
]
STACK_SHAPES_TEST = STACK_SHAPES + (
    CAMBRICON_STACK_SHAPES if flag_gems.vendor_name == "cambricon" else []
)


@pytest.mark.stack
@pytest.mark.parametrize("shape", STACK_SHAPES_TEST)
@pytest.mark.parametrize("dim", STACK_DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_stack(shape, dim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
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
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
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
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
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
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
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
    inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
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

CAMBRICON_VSTACK_SHAPES = [
    [(16, 128, 64, 64), (16, 128, 64, 64), (16, 128, 64, 64), (16, 128, 64, 64)],
    [
        (32, 64, 128, 8),
        (32, 64, 128, 8),
        (32, 64, 128, 8),
        (32, 64, 128, 8),
        (32, 64, 128, 8),
    ],
]
VSTACK_SHAPES_TEST = VSTACK_SHAPES + (
    CAMBRICON_VSTACK_SHAPES if flag_gems.vendor_name == "cambricon" else []
)


@pytest.mark.vstack
@pytest.mark.parametrize("shape", VSTACK_SHAPES_TEST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_vstack(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
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
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    repeats = 2
    ref_inp = to_reference(inp)

    ref_out = torch.repeat_interleave(ref_inp, repeats, dim)
    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(inp, repeats, dim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.repeat_interleave
@pytest.mark.parametrize("shape", REPEAT_INTERLEAVE_SHAPES)
@pytest.mark.parametrize("dim", REPEAT_INTERLEAVE_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_repeat_interleave_self_int_non_contiguous(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)[::2]
    repeats = 2
    ref_inp = to_reference(inp)

    ref_out = torch.repeat_interleave(ref_inp, repeats, dim)
    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(inp, repeats, dim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.repeat_interleave
@pytest.mark.parametrize("shape", UT_SHAPES_1D)
@pytest.mark.parametrize("dtype", [torch.int32])
def test_accuracy_repeat_interleave_tensor(shape, dtype):
    repeats = torch.randint(0, 30, shape, dtype=dtype, device=flag_gems.device)
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
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    repeats = torch.randint(0, 30, (shape[dim],), device=flag_gems.device)
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
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype in BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        inp = torch.randint(0, 0x7FFF, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
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
        if dim_pairs:
            dim1, dim2 = random.choice(dim_pairs)
            result.append((s, dim1, dim2))

    return result


@pytest.mark.diag_embed
@pytest.mark.parametrize("shape, dim1, dim2", get_diag_embed_shape_and_dims())
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + BOOL_TYPES)
def test_accuracy_diag_embed(shape, dtype, offset, dim1, dim2):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype in INT_DTYPES:
        inp = torch.randint(
            low=0, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    else:
        inp = torch.randint(low=0, high=2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )

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
        if dim_pairs:
            dim1, dim2 = random.choice(dim_pairs)
            result.append((s, dim1, dim2))

    return result


@pytest.mark.skipif(flag_gems.device == "kunlunxin", reason="tmp skip")
@pytest.mark.diagonal
@pytest.mark.parametrize("shape, dim1, dim2", get_diagonal_backward_shape_and_dims())
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_diagonal_backward(shape, dtype, dim1, dim2, offset):
    if flag_gems.vendor_name == "mthreads":
        torch.manual_seed(123)
        torch.musa.manual_seed_all(123)

    torch.empty(1, device=flag_gems.device, requires_grad=True).backward()
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp)

    ref_out = torch.diagonal(ref_inp, offset, dim1, dim2)
    with flag_gems.use_gems():
        res_out = torch.diagonal(inp, offset, dim1, dim2)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_equal(res_out, ref_out)
    gems_assert_equal(res_in_grad, ref_in_grad)


# @pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RESULT TODOFIX")
@pytest.mark.sort
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize(
    "hiddensize", [1, 256, 2048, 9333, 65536, 32768, 128 * 1024, 256 * 1024]
)
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
@pytest.mark.parametrize("dim", [0, -1])
def test_sort(batch_size, hiddensize, descending, dtype, dim):
    if dtype in BOOL_TYPES:
        y = torch.randint(
            0, 2, (batch_size, hiddensize), dtype=dtype, device=flag_gems.device
        )
    elif dtype in ALL_INT_DTYPES:
        min_v, max_v = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        y = torch.randint(
            min_v, max_v, (batch_size, hiddensize), dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    else:
        y = torch.randn((batch_size, hiddensize), dtype=dtype, device=flag_gems.device)

    ref_y = to_reference(y)
    # we only implement stable sort, non-stable sort is undefined
    ref_value, ref_index = torch.sort(
        ref_y, dim=dim, stable=True, descending=descending
    )

    with flag_gems.use_gems():
        res_value, res_index = torch.sort(
            y, dim=dim, stable=True, descending=descending
        )

    gems_assert_close(res_value, ref_value, dtype)
    gems_assert_equal(res_index, ref_index)


@pytest.mark.kron
@pytest.mark.parametrize("shape", KRON_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + BOOL_TYPES)
def test_accuracy_kron(shape, dtype):
    if dtype in INT_DTYPES:
        inp1 = torch.randint(
            low=-10, high=10, size=shape[0], dtype=dtype, device="cpu"
        ).to(flag_gems.device)
        inp2 = torch.randint(
            low=-10, high=10, size=shape[1], dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    elif dtype in FLOAT_DTYPES:
        inp1 = torch.randn(shape[0], dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape[1], dtype=dtype, device=flag_gems.device)
    else:
        inp1 = torch.randint(0, 2, size=shape[0], dtype=dtype, device=flag_gems.device)
        inp2 = torch.randint(0, 2, size=shape[1], dtype=dtype, device=flag_gems.device)

    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.bfloat16:
        # Pytorch 2.0.1 Bfloat16 CPU Backend Precision Failed
        inp1 = torch.randn(shape[0], dtype=torch.float32, device=flag_gems.device)
        inp2 = torch.randn(shape[1], dtype=torch.float32, device=flag_gems.device)

    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.kron(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.kron(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.contiguous
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_contiguous(shape, dtype):
    if shape[0] <= 2:
        return
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(
            low=-10000, high=10000, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)

    inp = inp[::2]
    assert inp.is_contiguous() is False

    ref_inp = to_reference(inp)
    ref_out = ref_inp.contiguous()
    with flag_gems.use_gems():
        res_out = inp.contiguous()

    assert res_out.is_contiguous() is True
    assert res_out.is_contiguous() is True
    assert res_out.stride() == ref_out.stride()
    gems_assert_equal(res_out, ref_out)


def native_per_token_group_quant_fp8(x, group_size, eps=1e-10, dtype=None):
    if dtype is None:
        dtype = flag_gems.SUPPORTED_FP8_DTYPE

    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / fp8_max
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))

    return x_q, x_s


@pytest.mark.per_token_group_quant_fp8
@pytest.mark.parametrize("seed", FP8_QUANT_SHAPES["SEEDS"])
@pytest.mark.parametrize("group_size", FP8_QUANT_SHAPES["GROUP_SIZE"])
@pytest.mark.parametrize("dtype", FP8_QUANT_SHAPES["DTYPES"])
@pytest.mark.parametrize("d", FP8_QUANT_SHAPES["D"])
@pytest.mark.parametrize("num_tokens", FP8_QUANT_SHAPES["NUM_TOKENS"])
def test_accuracy_per_token_group_quant_fp8(num_tokens, d, dtype, group_size, seed):
    torch.manual_seed(seed)
    x = torch.rand(num_tokens, d, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    ref_out, ref_scale = native_per_token_group_quant_fp8(ref_x, group_size)
    with flag_gems.use_gems():
        out, scale = flag_gems.per_token_group_quant_fp8(x, group_size)

    gems_assert_close(scale, ref_scale, dtype=torch.float32)

    out_fp32 = to_cpu(out, ref_out).to(torch.float32)
    ref_out_fp32 = ref_out.to(torch.float32)
    assert torch.allclose(out_fp32, ref_out_fp32, rtol=0.15)


@pytest.mark.rwkv_ka_fusion
@pytest.mark.parametrize("T", [2**d for d in range(4, 15, 2)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rwkv_kafusion(T, dtype):
    H = 8
    N = 64
    C = H * N
    k = torch.rand(T, C, dtype=dtype, device=flag_gems.device)
    kk = torch.rand(C, dtype=dtype, device=flag_gems.device)
    a = torch.rand(T, C, dtype=dtype, device=flag_gems.device)
    ka = torch.rand(C, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        o_k, o_kk, o_kka = flag_gems.rwkv_ka_fusion(k, kk, a, ka, H, N)

    ref_k = to_reference(k, True)
    ref_kk = to_reference(kk, True)
    ref_a = to_reference(a, True)
    ref_ka = to_reference(ka, True)

    ref_o_kk = torch.nn.functional.normalize(
        (ref_k * ref_kk).view(T, H, N), dim=-1, p=2.0
    ).view(T, H * N)
    ref_o_k = ref_k * (1 + (ref_a - 1) * ref_ka)
    ref_o_kka = ref_o_kk * ref_a

    gems_assert_close(o_k, ref_o_k, dtype, equal_nan=True)
    gems_assert_close(o_kk, ref_o_kk, dtype, equal_nan=True)
    gems_assert_close(o_kka, ref_o_kka, dtype, equal_nan=True)


@pytest.mark.rwkv_mm_sparsity
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rwkv_mmsparsity(dtype):
    n = 16384
    embedding_dim = 4096

    k = torch.randn(n, dtype=dtype, device=flag_gems.device)
    k = torch.relu(k)
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(42)
        # kunlunxin sparsity test require 90% sparsity
        sparsity_levels = [0.9]
        for target_sparsity in sparsity_levels:
            threshold = torch.quantile(k.abs().to(torch.float32), target_sparsity).to(
                dtype
            )
            k = torch.relu(k - threshold)

    V_ = torch.randn(n, embedding_dim, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res = flag_gems.rwkv_mm_sparsity(k, V_)

    ref_k = to_reference(k, True)
    ref_V_ = to_reference(V_, True)
    ref_res = ref_k @ ref_V_

    gems_assert_close(res, ref_res, dtype, equal_nan=True)


M_VALUES = [1, 33, 64, 222]
TOP_KS = [2, 6]
K_VALUES = [128, 511, 1024]
MOE_SHAPES = list(itertools.product(M_VALUES, TOP_KS, K_VALUES))


@pytest.mark.moe_sum
@pytest.mark.parametrize("shape", MOE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_moe_sum(shape, dtype):
    m, topk, k = shape
    inp1 = torch.randn((m, topk, k), dtype=dtype, device=flag_gems.device)
    res_out = torch.empty((m, k), dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_out = torch.sum(ref_inp1, dim=1)
    with flag_gems.use_gems():
        flag_gems.moe_sum(inp1, res_out)
    gems_assert_close(res_out, ref_out, dtype)
