# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent numerical and graph checks for the migrated common operators."""

import pytest
import torch
import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires SM90 FP8 support",
)


@triton.jit
def _unit_scale(x, dtype: tl.constexpr):
    return tl.full((), 1.0, tl.float32)


@triton.jit
def _store_unit_scale(value, dtype: tl.constexpr, a, b):
    return value, value


@triton.jit
def _identity_quant(x, dtype: tl.constexpr):
    return x


@triton.jit
def _silu(x, kind: tl.constexpr):
    return x.to(tl.float32) / (1.0 + tl.exp(-x.to(tl.float32)))


def test_fp4_unpack_and_query_quant_against_explicit_codebook():
    from flag_gems.fused.dsv41.fp4_prefill import (
        quantize_bf16_index_queries_fp8,
        unpack_fp4_index_keys_to_fp8,
    )

    page_size = 64
    table = torch.zeros((2, page_size * 68), dtype=torch.uint8, device="cuda")
    table[:, page_size * 64 :] = 127
    slot_values = [0, 63, 64]
    codebook = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    packed = torch.arange(64, device="cuda", dtype=torch.int32) % 16
    packed = (packed | ((15 - packed) << 4)).to(torch.uint8)
    for slot in slot_values:
        page, offset = divmod(slot, page_size)
        table[page, offset * 64 : (offset + 1) * 64] = packed
    slots = torch.tensor(slot_values, dtype=torch.int64, device="cuda")
    actual = unpack_fp4_index_keys_to_fp8(slots, table, page_size).float()
    codes = torch.arange(64, device="cuda") % 16
    expected = torch.empty(128, device="cuda")
    expected[0::2], expected[1::2] = codebook[codes], codebook[15 - codes]
    torch.testing.assert_close(actual, expected.repeat(3, 1), rtol=0, atol=0)
    q = expected.repeat(3, 2, 1).bfloat16()
    torch.testing.assert_close(
        quantize_bf16_index_queries_fp8(q).float(), q.float(), rtol=0, atol=0
    )


@pytest.mark.parametrize("heads", [8, 32])
def test_dense_score_changed_input_graph(heads):
    from flag_gems.fused.dsv41.fp4_prefill import dense_fp8_index_logits

    length = 131
    q = torch.randint(-2, 3, (3, heads, 128), device="cuda").to(torch.float8_e4m3fn)
    k = torch.randint(-2, 3, (length, 128), device="cuda").to(torch.float8_e4m3fn)
    w = torch.randint(-2, 3, (3, heads), device="cuda").bfloat16()
    lens = torch.tensor([0, length // 2, length], device="cuda")
    dense_fp8_index_logits(q, w, k, lens)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = dense_fp8_index_logits(q, w, k, lens)
    q.copy_((-q.float()).to(q.dtype))
    lens.copy_(torch.tensor([length, 17, 0], device="cuda"))
    result.fill_(123)
    graph.replay()
    dot = (q.float() @ k.float().T).bfloat16().float().clamp_min(0)
    expected = (dot * w.float()[..., None]).bfloat16().float().sum(1).bfloat16().float()
    expected.masked_fill_(
        torch.arange(length, device="cuda")[None, :] >= lens[:, None], -float("inf")
    )
    torch.testing.assert_close(result[:, :length], expected, rtol=0, atol=0)
    assert torch.isneginf(result[:, length:]).all()


@pytest.mark.parametrize("row_splits", [None, 16])
def test_masked_quant_writes_only_valid_rows(row_splits):
    from flag_gems.fused.dsv41.masked_row_quant import masked_row_quant

    x = torch.arange(2 * 4 * 8, device="cuda").reshape(2, 4, 8).remainder(8).bfloat16()
    out = torch.zeros((8, 8), device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.full((8, 1), -1.0, device="cuda")
    counts = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
    masked_row_quant(
        x,
        out,
        scales,
        counts,
        calc_scale=_unit_scale,
        finalize_scale=_store_unit_scale,
        quant_tensor=_identity_quant,
        row_splits=row_splits,
    )
    valid = torch.tensor([0, 4, 5, 6], device="cuda")
    invalid = torch.tensor([1, 2, 3, 7], device="cuda")
    torch.testing.assert_close(
        out[valid].float(), x.reshape(8, 8)[valid].float(), rtol=0, atol=0
    )
    assert torch.equal(scales[valid], torch.ones_like(scales[valid]))
    assert torch.equal(scales[invalid], torch.full_like(scales[invalid], -1))
    assert torch.equal(out[invalid].float(), torch.zeros_like(out[invalid].float()))


def test_masked_swiglu_preserves_padding_and_clipping():
    from flag_gems.fused.dsv41.masked_swiglu import masked_swiglu

    x = torch.full((2, 4, 8), 20.0, device="cuda", dtype=torch.bfloat16)
    x[:, :, 4:] = -20
    out = torch.full((8, 4), -7.0, device="cuda", dtype=torch.bfloat16)
    counts = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
    masked_swiglu(x, out, counts, activate=_silu)
    valid = torch.tensor([0, 4, 5, 6], device="cuda")
    invalid = torch.tensor([1, 2, 3, 7], device="cuda")
    expected = (
        (
            torch.nn.functional.silu(x[..., :4].float().clamp(max=10))
            * x[..., 4:].float().clamp(-10, 10)
        )
        .bfloat16()
        .reshape(8, 4)
    )
    torch.testing.assert_close(out[valid], expected[valid], rtol=0, atol=0)
    assert torch.equal(out[invalid], torch.full_like(out[invalid], -7))
