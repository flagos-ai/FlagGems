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

"""Correctness tests for aten::_scaled_dot_product_flash_attention_for_cpu.

The kernel has a CPU-only implementation, so the native reference and the
injected candidate both receive CPU tensors; _cpu_input repeats the shared
tu.make_input value grid and localizes it to the CPU device.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu


def _cpu_input(dtype, shape, value_range):
    """Shared value-range grid localized to the CPU-only kernel."""
    return tu.make_input(dtype, shape, value_range).cpu()


def _qkv(dtype, shape, value_range):
    """Three independent query/key/value tensors drawn from one value range."""
    return tuple(_cpu_input(dtype, shape, value_range) for _ in range(3))


def _head_transposed(dtype, shape, value_range):
    """(B, H, S, E) operand stored as (B, S, H, E) and transposed into place."""
    batch, heads, seq, head_dim = shape
    return _cpu_input(dtype, (batch, seq, heads, head_dim), value_range).transpose(1, 2)


# Probe: the kernel accepts FP32/FP16/BF16/FP64 only; int8, uint8, int32, int64,
# bool and both FP8 types raise RuntimeError (see the negative dtype test).
ATTENTION_DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.float64]

# The kernel accepts 4-D (B, H, S, E) tensors only, so the spec's shape set is
# re-expressed in that layout; quick adapts (2, 19, 7) by inserting a head dim.
ATTENTION_SHAPES = tu.selected_cases(
    [
        (2, 3, 19, 7),
        (1, 1, 1024, 64),
        (2, 4, 320, 15),
        (16, 128, 64, 60),
        (16, 7, 57, 32),
    ],
    quick=[(2, 3, 19, 7)],
)

# B == 0 and a singleton head dimension are valid. Zero-length query/key
# sequences are excluded: the native CPU kernel divides by the sequence length
# and kills the process with SIGFPE instead of raising, so no case can observe
# it, and it cannot serve as a negative test either.
BOUNDARY_SHAPES = tu.selected_cases(
    [(0, 3, 5, 8), (1, 1, 4, 1)], quick=[(0, 3, 5, 8), (1, 1, 4, 1)]
)

# Small cross-attention rows: the kernel supports Sq != Sk in both directions.
CROSS_SHAPES = tu.selected_cases(
    [
        ((2, 3, 5, 8), (2, 3, 7, 8)),
        ((2, 3, 7, 8), (2, 3, 5, 8)),
        ((2, 4, 16, 32), (2, 4, 129, 32)),
    ],
    quick=[((2, 3, 5, 8), (2, 3, 7, 8)), ((2, 3, 7, 8), (2, 3, 5, 8))],
)

# A head-transposed operand is non-contiguous. Probe: the native kernel accepts
# it and matches the contiguous operand bit for bit (maxdiff 0.0) for all four
# dtypes, so a non-contiguous input can be compared against its own reference.
STRIDED_SHAPES = tu.selected_cases(
    [(2, 3, 7, 8), (2, 8, 32, 15), (1, 4, 16, 64)], quick=[(2, 3, 7, 8)]
)

MASK_QUERY_SHAPE = (2, 3, 5, 8)

# A 2-D (Sq, Sk) mask broadcasts over batch and heads; a 4-D mask may broadcast
# over either leading dimension.
MASK_SHAPES = tu.selected_cases(
    [(5, 5), (2, 3, 5, 5), (2, 1, 5, 5), (1, 3, 5, 5)],
    quick=[(5, 5), (2, 3, 5, 5), (2, 1, 5, 5), (1, 3, 5, 5)],
)

# The kernel accepts float32 masks with FP16, BF16 and FP64 queries;
# equal-dtype masks are covered above.
MIXED_MASK_CASES = tu.selected_cases(
    [
        (dtype, torch.float32)
        for dtype in (torch.float16, torch.bfloat16, torch.float64)
    ],
    quick=[
        (dtype, torch.float32)
        for dtype in (torch.float16, torch.bfloat16, torch.float64)
    ],
)

# (q_shape, kv_shape) pairs; causal masking supports Sq != Sk by aligning the
# query/key positions according to the native causal rule.
CAUSAL_CASES = tu.selected_cases(
    [
        ((2, 3, 7, 8), (2, 3, 7, 8)),
        ((2, 3, 7, 8), (2, 3, 129, 8)),
        ((2, 3, 7, 8), (2, 3, 5, 8)),
    ],
    quick=[
        ((2, 3, 7, 8), (2, 3, 7, 8)),
        ((2, 3, 7, 8), (2, 3, 129, 8)),
        ((2, 3, 7, 8), (2, 3, 5, 8)),
    ],
)

# float scale: positive, negative, zero plus the inf and nan boundaries; the
# schema default (scale omitted) is covered by the grid and causal tests.
SCALES = tu.selected_cases(
    [0.5, -0.5, 0.0, float("inf"), float("nan")], quick=[0.5, -0.5, 0.0]
)

# Grouped-query attention (fewer key/value heads than query heads) is NOT a
# supported CPU-kernel form and is deliberately not tested: two identical native
# calls with the same inputs disagree, and comparing against the same call with
# k/v repeated to the query head count differs by ~1.7e38, i.e. the head
# broadcast reads outside the key/value head storage. Its output is undefined,
# so it cannot serve as an oracle, and its backward path aborts the process.

SPECIAL_SHAPE = (1, 1, 5, 8)
SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(ATTENTION_DTYPES), quick=[])

# Gradient support is probe-verified for all four supported dtypes in both the
# square and the causal rectangular form.
BACKWARD_CASES = tu.selected_cases(
    [
        ((2, 3, 5, 8), (2, 3, 5, 8), False, torch.float32),
        ((2, 3, 7, 8), (2, 3, 129, 8), True, torch.float32),
        ((2, 3, 5, 8), (2, 3, 5, 8), False, torch.float16),
        ((2, 3, 7, 8), (2, 3, 129, 8), True, torch.float16),
        ((2, 3, 5, 8), (2, 3, 5, 8), False, torch.bfloat16),
        ((2, 3, 7, 8), (2, 3, 129, 8), True, torch.bfloat16),
        ((2, 3, 5, 8), (2, 3, 5, 8), False, torch.float64),
        ((2, 3, 7, 8), (2, 3, 129, 8), True, torch.float64),
    ],
    quick=[],
)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", ATTENTION_DTYPES)
def test__scaled_dot_product_flash_attention_for_cpu(shape, value_range, dtype):
    q, k, v = _qkv(dtype, shape, value_range)
    ref_q, ref_k, ref_v = (tu.to_reference(t) for t in (q, k, v))

    ref_out, ref_lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        ref_q, ref_k, ref_v
    )
    res_out, res_lse = flag_gems._scaled_dot_product_flash_attention_for_cpu(q, k, v)

    # The [0, max] and [min, 0] ranges overflow the softmax for FP32/BF16/FP64 and
    # the native result is NaN; equal_nan in the shared assertion matches it.
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(res_lse, ref_lse)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
@pytest.mark.parametrize("shape", BOUNDARY_SHAPES)
@pytest.mark.parametrize("dtype", ATTENTION_DTYPES)
def test__scaled_dot_product_flash_attention_for_cpu_boundary_shapes(shape, dtype):
    q, k, v = _qkv(dtype, shape, ["-1", "1"])
    ref_q, ref_k, ref_v = (tu.to_reference(t) for t in (q, k, v))

    ref_out, ref_lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        ref_q, ref_k, ref_v
    )
    res_out, res_lse = flag_gems._scaled_dot_product_flash_attention_for_cpu(q, k, v)

    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(res_lse, ref_lse)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
@pytest.mark.parametrize("q_shape,kv_shape", CROSS_SHAPES)
@pytest.mark.parametrize("dtype", ATTENTION_DTYPES)
def test__scaled_dot_product_flash_attention_for_cpu_cross_attention(
    q_shape, kv_shape, dtype
):
    q = _cpu_input(dtype, q_shape, ["-1", "1"])
    k = _cpu_input(dtype, kv_shape, ["-1", "1"])
    v = _cpu_input(dtype, kv_shape, ["-1", "1"])
    ref_q, ref_k, ref_v = (tu.to_reference(t) for t in (q, k, v))

    ref_out, ref_lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        ref_q, ref_k, ref_v
    )
    res_out, res_lse = flag_gems._scaled_dot_product_flash_attention_for_cpu(q, k, v)

    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(res_lse, ref_lse)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
@pytest.mark.parametrize("shape", STRIDED_SHAPES)
@pytest.mark.parametrize("dtype", ATTENTION_DTYPES)
def test__scaled_dot_product_flash_attention_for_cpu_non_contiguous(shape, dtype):
    q, k, v = (_head_transposed(dtype, shape, ["-1", "1"]) for _ in range(3))
    # tu.to_reference keeps the view metadata, so the reference operand is an
    # independent storage with the same non-contiguous layout.
    ref_q, ref_k, ref_v = (tu.to_reference(t) for t in (q, k, v))

    ref_out, ref_lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        ref_q, ref_k, ref_v
    )
    res_out, res_lse = flag_gems._scaled_dot_product_flash_attention_for_cpu(q, k, v)

    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(res_lse, ref_lse)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
@pytest.mark.parametrize("mask_shape", MASK_SHAPES)
@pytest.mark.parametrize("dtype", ATTENTION_DTYPES)
def test__scaled_dot_product_flash_attention_for_cpu_with_mask(mask_shape, dtype):
    q, k, v = _qkv(dtype, MASK_QUERY_SHAPE, ["-1", "1"])
    ref_q, ref_k, ref_v = (tu.to_reference(t) for t in (q, k, v))
    mask = _cpu_input(dtype, mask_shape, ["-1", "1"])
    ref_mask = tu.to_reference(mask)

    ref_out, ref_lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        ref_q, ref_k, ref_v, attn_mask=ref_mask
    )
    res_out, res_lse = flag_gems._scaled_dot_product_flash_attention_for_cpu(
        q, k, v, attn_mask=mask
    )

    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(res_lse, ref_lse)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
@pytest.mark.parametrize("dtype,mask_dtype", MIXED_MASK_CASES)
def test__scaled_dot_product_flash_attention_for_cpu_float32_mask(dtype, mask_dtype):
    q, k, v = _qkv(dtype, MASK_QUERY_SHAPE, ["-1", "1"])
    ref_q, ref_k, ref_v = (tu.to_reference(t) for t in (q, k, v))
    mask = _cpu_input(mask_dtype, (5, 5), ["-1", "1"])
    ref_mask = tu.to_reference(mask)

    ref_out, ref_lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        ref_q, ref_k, ref_v, attn_mask=ref_mask
    )
    res_out, res_lse = flag_gems._scaled_dot_product_flash_attention_for_cpu(
        q, k, v, attn_mask=mask
    )

    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(res_lse, ref_lse)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
@pytest.mark.parametrize("q_shape,kv_shape", CAUSAL_CASES)
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dtype", ATTENTION_DTYPES)
def test__scaled_dot_product_flash_attention_for_cpu_causal(
    q_shape, kv_shape, is_causal, dtype
):
    q = _cpu_input(dtype, q_shape, ["-1", "1"])
    k = _cpu_input(dtype, kv_shape, ["-1", "1"])
    v = _cpu_input(dtype, kv_shape, ["-1", "1"])
    ref_q, ref_k, ref_v = (tu.to_reference(t) for t in (q, k, v))

    # dropout_p and is_causal are passed positionally, as in the ATen schema.
    ref_out, ref_lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        ref_q, ref_k, ref_v, 0.0, is_causal
    )
    res_out, res_lse = flag_gems._scaled_dot_product_flash_attention_for_cpu(
        q, k, v, 0.0, is_causal
    )

    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(res_lse, ref_lse)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("dtype", ATTENTION_DTYPES)
def test__scaled_dot_product_flash_attention_for_cpu_scale(scale, dtype):
    q, k, v = _qkv(dtype, MASK_QUERY_SHAPE, ["-1", "1"])
    ref_q, ref_k, ref_v = (tu.to_reference(t) for t in (q, k, v))

    ref_out, ref_lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        ref_q, ref_k, ref_v, 0.0, False, scale=scale
    )
    res_out, res_lse = flag_gems._scaled_dot_product_flash_attention_for_cpu(
        q, k, v, 0.0, False, scale=scale
    )

    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(res_lse, ref_lse)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
@pytest.mark.parametrize("dtype", ATTENTION_DTYPES)
def test__scaled_dot_product_flash_attention_for_cpu_dropout_zero(dtype):
    # dropout_p=0.0 is the only value the CPU kernel accepts; the schema default
    # (argument omitted) is covered by the grid and causal tests.
    q, k, v = _qkv(dtype, MASK_QUERY_SHAPE, ["-1", "1"])
    ref_q, ref_k, ref_v = (tu.to_reference(t) for t in (q, k, v))

    ref_out, ref_lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        ref_q, ref_k, ref_v, 0.0, False
    )
    res_out, res_lse = flag_gems._scaled_dot_product_flash_attention_for_cpu(
        q, k, v, 0.0, False
    )

    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(res_lse, ref_lse)


def _special_input(dtype, scenario, shape):
    """Symmetric grid with the shared special-value payload in the first row.

    The payload occupies the head dimension of the first sequence position, so
    NaN and Inf reach the softmax through every operand.
    """
    inp = _cpu_input(dtype, shape, ["-1", "1"])
    payload = tu.make_special_input(dtype, scenario).cpu()
    n = payload.numel()
    inp[0, 0, 0, :n] = payload
    return inp


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
@pytest.mark.parametrize("slot", ["q", "k", "v", "all"])
def test__scaled_dot_product_flash_attention_for_cpu_special_values(
    dtype, scenario, slot
):
    q, k, v = (
        (
            _special_input(dtype, scenario, SPECIAL_SHAPE)
            if slot in (name, "all")
            else _cpu_input(dtype, SPECIAL_SHAPE, ["-1", "1"])
        )
        for name in ("q", "k", "v")
    )
    ref_q, ref_k, ref_v = (tu.to_reference(t) for t in (q, k, v))

    ref_out, ref_lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        ref_q, ref_k, ref_v
    )
    res_out, res_lse = flag_gems._scaled_dot_product_flash_attention_for_cpu(q, k, v)

    # Matching NaNs are part of the native semantics (equal_nan in tu).
    tu.assert_result_close(res_out, ref_out)
    tu.assert_result_close(res_lse, ref_lse)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
@pytest.mark.parametrize("q_shape,kv_shape,is_causal,dtype", BACKWARD_CASES)
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("scale", [None, 0.5])
def test__scaled_dot_product_flash_attention_for_cpu_backward(
    q_shape, kv_shape, is_causal, dtype, use_mask, scale
):
    q = _cpu_input(dtype, q_shape, ["-1", "1"]).requires_grad_()
    k = _cpu_input(dtype, kv_shape, ["-1", "1"]).requires_grad_()
    v = _cpu_input(dtype, kv_shape, ["-1", "1"]).requires_grad_()
    ref_q = q.detach().clone().requires_grad_()
    ref_k = k.detach().clone().requires_grad_()
    ref_v = v.detach().clone().requires_grad_()

    mask = (
        _cpu_input(dtype, (q_shape[-2], kv_shape[-2]), ["-1", "1"])
        if use_mask
        else None
    )
    ref_mask = tu.to_reference(mask)
    res_out, _ = flag_gems._scaled_dot_product_flash_attention_for_cpu(
        q, k, v, 0.0, is_causal, attn_mask=mask, scale=scale
    )
    ref_out, _ = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        ref_q, ref_k, ref_v, 0.0, is_causal, attn_mask=ref_mask, scale=scale
    )

    tu.assert_result_close(res_out, ref_out)

    # Each output is differentiated with respect to its own query, key and value
    # leaves. The upstream gradient is non-uniform so that every output position
    # contributes to all three gradients.
    upstream = _cpu_input(dtype, tuple(ref_out.shape), ["-1", "1"])
    res_grads = torch.autograd.grad(res_out, (q, k, v), grad_outputs=upstream)
    ref_grads = torch.autograd.grad(
        ref_out, (ref_q, ref_k, ref_v), grad_outputs=upstream
    )

    for res_grad, ref_grad in zip(res_grads, ref_grads):
        tu.assert_result_close(res_grad, ref_grad)


# ---------------------------------------------------------------------------
# Negative cases: unsupported rank, dtype and parameter values. Every row is
# collected in both quick and default modes.
# ---------------------------------------------------------------------------

BAD_RANKS = [(1, 1, 2, 3, 4), (2, 5, 8), (5, 8), (8,)]


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
@pytest.mark.parametrize("shape", BAD_RANKS)
def test__scaled_dot_product_flash_attention_for_cpu_negative_rank(shape):
    q = _cpu_input(torch.float32, shape, ["-1", "1"])
    # Rank 3 and above hit the native rank guard (RuntimeError); below that the
    # kernel indexes dim -4 before the guard and raises IndexError.
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems._scaled_dot_product_flash_attention_for_cpu(q, q, q)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
def test__scaled_dot_product_flash_attention_for_cpu_negative_head_dim():
    dtype = torch.float32
    q = _cpu_input(dtype, (2, 3, 5, 8), ["-1", "1"])
    k = _cpu_input(dtype, (2, 3, 5, 16), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._scaled_dot_product_flash_attention_for_cpu(q, k, k)


BAD_MASK_SHAPES = [(5,), (2, 5, 5), (5, 6)]


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
@pytest.mark.parametrize("mask_shape", BAD_MASK_SHAPES)
def test__scaled_dot_product_flash_attention_for_cpu_negative_mask_shape(mask_shape):
    dtype = torch.float32
    q, k, v = _qkv(dtype, MASK_QUERY_SHAPE, ["-1", "1"])
    mask = _cpu_input(dtype, mask_shape, ["-1", "1"])
    # 1-D and 3-D masks fail the native dim guard; (5, 6) fails the internal
    # expand against (B, H, Sq, Sk).
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._scaled_dot_product_flash_attention_for_cpu(q, k, v, attn_mask=mask)


BAD_MASK_DTYPES = [torch.float16, torch.float64, torch.bool]


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
@pytest.mark.parametrize("mask_dtype", BAD_MASK_DTYPES)
def test__scaled_dot_product_flash_attention_for_cpu_negative_mask_dtype(mask_dtype):
    # A float32 query accepts a float32 mask only.
    dtype = torch.float32
    q, k, v = _qkv(dtype, MASK_QUERY_SHAPE, ["-1", "1"])
    mask = _cpu_input(mask_dtype, (5, 5), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._scaled_dot_product_flash_attention_for_cpu(q, k, v, attn_mask=mask)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
@pytest.mark.parametrize("dropout_p", [0.5, -1.0])
def test__scaled_dot_product_flash_attention_for_cpu_negative_dropout(dropout_p):
    q, k, v = _qkv(torch.float32, MASK_QUERY_SHAPE, ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._scaled_dot_product_flash_attention_for_cpu(q, k, v, dropout_p)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
def test__scaled_dot_product_flash_attention_for_cpu_negative_scale():
    q, k, v = _qkv(torch.float32, MASK_QUERY_SHAPE, ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._scaled_dot_product_flash_attention_for_cpu(
            q, k, v, 0.0, False, scale="x"
        )


UNSUPPORTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]


@pytest.mark.scaled_dot_product_flash_attention_for_cpu
@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES)
def test__scaled_dot_product_flash_attention_for_cpu_negative_dtype(dtype):
    # Values are irrelevant: the native kernel rejects the dtype before reading
    # the operands, so a non-negative grid is cast straight to the target dtype.
    q = _cpu_input(torch.float32, (2, 3, 5, 8), ["0", "1"]).to(dtype)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._scaled_dot_product_flash_attention_for_cpu(q, q, q)
