# Copyright 2026, The FlagOS Contributors.
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

from typing import NamedTuple

import pytest
import torch

import flag_gems

from . import test_utils as tu

# `_scaled_dot_product_flash_attention_for_cpu_backward(grad_out, query, key,
# value, out, logsumexp, dropout_p, is_causal, attn_mask=None, scale=None)` is
# the CPU attention backward and returns the tuple
# `(grad_query, grad_key, grad_value)`. The native kernel is CPU-only, so the
# reference and the candidate both receive these CPU tensors unchanged. Every
# tensor operand is 4-D `(batch, head, seq, head_dim)`, which is how the spec's
# rank ladder is expressed here.
_SHAPES = [
    (1, 2, 19, 8),
    (1, 1, 1, 8),
    (1, 1, 2, 8),
    (1, 2, 16, 8),
    (2, 2, 32, 64),
    (2, 4, 64, 64),
    (2, 8, 128, 64),
    (1, 3, 128, 8),
]
_QUICK_SHAPE = (1, 2, 19, 8)

# Probed: int8/uint8/int32/int64/bool and the fp8 types raise
# "flash_attention_backward" not implemented for <type>; half, bfloat16,
# float and double are the implemented ones.
_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]

# One row per (query shape, key/value shape, value range); the last two rows are
# cross attention, where the query and key/value sequence lengths differ.
_GRID_ROWS = [
    (shape, shape, value_range)
    for shape in _SHAPES
    for value_range in tu.selected_ranges()
]
_GRID_ROWS += [
    ((1, 2, 19, 8), (1, 2, 16, 8), ["-1", "1"]),
    ((1, 2, 32, 8), (1, 2, 64, 8), ["-1", "1"]),
    ((2, 4, 64, 64), (2, 4, 16, 64), ["-1", "1"]),
]
# Quick keeps both call forms (self attention and cross attention) with small
# operands, over every supported dtype.
_GRID = tu.selected_cases(
    _GRID_ROWS,
    quick=[
        (_QUICK_SHAPE, _QUICK_SHAPE, ["-1", "1"]),
        ((1, 1, 1, 8), (1, 1, 1, 8), ["-1", "1"]),
        ((1, 1, 2, 8), (1, 1, 2, 8), ["-1", "1"]),
        ((1, 2, 19, 8), (1, 2, 16, 8), ["-1", "1"]),
    ],
)

_SPECIAL_SHAPE = (1, 2, 16, 8)
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_DTYPES), quick=[])

# dropout_p and is_causal are required positionals; attn_mask and scale are the
# optional arguments. is_causal and scale are forwarded to the forward as well,
# so the gradients belong to the state that was handed to the backward.
_PARAM_SHAPE = (1, 4, 128, 64)
_PARAM_ROWS = [
    (0.0, False, None),
    (0.5, False, None),
    (0.0, True, None),
    (0.0, False, 0.0),
    (0.0, False, 0.5),
    (0.0, False, -0.5),
    (0.0, False, 2.0),
    (0.0, False, 1e6),
    (0.0, False, float("inf")),
    (0.0, False, float("nan")),
]
# Quick keeps the cheap branches: the omitted-optional-argument call form, both
# dropout_p values and both is_causal values. Scale values stay default-only.
# dropout_p is a probability, so only in-range values are valid, and the CPU
# backward ignores it (probed: 0.5 is bit-identical to 0.0).
_PARAM_CASES = tu.selected_cases(
    _PARAM_ROWS,
    quick=_PARAM_ROWS[:8],
)

# Probed accepted float-mask shapes for (batch, head, seq, head_dim) =
# (1, 4, 128, 64): 2-D (seq, seq) and 4-D (batch, head, seq, seq). A bool mask
# is silently ignored by this kernel, so it is not used.
_MASK_SHAPE = (1, 4, 128, 64)
_MASK_CASES = tu.selected_cases(
    [(128, 128), (1, 4, 128, 128)], quick=[(128, 128), (1, 4, 128, 128)]
)

_NEG_SHAPE = (1, 2, 4, 8)
_NEG_DTYPE = torch.float32
# Probed: a rank-2/rank-3 query or a rank-1 mask raises IndexError; a rank-3
# mask or a 4-D mask of the wrong extent raises RuntimeError.
_BAD_QUERY_SHAPES = [(8, 8), (2, 2, 16)]
_BAD_MASK_SHAPES = [(64,), (1, 2, 4, 63)]
_UNSUPPORTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]


class _Operands(NamedTuple):
    grad_out: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    out: torch.Tensor
    logsumexp: torch.Tensor


def _operands(
    shape, dtype, value_range, *, kv_shape=None, fwd_kwargs=None, special=None
):
    """Build one operand group plus a reference copy of those same values.

    The upstream gradient is an input of this operator, so the reference group
    is a conversion of the candidate's operands; re-sampling it would compare
    the gradients of two different problems. `out` and `logsumexp` come from
    the real native forward, so the backward receives genuine workspace.
    """
    kv_shape = shape if kv_shape is None else kv_shape
    fwd_kwargs = {} if fwd_kwargs is None else fwd_kwargs
    query = tu.make_input(dtype, shape, value_range).cpu()
    key = tu.make_input(dtype, kv_shape, value_range).cpu()
    value = tu.make_input(dtype, kv_shape, value_range).cpu()
    out, logsumexp = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        query, key, value, **fwd_kwargs
    )
    grad_out = tu.make_input(dtype, shape, ["-1", "1"]).cpu()
    if special is not None:
        # The nan/inf payload goes into the upstream gradient, which keeps the
        # forward state finite and reaches both sides identically.
        payload = tu.make_special_input(dtype, special).flatten().cpu()
        count = min(payload.numel(), grad_out.numel())
        grad_out.reshape(-1)[:count] = payload[:count]
    inp = _Operands(grad_out, query, key, value, out, logsumexp)
    ref = _Operands(*(tu.to_reference(tensor) for tensor in inp))
    return inp, ref


def _compare_grads(res_grads, ref_grads):
    """Compare every component of (grad_query, grad_key, grad_value)."""
    tu.assert_result_close(res_grads[0], ref_grads[0])
    tu.assert_result_close(res_grads[1], ref_grads[1])
    tu.assert_result_close(res_grads[2], ref_grads[2])


@pytest.mark.scaled_dot_product_flash_attention_for_cpu_backward
@pytest.mark.parametrize("shape,kv_shape,value_range", _GRID)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_scaled_dot_product_flash_attention_for_cpu_backward(
    shape, kv_shape, value_range, dtype
):
    inp, ref = _operands(shape, dtype, value_range, kv_shape=kv_shape)

    ref_grads = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward(
        ref.grad_out, ref.query, ref.key, ref.value, ref.out, ref.logsumexp, 0.0, False
    )
    res_grads = flag_gems._scaled_dot_product_flash_attention_for_cpu_backward(
        inp.grad_out, inp.query, inp.key, inp.value, inp.out, inp.logsumexp, 0.0, False
    )

    # The native kernel is CPU-only: the gradients stay on the operand device.
    assert res_grads[0].device == inp.query.device
    _compare_grads(res_grads, ref_grads)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu_backward
@pytest.mark.parametrize("dropout_p,is_causal,scale", _PARAM_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_scaled_dot_product_flash_attention_for_cpu_backward_with_params(
    dropout_p, is_causal, scale, dtype
):
    fwd_kwargs = {"is_causal": is_causal}
    if scale is not None:
        fwd_kwargs["scale"] = scale
    inp, ref = _operands(_PARAM_SHAPE, dtype, ["-1", "1"], fwd_kwargs=fwd_kwargs)
    bwd_kwargs = {} if scale is None else {"scale": scale}

    ref_grads = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward(
        ref.grad_out,
        ref.query,
        ref.key,
        ref.value,
        ref.out,
        ref.logsumexp,
        dropout_p,
        is_causal,
        **bwd_kwargs,
    )
    res_grads = flag_gems._scaled_dot_product_flash_attention_for_cpu_backward(
        inp.grad_out,
        inp.query,
        inp.key,
        inp.value,
        inp.out,
        inp.logsumexp,
        dropout_p,
        is_causal,
        **bwd_kwargs,
    )

    _compare_grads(res_grads, ref_grads)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu_backward
@pytest.mark.parametrize("mask_shape", _MASK_CASES)
@pytest.mark.parametrize(
    "dtype,mask_dtype",
    [
        (dtype, mask_dtype)
        for dtype in _DTYPES
        for mask_dtype in dict.fromkeys([dtype, torch.float32])
    ],
)
def test_scaled_dot_product_flash_attention_for_cpu_backward_with_attn_mask(
    mask_shape, dtype, mask_dtype
):
    mask = tu.make_input(mask_dtype, mask_shape, ["-1", "1"]).cpu()
    inp, ref = _operands(
        _MASK_SHAPE, dtype, ["-1", "1"], fwd_kwargs={"attn_mask": mask}
    )

    ref_grads = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward(
        ref.grad_out,
        ref.query,
        ref.key,
        ref.value,
        ref.out,
        ref.logsumexp,
        0.0,
        False,
        attn_mask=tu.to_reference(mask),
    )
    res_grads = flag_gems._scaled_dot_product_flash_attention_for_cpu_backward(
        inp.grad_out,
        inp.query,
        inp.key,
        inp.value,
        inp.out,
        inp.logsumexp,
        0.0,
        False,
        attn_mask=mask,
    )

    _compare_grads(res_grads, ref_grads)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu_backward
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_scaled_dot_product_flash_attention_for_cpu_backward_special_values(
    dtype, scenario
):
    inp, ref = _operands(_SPECIAL_SHAPE, dtype, ["-1", "1"], special=scenario)

    ref_grads = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward(
        ref.grad_out, ref.query, ref.key, ref.value, ref.out, ref.logsumexp, 0.0, False
    )
    res_grads = flag_gems._scaled_dot_product_flash_attention_for_cpu_backward(
        inp.grad_out, inp.query, inp.key, inp.value, inp.out, inp.logsumexp, 0.0, False
    )

    # A nan/inf upstream gradient makes the native gradients nan, so the
    # comparison is nan-aware rather than expecting finite values.
    _compare_grads(res_grads, ref_grads)


@pytest.mark.scaled_dot_product_flash_attention_for_cpu_backward
@pytest.mark.parametrize("query_shape", _BAD_QUERY_SHAPES)
def test_scaled_dot_product_flash_attention_for_cpu_backward_rejects_bad_query_rank(
    query_shape,
):
    inp, _ = _operands(_NEG_SHAPE, _NEG_DTYPE, ["-1", "1"])
    query = tu.make_input(_NEG_DTYPE, query_shape, ["-1", "1"]).cpu()
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems._scaled_dot_product_flash_attention_for_cpu_backward(
            inp.grad_out, query, inp.key, inp.value, inp.out, inp.logsumexp, 0.0, False
        )


@pytest.mark.scaled_dot_product_flash_attention_for_cpu_backward
@pytest.mark.parametrize("mask_shape", _BAD_MASK_SHAPES)
def test_scaled_dot_product_flash_attention_for_cpu_backward_rejects_bad_mask(
    mask_shape,
):
    inp, _ = _operands(_NEG_SHAPE, _NEG_DTYPE, ["-1", "1"])
    mask = tu.make_input(_NEG_DTYPE, mask_shape, ["-1", "1"]).cpu()
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems._scaled_dot_product_flash_attention_for_cpu_backward(
            inp.grad_out,
            inp.query,
            inp.key,
            inp.value,
            inp.out,
            inp.logsumexp,
            0.0,
            False,
            attn_mask=mask,
        )


@pytest.mark.scaled_dot_product_flash_attention_for_cpu_backward
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test_scaled_dot_product_flash_attention_for_cpu_backward_rejects_unsupported_dtype(
    dtype,
):
    # No native kernel exists for these dtypes, so the operands are zeros
    # instead of a forward result.
    q = torch.zeros(_NEG_SHAPE, dtype=dtype, device=flag_gems.device).cpu()
    grad_out = torch.zeros(_NEG_SHAPE, dtype=dtype, device=flag_gems.device).cpu()
    logsumexp = torch.zeros(
        _NEG_SHAPE[:3], dtype=torch.float32, device=flag_gems.device
    ).cpu()
    with pytest.raises(RuntimeError):
        flag_gems._scaled_dot_product_flash_attention_for_cpu_backward(
            grad_out, q, q, q, q, logsumexp, 0.0, False
        )


@pytest.mark.scaled_dot_product_flash_attention_for_cpu_backward
def test_scaled_dot_product_flash_attention_for_cpu_backward_rejects_mixed_dtypes():
    inp, _ = _operands(_NEG_SHAPE, _NEG_DTYPE, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems._scaled_dot_product_flash_attention_for_cpu_backward(
            inp.grad_out,
            inp.query,
            inp.key,
            inp.value.double(),
            inp.out,
            inp.logsumexp,
            0.0,
            False,
        )


@pytest.mark.scaled_dot_product_flash_attention_for_cpu_backward
def test_scaled_dot_product_flash_attention_for_cpu_backward_rejects_head_mismatch():
    inp, _ = _operands(_NEG_SHAPE, _NEG_DTYPE, ["-1", "1"])
    key = tu.make_input(_NEG_DTYPE, (1, 2, 4, 16), ["-1", "1"]).cpu()
    value = tu.make_input(_NEG_DTYPE, (1, 2, 4, 16), ["-1", "1"]).cpu()
    with pytest.raises(RuntimeError):
        flag_gems._scaled_dot_product_flash_attention_for_cpu_backward(
            inp.grad_out, inp.query, key, value, inp.out, inp.logsumexp, 0.0, False
        )


@pytest.mark.scaled_dot_product_flash_attention_for_cpu_backward
def test_scaled_dot_product_flash_attention_for_cpu_backward_rejects_string_scale():
    inp, _ = _operands(_NEG_SHAPE, _NEG_DTYPE, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems._scaled_dot_product_flash_attention_for_cpu_backward(
            inp.grad_out,
            inp.query,
            inp.key,
            inp.value,
            inp.out,
            inp.logsumexp,
            0.0,
            False,
            scale="not-a-number",
        )


@pytest.mark.scaled_dot_product_flash_attention_for_cpu_backward
def test_scaled_dot_product_flash_attention_for_cpu_backward_rejects_tensor_scale():
    inp, _ = _operands(_NEG_SHAPE, _NEG_DTYPE, ["-1", "1"])
    # A 0-dim or single-element tensor is accepted natively; a multi-element
    # tensor is not.
    with pytest.raises(RuntimeError):
        flag_gems._scaled_dot_product_flash_attention_for_cpu_backward(
            inp.grad_out,
            inp.query,
            inp.key,
            inp.value,
            inp.out,
            inp.logsumexp,
            0.0,
            False,
            scale=torch.tensor([1.0, 2.0]),
        )
