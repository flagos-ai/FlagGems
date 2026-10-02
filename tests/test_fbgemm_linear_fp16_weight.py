# Copyright 2025, The FlagOS Contributors.
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

import pytest
import torch

import flag_gems

from . import test_utils as tu

pytestmark = [
    # Native FBGEMM entry points emit vendor deprecation notices; only warnings
    # are filtered, so a vendor notice cannot be read as a numeric failure.
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

# aten::fbgemm_linear_fp16_weight is a CPU-only FBGEMM entry point: the activation
# and the bias are dense CPU tensors and packed_weight is the opaque uint8 handle
# produced by aten::fbgemm_pack_gemm_matrix_fp16. That handle carries library
# internal state which cannot be rebuilt from plain tensor data, so both sides pack
# through the native companion op and receive the same native CPU / opaque types.
_DEVICE = torch.device("cpu")
_DTYPE = torch.float32
_FINITE = ("-1", "1")
_INVALID_ERRORS = (RuntimeError, TypeError, ValueError, IndexError)
_BIAS_FORMS = ("broadcast", "full")


def _input(dtype, shape, value_range):
    # The shared generator places tensors on flag_gems.device; this operator only
    # accepts CPU operands, so move them there.
    return tu.make_input(dtype, shape, value_range).cpu()


def _bias_shape(num_rows, bias_form):
    return (1,) if bias_form == "broadcast" else (num_rows,)


def _packed(weight):
    # The native packer saturates and writes into the contiguous weight it is
    # handed, so each call packs an independent clone: the oracle and the candidate
    # must not share one pack buffer.
    return torch.ops.aten.fbgemm_pack_gemm_matrix_fp16(
        weight.detach().clone().contiguous()
    )


def _native(inp, weight, bias):
    return torch.ops.aten.fbgemm_linear_fp16_weight(
        tu.to_reference(inp), _packed(tu.to_reference(weight)), tu.to_reference(bias)
    )


# Quick rows: the small batched case, a singleton inner dim / single output feature
# case and an empty-M case.
_QUICK_SHAPE_ROWS = [
    ((2, 19, 7), (16, 7)),
    ((256, 1), (4, 1)),
    ((0, 7), (5, 7)),
    ((1, 1), (1, 1)),
]

# (activation_shape, weight_shape); the activation last dim is the packed weight
# inner dim. Rank 0 / rank 1 activations are rejected by the native schema
# (input.dim() >= 2 and input.size(-1) == packed weight columns), so the 1-D spec
# shapes () / (1,) / (256,) are represented by the rank-2 counterparts (256, 1) and
# (3, 1) instead of being dropped.
_SHAPE_ROWS = tu.selected_cases(
    _QUICK_SHAPE_ROWS
    + [
        ((20, 320, 15), (16, 15)),
        ((1024, 1024), (64, 1024)),
        ((16, 128, 64, 60), (32, 60)),
        ((16, 7, 57, 32, 29), (8, 29)),
        ((3, 1), (1, 1)),
        ((2, 5), (4096, 5)),
        ((2, 4096), (8, 4096)),
    ],
    quick=_QUICK_SHAPE_ROWS,
)


@pytest.mark.fbgemm_linear_fp16_weight
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("bias_form", _BIAS_FORMS)
@pytest.mark.parametrize("shape_row", _SHAPE_ROWS)
def test_fbgemm_linear_fp16_weight(shape_row, bias_form, value_range):
    act_shape, weight_shape = shape_row
    inp = _input(_DTYPE, act_shape, value_range)
    weight = _input(_DTYPE, weight_shape, value_range)
    bias = _input(_DTYPE, _bias_shape(weight_shape[0], bias_form), value_range)

    ref_out = _native(inp, weight, bias)
    res_out = flag_gems.fbgemm_linear_fp16_weight(inp, _packed(weight), bias)

    tu.assert_result_close(res_out, ref_out)
    assert res_out.device == inp.device


_LAYOUT_SHAPE = ((2, 19, 14), (16, 14), (16,))
_LAYOUT_ROWS = [
    ("act_slice",) + _LAYOUT_SHAPE,
    ("act_permute",) + _LAYOUT_SHAPE,
    ("weight_offset",) + _LAYOUT_SHAPE,
    ("bias_strided",) + _LAYOUT_SHAPE,
    ("bias_offset",) + _LAYOUT_SHAPE,
]
_LAYOUT_CASES = tu.selected_cases(_LAYOUT_ROWS, quick=_LAYOUT_ROWS)


def _strided_operands(kind, act_shape, weight_shape, bias_shape):
    # Operands whose storage layout exercises strided / offset reads.
    if kind == "act_slice":
        base = _input(_DTYPE, (2 * act_shape[0],) + act_shape[1:], _FINITE)
        inp = base[::2]
    elif kind == "act_permute":
        base = _input(_DTYPE, (act_shape[1], act_shape[0], act_shape[2]), _FINITE)
        inp = base.permute(1, 0, 2)
    else:
        inp = _input(_DTYPE, act_shape, _FINITE)
    if kind == "weight_offset":
        weight = _input(_DTYPE, (2,) + weight_shape, _FINITE)[1]
    else:
        weight = _input(_DTYPE, weight_shape, _FINITE)
    if kind == "bias_strided":
        bias = _input(_DTYPE, bias_shape + (2,), _FINITE)[..., 0]
    elif kind == "bias_offset":
        bias = _input(_DTYPE, (bias_shape[0] + 4,), _FINITE)[4:]
    else:
        bias = _input(_DTYPE, bias_shape, _FINITE)
    return inp, weight, bias


@pytest.mark.fbgemm_linear_fp16_weight
@pytest.mark.parametrize("layout", _LAYOUT_CASES)
def test_fbgemm_linear_fp16_weight_strided_operands(layout):
    kind, act_shape, weight_shape, bias_shape = layout
    inp, weight, bias = _strided_operands(kind, act_shape, weight_shape, bias_shape)
    inp_before = inp.clone()
    weight_before = weight.clone()
    bias_before = bias.clone()

    ref_out = _native(inp, weight, bias)
    res_out = flag_gems.fbgemm_linear_fp16_weight(inp, _packed(weight), bias)

    tu.assert_result_close(res_out, ref_out)
    # Strided / offset storage has to be read through, and the native contract is
    # read-only for all three operands.
    tu.assert_result_equal(inp, inp_before)
    tu.assert_result_equal(weight, weight_before)
    tu.assert_result_equal(bias, bias_before)


# Every operand position crossed with every representable special scenario. The
# activation and the packed weight use float32 (their native dtype); the bias
# specials additionally cover the other supported floating bias dtypes.
_SPECIAL_ROWS = tu.selected_cases(
    [("input", _DTYPE, scenario) for _, scenario in tu.special_value_cases([_DTYPE])]
    + [("weight", _DTYPE, scenario) for _, scenario in tu.special_value_cases([_DTYPE])]
    + [
        ("bias", dtype, scenario)
        for dtype in (_DTYPE, torch.bfloat16, torch.float16, torch.float64)
        for _, scenario in tu.special_value_cases([dtype])
    ],
    quick=[],
)


def _special_operand(shape, dtype, scenario):
    # Reuse the shared special-value pattern, widened to the operand shape.
    pattern = tu.make_special_input(dtype, scenario).to(_DEVICE).reshape(-1)
    numel = 1
    for dim in shape:
        numel *= dim
    repeats = (numel + pattern.numel() - 1) // pattern.numel()
    return pattern.repeat(repeats)[:numel].reshape(shape)


@pytest.mark.fbgemm_linear_fp16_weight
@pytest.mark.parametrize("operand,dtype,scenario", _SPECIAL_ROWS)
def test_fbgemm_linear_fp16_weight_special_values(operand, dtype, scenario):
    act_shape, weight_shape = (2, 19, 7), (16, 7)
    inp = _input(_DTYPE, act_shape, _FINITE)
    weight = _input(_DTYPE, weight_shape, _FINITE)
    bias = _input(_DTYPE, (weight_shape[0],), _FINITE)
    if operand == "input":
        inp = _special_operand(act_shape, dtype, scenario)
    elif operand == "weight":
        weight = _special_operand(weight_shape, dtype, scenario)
    else:
        bias = _special_operand((weight_shape[0],), dtype, scenario)

    ref_out = _native(inp, weight, bias)
    res_out = flag_gems.fbgemm_linear_fp16_weight(inp, _packed(weight), bias)

    tu.assert_result_close(res_out, ref_out)


# Only the trailing bias add is differentiable: the GEMM is evaluated from the
# opaque handle, so differentiating the activation raises 'does not require grad'.
# The bias leaf is the differentiated input, including the numel-1 broadcast form
# whose gradient is reduced over the leading dims.
_BACKWARD_ROWS = tu.selected_cases(
    [
        ((2, 19, 7), (16, 7), (16,)),
        ((20, 320, 15), (16, 15), (16,)),
        ((2, 5), (4, 5), (1,)),
    ],
    quick=[],
)


@pytest.mark.fbgemm_linear_fp16_weight
@pytest.mark.parametrize("act_shape,weight_shape,bias_shape", _BACKWARD_ROWS)
@pytest.mark.parametrize(
    "bias_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_fbgemm_linear_fp16_weight_backward(
    act_shape, weight_shape, bias_shape, bias_dtype
):
    inp = _input(_DTYPE, act_shape, _FINITE).requires_grad_()
    ref_inp = tu.to_reference(inp)
    weight = _input(_DTYPE, weight_shape, _FINITE)
    bias = _input(bias_dtype, bias_shape, _FINITE).requires_grad_(True)
    ref_bias = bias.detach().clone().requires_grad_(True)
    upstream = _input(_DTYPE, act_shape[:-1] + (weight_shape[0],), _FINITE)

    ref_out = torch.ops.aten.fbgemm_linear_fp16_weight(
        ref_inp, _packed(tu.to_reference(weight)), ref_bias
    )
    res_out = flag_gems.fbgemm_linear_fp16_weight(inp, _packed(weight), bias)

    tu.assert_result_close(res_out, ref_out)
    assert res_out.requires_grad == ref_out.requires_grad
    ref_input_grad, ref_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_bias), grad_outputs=upstream, allow_unused=True
    )
    res_input_grad, res_grad = torch.autograd.grad(
        res_out, (inp, bias), grad_outputs=upstream, allow_unused=True
    )
    assert res_input_grad is ref_input_grad is None

    tu.assert_result_close(res_grad, ref_grad)


# Bias dtypes accepted by the native trailing add; the unsupported fp8 biases are
# covered as negative rows.
_BIAS_DTYPES = [
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.bool,
]
_BIAS_SHAPE_ROWS = tu.selected_cases(
    [((2, 19, 7), (16, 7)), ((20, 320, 15), (16, 15))],
    quick=[((2, 19, 7), (16, 7))],
)


@pytest.mark.fbgemm_linear_fp16_weight
@pytest.mark.parametrize("bias_dtype", _BIAS_DTYPES)
@pytest.mark.parametrize("shape_row", _BIAS_SHAPE_ROWS)
def test_fbgemm_linear_fp16_weight_bias_dtype(shape_row, bias_dtype):
    act_shape, weight_shape = shape_row
    inp = _input(_DTYPE, act_shape, _FINITE)
    weight = _input(_DTYPE, weight_shape, _FINITE)
    bias = _input(bias_dtype, (weight_shape[0],), _FINITE)

    ref_out = _native(inp, weight, bias)
    res_out = flag_gems.fbgemm_linear_fp16_weight(inp, _packed(weight), bias)

    tu.assert_result_close(res_out, ref_out)


# A packed weight with zero columns (K = 0) leaves the native accumulator
# uninitialized, so the result contents are undefined; only the allocation
# metadata contract is asserted here and no values are compared.
# Source: PyTorch5228986c39 QuantizedLinear.cpp allocates at::empty; linked
# FBGEMM dbc3157bf256f1339b3fa1fef2be89ac4078be0e FbgemmFPCommon.h
# writes only within k_ind<k. This is a numerical oracle gap, not a value pass.
_ZERO_K_ROWS = tu.selected_cases(
    [((3, 0), (2, 0), (2,)), ((2, 0), (1, 0), (1,))],
    quick=[((3, 0), (2, 0), (2,))],
)


@pytest.mark.fbgemm_linear_fp16_weight
@pytest.mark.parametrize("act_shape,weight_shape,bias_shape", _ZERO_K_ROWS)
def test_fbgemm_linear_fp16_weight_zero_k_metadata(act_shape, weight_shape, bias_shape):
    inp = _input(_DTYPE, act_shape, _FINITE)
    weight = _input(_DTYPE, weight_shape, _FINITE)
    bias = _input(_DTYPE, bias_shape, _FINITE)

    ref_out = _native(inp, weight, bias)
    res_out = flag_gems.fbgemm_linear_fp16_weight(inp, _packed(weight), bias)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.device == inp.device


_ACT_SHAPE = (3, 7)
_WEIGHT_SHAPE = (5, 7)
_NUM_ROWS = 5

_NEGATIVE_ROWS = [
    ("input_rank", (7,)),
    ("input_rank", ()),
    ("input_dtype", torch.float16),
    ("input_dtype", torch.int32),
    *[
        ("input_dtype", dtype)
        for dtype in [
            torch.bfloat16,
            torch.float64,
            torch.int8,
            torch.uint8,
            torch.int64,
            torch.bool,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.complex64,
            torch.complex128,
        ]
    ],
    ("input_last_dim", 6),
    ("input_type", None),
    ("input_type", 1.5),
    ("weight_type", ("tensor", torch.uint8)),
    ("weight_type", ("tensor", torch.float32)),
    ("weight_type", ("tensor", torch.float16)),
    ("weight_type", 3),
    ("weight_type", [1.0, 2.0]),
    ("bias_shape", (5, 2)),
    ("bias_shape", (3,)),
    ("bias_dtype", torch.float8_e4m3fn),
    ("bias_dtype", torch.float8_e5m2),
    ("bias_dtype", torch.complex64),
    ("bias_dtype", torch.complex128),
    ("bias_type", None),
    ("arity", None),
    ("unknown_kwarg", None),
    ("out_kwarg", None),
]


def _invalid_call(kind, value):
    # Build one negative row's arguments outside the raises block, so a constructor
    # failure cannot fake a pass.
    inp = _input(_DTYPE, _ACT_SHAPE, _FINITE)
    weight = _input(_DTYPE, _WEIGHT_SHAPE, _FINITE)
    bias = _input(_DTYPE, (_NUM_ROWS,), _FINITE)
    packed = _packed(weight)

    if kind == "input_rank":
        return (_input(_DTYPE, value, _FINITE), packed, bias), {}
    if kind == "input_dtype":
        return (torch.zeros(_ACT_SHAPE, dtype=value, device="cpu"), packed, bias), {}
    if kind == "input_last_dim":
        return (_input(_DTYPE, (_ACT_SHAPE[0], value), _FINITE), packed, bias), {}
    if kind == "input_type":
        return (value, packed, bias), {}
    if kind == "weight_type":
        if isinstance(value, tuple) and value and value[0] == "tensor":
            return (inp, _input(value[1], _WEIGHT_SHAPE, _FINITE), bias), {}
        return (inp, value, bias), {}
    if kind == "bias_shape":
        return (inp, packed, _input(_DTYPE, value, _FINITE)), {}
    if kind == "bias_dtype":
        return (inp, packed, torch.zeros((_NUM_ROWS,), dtype=value, device="cpu")), {}
    if kind == "bias_type":
        return (inp, packed, value), {}
    if kind == "arity":
        return (inp, packed), {}
    if kind == "unknown_kwarg":
        return (inp, packed, bias), {"tensor": inp}
    if kind == "out_kwarg":
        # The runtime overload list only has 'default'; no .out kernel exists.
        return (inp, packed, bias), {
            "out": torch.empty(_ACT_SHAPE[0], _NUM_ROWS, device=_DEVICE)
        }
    raise AssertionError(kind)


@pytest.mark.fbgemm_linear_fp16_weight
@pytest.mark.parametrize("row", _NEGATIVE_ROWS)
def test_fbgemm_linear_fp16_weight_invalid(row):
    args, kwargs = _invalid_call(*row)
    with pytest.raises(_INVALID_ERRORS):
        flag_gems.fbgemm_linear_fp16_weight(*args, **kwargs)
