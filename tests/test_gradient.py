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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# aten::gradient differentiates a single tensor, so there is no operand to
# broadcast and the spec's broadcast dimension does not apply. Native exposes
# named overloads (spacing / dim / edge_order combinations) rather than a
# .default or .out overload; every one is reached through the single public
# entry point flag_gems.gradient.
#
# A differentiated dimension must have size >= edge_order + 1, so the spec's
# size-1 shape cannot carry the default all-dim call; size-1 inputs stay covered
# by a non-differentiated size-1 dimension and by a negative case.

_GRAD_DTYPES = [
    torch.int8,
    torch.int32,
    torch.float16,
    torch.float32,
    torch.complex64,
]
# Static backend capability flags, exactly as BF16/FP64 below: no runtime probe.
if utils.int64_is_supported:
    _GRAD_DTYPES.append(torch.int64)
if utils.bf16_is_supported:
    _GRAD_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    _GRAD_DTYPES.append(torch.float64)

_GRAD_FLOAT_DTYPES = [
    dtype
    for dtype in _GRAD_DTYPES
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
]

# The parameter families (call form, dim, edge_order, spacing) are dtype
# independent; every dtype below was measured valid for all of them.
_PARAM_DTYPES = [torch.int32, torch.float16]
if utils.bf16_is_supported:
    _PARAM_DTYPES.append(torch.bfloat16)

_GRAD_SHAPES = [shape for shape in tu.selected_shapes() if all(d >= 2 for d in shape)]

# A 1e30 spacing scales the true gradient to ~1e-30; the zero atol on that row
# keeps a candidate that returns exact zeros from passing.
_TINY_SCALE_SPACING = 1e30
_TENSOR_SPACING = "tensor"


def _assert_gradients(result, reference, inp, *, atol=None):
    # aten::gradient returns Tensor[]; compare every component.
    assert isinstance(result, (list, tuple))
    assert len(result) == len(reference)
    for res_part, ref_part in zip(result, reference):
        assert res_part.device == inp.device
        if atol is None:
            tu.assert_result_close(res_part, ref_part)
        else:
            tu.assert_result_close(res_part, ref_part, atol=atol)


def _upstream(parts, device):
    # A non-constant upstream: a constant one cannot expose a wrong interior
    # gradient formula.
    return [
        torch.linspace(-1.0, 1.0, part.numel(), dtype=torch.float32, device=device)
        .reshape(part.shape)
        .to(part.dtype)
        for part in parts
    ]


def _tensor_spacing(shape, dims, dtype, device):
    # Squared, non-uniform coordinates: a candidate that folds a tensor spacing
    # into one constant step cannot reproduce them.
    return [
        torch.arange(1, shape[dim] + 1, dtype=dtype, device=device).pow(2)
        for dim in dims
    ]


@pytest.mark.gradient
@pytest.mark.parametrize("shape", _GRAD_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _GRAD_DTYPES)
def test_gradient(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.gradient(ref_inp)
    res_out = flag_gems.gradient(inp)

    _assert_gradients(res_out, ref_out, inp)


# One row per native call form: (shape, dim, edge_order, spacing). A None entry
# omits that argument, so the schema default is exercised. Default-only.
_CALL_FORM_ROWS = tu.selected_cases(
    [
        ((1024, 1024), None, None, None),
        ((1024, 1024), [0, 1], None, 2.0),
        ((1024, 1024), [0, 1], None, None),
        ((1024, 1024), [1, 0], None, None),
        ((20, 320, 15), None, None, [0.5, 2.0, 4.0]),
        ((20, 320, 15), [0, 1, 2], None, [0.5, 2.0, 4.0]),
        ((20, 320, 15), None, None, _TENSOR_SPACING),
        ((20, 320, 15), [1, 2], None, _TENSOR_SPACING),
        ((20, 320, 15), 2, 2, None),
        ((20, 320, 15), [0, 2], 2, None),
        ((20, 320, 15), 0, 1, None),
        ((20, 320, 15), -1, 1, None),
        ((20, 320, 15), [2], 1, None),
        ((20, 320, 15), [2, 0], 1, None),
        ((20, 320, 15), [], 1, None),
        ((16, 128, 64, 60), None, 1, None),
        ((16, 7, 57, 32, 29), None, 1, None),
        ((16, 1, 57, 32, 29), [0, 2, 3, 4], 1, None),
        ((1024, 1024), None, 1, -2.0),
        ((1024, 1024), None, 1, 0.0),
        ((1024, 1024), None, 1, 1e-30),
        ((1024, 1024), None, 1, _TINY_SCALE_SPACING),
        ((1024, 1024), None, 1, float("inf")),
        ((1024, 1024), None, 1, float("nan")),
    ],
    quick=[],
)


@pytest.mark.gradient
@pytest.mark.parametrize("shape,dim,edge_order,spacing", _CALL_FORM_ROWS)
@pytest.mark.parametrize("dtype", _PARAM_DTYPES)
def test_gradient_call_forms(shape, dim, edge_order, spacing, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    res_kwargs = {}
    ref_kwargs = {}
    if dim is not None:
        res_kwargs["dim"] = dim
        ref_kwargs["dim"] = dim
    if edge_order is not None:
        res_kwargs["edge_order"] = edge_order
        ref_kwargs["edge_order"] = edge_order
    if spacing == _TENSOR_SPACING:
        dims = dim if dim is not None else list(range(inp.dim()))
        coord_dtype = dtype if dtype.is_floating_point else torch.float32
        res_kwargs["spacing"] = _tensor_spacing(
            shape, dims, coord_dtype, flag_gems.device
        )
        # The spacing tensors are operands of the call, so the reference gets its
        # own copy on the configured reference device.
        ref_kwargs["spacing"] = [
            tu.to_reference(part) for part in res_kwargs["spacing"]
        ]
    elif spacing is not None:
        res_kwargs["spacing"] = spacing
        ref_kwargs["spacing"] = spacing

    ref_out = torch.ops.aten.gradient(ref_inp, **ref_kwargs)
    res_out = flag_gems.gradient(inp, **res_kwargs)

    _assert_gradients(
        res_out,
        ref_out,
        inp,
        atol=0.0 if spacing == _TINY_SCALE_SPACING else None,
    )


# Non-contiguous views with non-unit strides and a non-zero storage offset.
# Default-only.
_NON_CONTIGUOUS_CASES = tu.selected_cases(
    [
        (False, (slice(1, 5), slice(None, None, 2), slice(2, 8))),
        (True, (slice(None), slice(1, 5), slice(None))),
    ],
    quick=[],
)


@pytest.mark.gradient
@pytest.mark.parametrize(
    "permute,index", _NON_CONTIGUOUS_CASES, ids=["strided", "permuted"]
)
@pytest.mark.parametrize("dtype", _PARAM_DTYPES)
def test_gradient_non_contiguous(permute, index, dtype):
    storage = tu.make_input(dtype, (6, 10, 8), ["-1", "1"])
    source = storage.permute(2, 0, 1) if permute else storage
    inp = source[index]
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.gradient(ref_inp)
    res_out = flag_gems.gradient(inp)

    _assert_gradients(res_out, ref_out, inp)


# Backward: gradient is differentiable for floating and complex inputs, so the
# candidate output must stay in the autograd graph. Default-only.
_BACKWARD_CASES = tu.selected_cases(
    [
        ((256,), 0),
        ((64, 64), [0, 1]),
        ((2, 19, 7), 1),
        ((2, 19, 7), None),
    ],
    quick=[],
)


@pytest.mark.gradient
@pytest.mark.parametrize("shape,dim", _BACKWARD_CASES)
@pytest.mark.parametrize("dtype", _GRAD_FLOAT_DTYPES + [torch.complex64])
def test_gradient_backward(shape, dim, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp.detach()).requires_grad_(True)
    gradient_kwargs = {} if dim is None else {"dim": dim}

    ref_out = torch.ops.aten.gradient(ref_inp, **gradient_kwargs)
    res_out = flag_gems.gradient(inp, **gradient_kwargs)
    _assert_gradients(res_out, ref_out, inp)

    # Both sides receive the same upstream gradient, transferred for the
    # reference device.
    res_upstream = _upstream(res_out, flag_gems.device)
    ref_upstream = [tu.to_reference(part) for part in res_upstream]
    ref_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)
    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=res_upstream)

    _assert_gradients(res_grad, ref_grad, inp)


# NaN / Inf scenarios, one per supported floating dtype. Default-only.
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_GRAD_FLOAT_DTYPES), quick=[])


@pytest.mark.gradient
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_gradient_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.gradient(ref_inp)
    res_out = flag_gems.gradient(inp)

    _assert_gradients(res_out, ref_out, inp)


# Negative cases assert only the candidate's exception.
_UNSUPPORTED_DTYPES = [torch.uint8, torch.bool]
if flag_gems.vendor_name == "nvidia":
    # Vendor-scoped: the fp8 rejection comes from this vendor's fp8 ufunc, not
    # from gradient itself, so other backends may accept these dtypes.
    _UNSUPPORTED_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]


@pytest.mark.gradient
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test_gradient_unsupported_dtype(dtype):
    inp = torch.ones((2, 19, 7), dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.gradient(inp)


@pytest.mark.gradient
@pytest.mark.parametrize("dim", [3, -4])
def test_gradient_dim_out_of_range(dim):
    inp = tu.make_input(torch.float32, (2, 19, 7), ["-1", "1"])
    with pytest.raises((IndexError, RuntimeError)):
        flag_gems.gradient(inp, dim=dim)


@pytest.mark.gradient
def test_gradient_dim_repeated():
    inp = tu.make_input(torch.float32, (2, 19, 7), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.gradient(inp, dim=[1, 1])


# Native accepts only edge_order 1 and 2; 0, 3 and -1 are measured rejects.
@pytest.mark.gradient
@pytest.mark.parametrize("edge_order", [0, 3, -1])
def test_gradient_invalid_edge_order(edge_order):
    inp = tu.make_input(torch.float32, (2, 19, 7), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.gradient(inp, dim=2, edge_order=edge_order)


@pytest.mark.gradient
@pytest.mark.parametrize("shape", [(1,), (1, 8), (4, 1)])
def test_gradient_dim_below_edge_order(shape):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.gradient(inp)


@pytest.mark.gradient
def test_gradient_spacing_length_mismatch():
    inp = tu.make_input(torch.float32, (2, 19, 7), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.gradient(inp, spacing=[2.0], dim=[0, 1])


@pytest.mark.gradient
@pytest.mark.parametrize("spacing_shape", [(), (4, 4)])
def test_gradient_spacing_tensor_rank(spacing_shape):
    inp = tu.make_input(torch.float32, (2, 19, 7), ["-1", "1"])
    bad_spacing = torch.ones(spacing_shape, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.gradient(inp, spacing=[bad_spacing], dim=[2])
