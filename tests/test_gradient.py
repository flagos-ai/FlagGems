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
        if flag_gems.vendor_name == "mthreads" and res_part.is_complex():
            assert res_part.dtype == ref_part.dtype
            res_part, ref_part = res_part.cpu(), ref_part.cpu()
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


DEVICE = flag_gems.device

# torch.gradient does not support uint8 and bool inputs; every other dtype is
# promoted like the reference (integral -> default float dtype).
_TEST_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
if flag_gems.runtime.device.support_fp64:
    _TEST_DTYPES.append(torch.float64)


@pytest.fixture(autouse=True)
def _skip_unsupported_fp64(request):
    """Skip float64 cases on backends that do not support FP64.

    Some XPU backends (e.g. Kunlunxin) silently materialise a float64 request as
    float32, so no tensor can ever satisfy the float64 dtype assertion here.
    """
    dtype = getattr(getattr(request.node, "callspec", None), "params", {}).get("dtype")
    if (
        dtype == torch.float64
        and flag_gems.vendor_name == "kunlunxin"
        and not flag_gems.runtime.device.support_fp64
    ):
        pytest.skip("backend does not support float64")


# every grad dim needs at least edge_order + 1 points
GRADIENT_SHAPES = [
    (1024,),
    (16, 32),
    (8, 16, 24),
    (2, 4, 8, 12),
    (15, 33),  # odd sizes
]


def _make_input(shape, dtype, device):
    torch.manual_seed(42)
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device) * 3
    return torch.randint(-100, 100, shape, dtype=dtype, device=device)


@pytest.mark.gradient
@pytest.mark.parametrize("shape", GRADIENT_SHAPES)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("edge_order", [1, 2])
def test_gradient_array(shape, dtype, edge_order):
    # exercises aten::gradient.array (no spacing)
    if min(shape) < edge_order + 1:
        pytest.skip("dimension smaller than edge_order + 1")
    inp = _make_input(shape, dtype, DEVICE)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.gradient.array(
        ref_inp, dim=list(range(ref_inp.dim())), edge_order=edge_order
    )
    res_out = flag_gems.gradient(inp, edge_order=edge_order)

    assert len(res_out) == ref_inp.dim()
    for res, ref in zip(res_out, ref_out):
        utils.gems_assert_close(res, ref, dtype)


@pytest.mark.gradient
@pytest.mark.parametrize("shape", GRADIENT_SHAPES)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("spacing", [None, 1.0, 2.5])
def test_gradient_scalarint(shape, dtype, spacing):
    # exercises aten::gradient.scalarint (Scalar? spacing, int? dim)
    inp = _make_input(shape, dtype, DEVICE)
    ref_inp = utils.to_reference(inp)
    dim = ref_inp.dim() - 1

    ref_out = torch.ops.aten.gradient.scalarint(ref_inp, spacing=spacing, dim=dim)
    if spacing is None:
        res_out = flag_gems.gradient(inp, dim=dim)
    else:
        res_out = flag_gems.gradient(inp, spacing=spacing, dim=dim)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)


@pytest.mark.gradient
@pytest.mark.parametrize("shape", GRADIENT_SHAPES)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
def test_gradient_scalararray(shape, dtype):
    # exercises aten::gradient.scalararray (Scalar spacing + int[] dim)
    inp = _make_input(shape, dtype, DEVICE)
    ref_inp = utils.to_reference(inp)
    dims = list(range(ref_inp.dim()))

    ref_out = torch.ops.aten.gradient.scalararray(ref_inp, spacing=2.0, dim=dims)
    res_out = flag_gems.gradient(inp, spacing=2.0, dim=dims)

    assert len(res_out) == len(dims)
    for res, ref in zip(res_out, ref_out):
        utils.gems_assert_close(res, ref, dtype)


@pytest.mark.gradient
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradient_scalarrayint(dtype):
    # exercises aten::gradient.scalarrayint (Scalar[] spacing, dim=None)
    shape = (4, 8, 12)
    inp = _make_input(shape, dtype, DEVICE)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.gradient.scalarrayint(ref_inp, spacing=[1.0, 2.0, 0.5])
    res_out = flag_gems.gradient(inp, spacing=[1.0, 2.0, 0.5])

    for res, ref in zip(res_out, ref_out):
        utils.gems_assert_close(res, ref, dtype)


@pytest.mark.gradient
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_gradient_tensorarray(dtype):
    # exercises aten::gradient.tensorarray / tensorarrayint (1-D coordinate
    # spacing tensors, uniform and non-uniform, both edge orders)
    shape = (16, 32)
    inp = _make_input(shape, dtype, DEVICE)
    ref_inp = utils.to_reference(inp)
    dim = 1

    xs = torch.linspace(0, 1, shape[dim], dtype=dtype, device=DEVICE)
    ref_out = torch.ops.aten.gradient.tensorarray(
        ref_inp, spacing=[utils.to_reference(xs)], dim=[dim], edge_order=2
    )
    res_out = flag_gems.gradient(inp, spacing=[xs], dim=[dim], edge_order=2)
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)

    # non-uniform coordinates
    xs_nu = torch.cumsum(torch.rand(shape[dim], dtype=dtype, device=DEVICE) + 0.5, 0)
    xs_nu = xs_nu - xs_nu[0]
    ref_out = torch.ops.aten.gradient.tensorarray(
        ref_inp, spacing=[utils.to_reference(xs_nu)], dim=[dim], edge_order=1
    )
    res_out = flag_gems.gradient(inp, spacing=[xs_nu], dim=[dim], edge_order=1)
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)

    ref_out = torch.ops.aten.gradient.tensorarray(
        ref_inp, spacing=[utils.to_reference(xs_nu)], dim=[dim], edge_order=2
    )
    res_out = flag_gems.gradient(inp, spacing=[xs_nu], dim=[dim], edge_order=2)
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)


@pytest.mark.gradient
def test_gradient_tensorarrayint_promotion():
    # tensor spacing promotes the output dtype like the reference:
    # promote_types(self, coords)
    x32 = torch.randn(4, 16, dtype=torch.float32, device=DEVICE)
    xs64 = torch.linspace(0, 1, 16, dtype=torch.float64, device=DEVICE)
    res = flag_gems.gradient(x32, spacing=[xs64], dim=[1])
    assert res[0].dtype == torch.result_type(x32, xs64)

    xs16 = torch.linspace(0, 1, 16, dtype=torch.float16, device=DEVICE)
    res = flag_gems.gradient(x32, spacing=[xs16], dim=[1])
    assert res[0].dtype == torch.result_type(x32, xs16)


@pytest.mark.gradient
def test_gradient_output_order_and_negative_dim():
    # outputs follow the order of `dim`, negative dims are wrapped
    inp = _make_input((4, 8, 12), torch.float32, DEVICE)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.gradient.array(ref_inp, dim=[2, 0])
    res_out = flag_gems.gradient(inp, dim=[2, 0])
    utils.gems_assert_close(res_out[0], ref_out[0], torch.float32)
    utils.gems_assert_close(res_out[1], ref_out[1], torch.float32)

    ref_out = torch.ops.aten.gradient.array(ref_inp, dim=[-1])
    res_out = flag_gems.gradient(inp, dim=[-1])
    utils.gems_assert_close(res_out[0], ref_out[0], torch.float32)


@pytest.mark.gradient
def test_gradient_non_contiguous_dimension():
    inp = _make_input((8, 16, 12), torch.float32, DEVICE)
    inp_t = inp.transpose(1, 2)
    ref = torch.ops.aten.gradient.array(utils.to_reference(inp_t.contiguous()), dim=[1])
    res = flag_gems.gradient(inp_t, dim=[1])
    utils.gems_assert_close(res[0], ref[0], torch.float32)


@pytest.mark.gradient
def test_gradient_int_promotion():
    # integral inputs are promoted by true division with a Scalar
    inp = torch.arange(48, device=DEVICE).reshape(4, 12)
    res = flag_gems.gradient(inp, dim=[1])
    ref = torch.ops.aten.gradient.array(utils.to_reference(inp), dim=[1])
    assert res[0].dtype == ref[0].dtype == torch.result_type(inp, 1.0)
    utils.gems_assert_close(res[0], ref[0], torch.float32)


@pytest.mark.gradient
@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin" and not flag_gems.runtime.device.support_fp64,
    reason="backend does not support float64",
)
def test_gradient_quadratic_exactness():
    # edge_order=2 is exact on quadratics: y = t**2 with unit spacing
    t = torch.arange(64, dtype=torch.float64, device=DEVICE)
    y = t**2
    res = flag_gems.gradient(y, spacing=1.0, dim=0, edge_order=2)[0]
    ref = torch.ops.aten.gradient.scalarint(
        utils.to_reference(y), spacing=1.0, dim=0, edge_order=2
    )[0]
    utils.gems_assert_close(res, ref, torch.float64)
    assert torch.allclose(res, 2 * t, rtol=1e-12, atol=1e-12)


@pytest.mark.gradient
def test_gradient_error_paths():
    inp = _make_input((4, 8, 12), torch.float32, DEVICE)
    with pytest.raises(RuntimeError):
        flag_gems.gradient(inp, dim=[1, 1])  # duplicate dim
    with pytest.raises(IndexError):
        flag_gems.gradient(inp, dim=[3])  # out of range
    with pytest.raises(RuntimeError):
        flag_gems.gradient(inp, edge_order=3)  # unsupported edge_order
    with pytest.raises(RuntimeError):
        flag_gems.gradient(inp.byte(), dim=[0])  # uint8 rejected
    # eo=2 requires every grad dim >= 3
    with pytest.raises(RuntimeError):
        flag_gems.gradient(
            _make_input((2, 8), torch.float32, DEVICE), dim=[0], edge_order=2
        )
