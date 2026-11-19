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

# Select every overload and contract case with the operator marker.
pytestmark = pytest.mark.thnn_conv2d

# Test THNN conv2d with matching input/weight channels and kernel dimensions.
# Extreme ranges retain their bounds and use the original reference dtype to preserve overflow.
# Cases: (input shape, weight shape, kernel size, stride, padding).
THNN_CONV2D_CASES = tu.selected_cases(
    [
        ((1, 2, 5, 5), (1, 2, 3, 3), (3, 3), (1, 1), (1, 1)),
        ((2, 3, 9, 9), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0)),
        ((2, 3, 8, 8), (5, 3, 3, 3), (3, 3), (2, 2), (1, 1)),
        ((2, 3, 8, 8), (5, 3, 3, 5), (3, 5), (1, 1), (1, 2)),
        ((2, 8, 16, 16), (16, 8, 1, 1), (1, 1), (1, 1), (0, 0)),
        ((4, 16, 32, 32), (8, 16, 3, 3), (3, 3), (1, 1), (1, 1)),
        ((1, 4, 12, 12), (4, 4, 5, 5), (5, 5), (1, 1), (2, 2)),
        ((2, 3, 4, 4), (5, 3, 3, 3), (3, 3), (1, 1), (0, 0)),
    ],
    quick=[
        ((1, 2, 5, 5), (1, 2, 3, 3), (3, 3), (1, 1), (1, 1)),
    ],
)

# Extra same-overload geometries covered by master. Keep the existing cases.
THNN_FORWARD_CASES = THNN_CONV2D_CASES + tu.selected_cases(
    [
        ((2, 3, 16, 16), (4, 3, 3, 3), (3, 3), (1, 1), (1, 1)),
        ((1, 1, 8, 8), (2, 1, 3, 3), (3, 3), (1, 1), (1, 1)),
        ((4, 8, 9, 7), (8, 8, 3, 3), (3, 3), (1, 1), (1, 1)),
        ((1, 16, 32, 32), (8, 16, 3, 3), (3, 3), (1, 1), (1, 1)),
        ((8, 4, 28, 28), (4, 4, 3, 3), (3, 3), (1, 1), (1, 1)),
        ((2, 3, 19, 17), (5, 3, 3, 3), (3, 3), (1, 1), (1, 1)),
        ((2, 3, 19, 17), (5, 3, 1, 1), (1, 1), (1, 1), (0, 0)),
        ((2, 3, 19, 17), (5, 3, 5, 5), (5, 5), (1, 1), (2, 2)),
        ((2, 3, 19, 17), (5, 3, 3, 5), (3, 5), (1, 1), (1, 2)),
        ((2, 3, 19, 17), (5, 3, 3, 3), (3, 3), (2, 2), (1, 1)),
        ((2, 3, 19, 17), (5, 3, 4, 4), (4, 4), (3, 1), (0, 0)),
    ],
    quick=[],
)

BIASES = tu.selected_cases([True, False], quick=[True])

_BACKWARD_CASES = THNN_CONV2D_CASES[:3]

# Scale finite random inputs to limit low-precision reduction noise.
_INPUT_SCALE = 0.1

_GEMS_ERRORS = (TypeError, ValueError, RuntimeError, AttributeError)

_UNSUPPORTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int32,
    torch.int64,
    torch.bool,
]


def _conv_output_shape(inp_shape, weight_shape, kernel_size, stride, padding):
    n, _, h_in, w_in = inp_shape
    out_c, _, k_h, k_w = weight_shape
    h_out = (h_in + 2 * padding[0] - k_h) // stride[0] + 1
    w_out = (w_in + 2 * padding[1] - k_w) // stride[1] + 1
    return (n, out_c, h_out, w_out)


def _make_conv_inputs(inp_shape, weight_shape, with_bias, dtype, value_range):
    inp = tu.make_input(dtype, inp_shape, value_range)
    weight = tu.make_input(dtype, weight_shape, value_range)
    if with_bias:
        bias = tu.make_input(dtype, (weight_shape[0],), value_range)
    else:
        bias = None
    return inp, weight, bias


_MUSA = flag_gems.vendor_name == "mthreads"


def _reference_input(tensor, upcast=False):
    if not _MUSA or tensor is None:
        return tu.to_reference(tensor, upcast)
    # MUSA lacks this native convolution. Preserve the reference's intended
    # precision and gradients while evaluating the corresponding CPU ATen API.
    dtype = torch.float64 if upcast else tensor.dtype
    reference = tensor.detach().to(device="cpu", dtype=dtype, copy=True)
    return reference.requires_grad_(tensor.requires_grad)


def _comparison_output(result, expected_device):
    assert result.device == expected_device
    return result.cpu() if _MUSA else result


def _assert_close(res_out, ref_out, dtype, equal_nan=False):
    # Supplement dtype-relative tolerance for the fp64 reference with absolute reduction-error bounds.
    if dtype == torch.bfloat16:
        atol = 5e-2
    elif dtype == torch.float16:
        atol = 1e-2
    else:
        atol = 1e-4
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol, equal_nan=equal_nan)


def _reduction_dims(inp_shape, weight_shape, out_shape):
    n, _, _, _ = inp_shape
    out_c, _, k_h, k_w = weight_shape
    # grad_input contracts over C_out x kH x kW.
    in_reduce_dim = out_c * k_h * k_w
    # grad_weight / grad_bias contract over N x H_out x W_out.
    out_reduce_dim = n * out_shape[2] * out_shape[3]
    return in_reduce_dim, out_reduce_dim


def _assert_grads_close(res_grads, ref_grads, in_reduce_dim, out_reduce_dim, dtype):
    # Masked gradients must remain None; compare computed gradients using their reduction sizes.
    for res_g, ref_g, reduce_dim in zip(
        res_grads, ref_grads, (in_reduce_dim, out_reduce_dim, out_reduce_dim)
    ):
        if ref_g is None:
            assert res_g is None
        else:
            utils.gems_assert_close(
                res_g, ref_g.to(dtype), dtype, reduce_dim=reduce_dim
            )


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding", THNN_FORWARD_CASES
)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_thnn_conv2d(
    inp_shape, weight_shape, kernel_size, stride, padding, dtype, bias
):
    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, bias, dtype, ["-1", "1"]
    )
    ref_inp = _reference_input(inp, True)
    ref_weight = _reference_input(weight, True)
    ref_bias = _reference_input(bias_t, True)

    ref_out = torch.ops.aten.thnn_conv2d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding
    ).to(dtype)

    res_out = flag_gems.thnn_conv2d(inp, weight, kernel_size, bias_t, stride, padding)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    _assert_close(_comparison_output(res_out, inp.device), ref_out, dtype)


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding", THNN_CONV2D_CASES[:2]
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_thnn_conv2d_value_ranges(
    inp_shape, weight_shape, kernel_size, stride, padding, value_range, dtype
):
    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, True, dtype, value_range
    )
    ref_inp = _reference_input(inp, not tu.is_extreme_range(value_range))
    ref_weight = _reference_input(weight, not tu.is_extreme_range(value_range))
    ref_bias = _reference_input(bias_t, not tu.is_extreme_range(value_range))

    ref_out = torch.ops.aten.thnn_conv2d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding
    ).to(dtype)

    res_out = flag_gems.thnn_conv2d(inp, weight, kernel_size, bias_t, stride, padding)

    tu.assert_result_close(_comparison_output(res_out, inp.device), ref_out)


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding", _BACKWARD_CASES
)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("bias", tu.selected_cases(BIASES))
def test_thnn_conv2d_backward(
    inp_shape, weight_shape, kernel_size, stride, padding, dtype, bias
):
    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, bias, dtype, ["-1", "1"]
    )
    inp = (_INPUT_SCALE * inp).requires_grad_()
    weight = (_INPUT_SCALE * weight).requires_grad_()
    if bias_t is not None:
        bias_t = (_INPUT_SCALE * bias_t).requires_grad_()

    out_shape = _conv_output_shape(
        inp_shape, weight_shape, kernel_size, stride, padding
    )
    grad_out = tu.make_input(dtype, out_shape, ["-1", "1"])

    ref_inp = _reference_input(inp, True)
    ref_weight = _reference_input(weight, True)
    ref_bias = _reference_input(bias_t, True)
    ref_grad_out = _reference_input(grad_out, True)
    ref_out = torch.ops.aten.thnn_conv2d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding
    )
    if ref_bias is None:
        ref_gi, ref_gw = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight), ref_grad_out
        )
        ref_gb = None
    else:
        ref_gi, ref_gw, ref_gb = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, ref_bias), ref_grad_out
        )

    res_out = flag_gems.thnn_conv2d(inp, weight, kernel_size, bias_t, stride, padding)

    tu.assert_result_close(_comparison_output(res_out, inp.device), ref_out.to(dtype))

    assert res_out.requires_grad, "candidate must preserve autograd"

    in_reduce_dim, out_reduce_dim = _reduction_dims(inp_shape, weight_shape, out_shape)
    if bias_t is None:
        res_gi, res_gw = torch.autograd.grad(res_out, (inp, weight), grad_out)
        res_gb = None
    else:
        res_gi, res_gw, res_gb = torch.autograd.grad(
            res_out, (inp, weight, bias_t), grad_out
        )
    _assert_grads_close(
        tuple(
            _comparison_output(grad, inp.device) if grad is not None else None
            for grad in (res_gi, res_gw, res_gb)
        ),
        (ref_gi, ref_gw, ref_gb),
        in_reduce_dim,
        out_reduce_dim,
        dtype,
    )


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(utils.ALL_FLOAT_DTYPES))
)
@pytest.mark.parametrize("special_arg", ["inp", "weight", "bias"])
def test_thnn_conv2d_nan_inf(dtype, scenario, special_arg):
    inp_shape, weight_shape, kernel_size, stride, padding = (
        (1, 2, 5, 5),
        (5, 2, 3, 3),
        (3, 3),
        (1, 1),
        (1, 1),
    )
    inp = torch.ones(inp_shape, dtype=dtype, device=flag_gems.device)
    weight = torch.ones(weight_shape, dtype=dtype, device=flag_gems.device)
    bias = torch.ones((weight_shape[0],), dtype=dtype, device=flag_gems.device)

    specials = tu.make_special_input(dtype, scenario)
    target = {"inp": inp, "weight": weight, "bias": bias}[special_arg]
    target.flatten()[: specials.numel()] = specials

    ref_inp = _reference_input(inp, True)
    ref_weight = _reference_input(weight, True)
    ref_bias = _reference_input(bias, True)
    ref_out = torch.ops.aten.thnn_conv2d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding
    ).to(dtype)

    res_out = flag_gems.thnn_conv2d(inp, weight, kernel_size, bias, stride, padding)

    tu.assert_result_close(_comparison_output(res_out, inp.device), ref_out)


@pytest.mark.thnn_conv2d_out
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding", THNN_CONV2D_CASES
)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_thnn_conv2d_out(
    inp_shape, weight_shape, kernel_size, stride, padding, dtype, bias
):
    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, bias, dtype, ["-1", "1"]
    )
    ref_inp = _reference_input(inp, True)
    ref_weight = _reference_input(weight, True)
    ref_bias = _reference_input(bias_t, True)

    out_shape = _conv_output_shape(
        inp_shape, weight_shape, kernel_size, stride, padding
    )
    ref_out = torch.full(out_shape, 7.0, dtype=ref_inp.dtype, device=ref_inp.device)
    res_out = torch.full(out_shape, 7.0, dtype=dtype, device=flag_gems.device)

    torch.ops.aten.thnn_conv2d.out(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, out=ref_out
    )

    res_ret = flag_gems.thnn_conv2d(
        inp, weight, kernel_size, bias_t, stride, padding, out=res_out
    )

    # The .out overload must write into and return the caller's buffer.
    assert res_ret is res_out
    _assert_close(_comparison_output(res_out, inp.device), ref_out.to(dtype), dtype)


@pytest.mark.thnn_conv2d
def test_thnn_conv2d_rejects_kernel_size_mismatch():
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3, 3), ["-1", "1"])
    args = (inp, weight, (2, 2), None, (1, 1), (0, 0))
    with pytest.raises(RuntimeError):
        torch.ops.aten.thnn_conv2d(
            *(x.cpu() if _MUSA and isinstance(x, torch.Tensor) else x for x in args)
        )
    with pytest.raises(_GEMS_ERRORS):
        flag_gems.thnn_conv2d(*args)


@pytest.mark.thnn_conv2d
def test_thnn_conv2d_rejects_channel_mismatch():
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 3, 3, 3), ["-1", "1"])
    args = (inp, weight, (3, 3), None, (1, 1), (0, 0))
    with pytest.raises(RuntimeError):
        torch.ops.aten.thnn_conv2d(
            *(x.cpu() if _MUSA and isinstance(x, torch.Tensor) else x for x in args)
        )
    with pytest.raises(_GEMS_ERRORS):
        flag_gems.thnn_conv2d(*args)


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize("bad_shape", [(2, 5, 5), (2, 5, 5, 5, 5)])
def test_thnn_conv2d_rejects_non_4d_input(bad_shape):
    inp = tu.make_input(torch.float32, bad_shape, ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3, 3), ["-1", "1"])
    args = (inp, weight, (3, 3), None, (1, 1), (0, 0))
    with pytest.raises(RuntimeError):
        torch.ops.aten.thnn_conv2d(
            *(x.cpu() if _MUSA and isinstance(x, torch.Tensor) else x for x in args)
        )
    with pytest.raises(_GEMS_ERRORS):
        flag_gems.thnn_conv2d(*args)


@pytest.mark.thnn_conv2d
def test_thnn_conv2d_rejects_non_4d_weight():
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3), ["-1", "1"])
    args = (inp, weight, (3, 3), None, (1, 1), (0, 0))
    with pytest.raises(RuntimeError):
        torch.ops.aten.thnn_conv2d(
            *(x.cpu() if _MUSA and isinstance(x, torch.Tensor) else x for x in args)
        )
    with pytest.raises(_GEMS_ERRORS):
        flag_gems.thnn_conv2d(*args)


@pytest.mark.thnn_conv2d
def test_thnn_conv2d_rejects_bias_length_mismatch():
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3, 3), ["-1", "1"])
    for bad_len in (2, 0):
        bad_bias = tu.make_input(torch.float32, (bad_len,), ["-1", "1"])
        args = (inp, weight, (3, 3), bad_bias, (1, 1), (0, 0))
        with pytest.raises(RuntimeError):
            torch.ops.aten.thnn_conv2d(
                *(x.cpu() if _MUSA and isinstance(x, torch.Tensor) else x for x in args)
            )
        with pytest.raises(_GEMS_ERRORS):
            flag_gems.thnn_conv2d(*args)


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize("stride", [(0, 0), (-1, 1), (1, -1)])
def test_thnn_conv2d_rejects_nonpositive_stride(stride):
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3, 3), ["-1", "1"])
    args = (inp, weight, (3, 3), None, stride, (0, 0))
    with pytest.raises(RuntimeError):
        torch.ops.aten.thnn_conv2d(
            *(x.cpu() if _MUSA and isinstance(x, torch.Tensor) else x for x in args)
        )
    with pytest.raises(_GEMS_ERRORS):
        flag_gems.thnn_conv2d(*args)


@pytest.mark.thnn_conv2d
def test_thnn_conv2d_rejects_kernel_larger_than_input():
    inp = tu.make_input(torch.float32, (1, 2, 2, 2), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3, 3), ["-1", "1"])
    args = (inp, weight, (3, 3), None, (1, 1), (0, 0))
    with pytest.raises(RuntimeError):
        torch.ops.aten.thnn_conv2d(
            *(x.cpu() if _MUSA and isinstance(x, torch.Tensor) else x for x in args)
        )
    with pytest.raises(_GEMS_ERRORS):
        flag_gems.thnn_conv2d(*args)


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize("scalar_param", ["kernel_size", "stride", "padding"])
def test_thnn_conv2d_rejects_scalar_params(scalar_param):
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3, 3), ["-1", "1"])
    kwargs = {"kernel_size": (3, 3), "stride": (1, 1), "padding": (0, 0)}
    kwargs[scalar_param] = {"kernel_size": 3, "stride": 1, "padding": 0}[scalar_param]
    with pytest.raises(RuntimeError):
        torch.ops.aten.thnn_conv2d(
            _reference_input(inp), _reference_input(weight), **kwargs
        )
    with pytest.raises(_GEMS_ERRORS):
        flag_gems.thnn_conv2d(inp, weight, **kwargs)


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test_thnn_conv2d_rejects_unsupported_dtype(dtype):
    inp = tu.make_input(dtype, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(dtype, (1, 2, 3, 3), ["-1", "1"])
    args = (inp, weight, (3, 3), None, (1, 1), (1, 1))
    # CPU ATen accepts several integer types that the device implementation
    # rejects. Keep the device contract check against the candidate.
    if not (_MUSA and dtype in {torch.int8, torch.uint8, torch.int32, torch.int64}):
        with pytest.raises(RuntimeError):
            torch.ops.aten.thnn_conv2d(
                *(x.cpu() if _MUSA and isinstance(x, torch.Tensor) else x for x in args)
            )
    with pytest.raises(_GEMS_ERRORS):
        flag_gems.thnn_conv2d(*args)


_MASTER_EMPTY_SHAPES = [((0, 3, 16, 16), 4), ((2, 3, 0, 0), 4)]


def _master_conv_operands(shape, out_channels, dtype, kernel):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(
        (out_channels, shape[1], kernel, kernel), dtype=dtype, device=flag_gems.device
    )
    bias = torch.randn(out_channels, dtype=dtype, device=flag_gems.device)
    return inp, weight, bias


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_thnn_conv2d_default_bias(dtype):
    """Omitting bias uses the schema default (None), not a required argument."""
    inp, weight, _ = _master_conv_operands((2, 3, 16, 16), 4, dtype, 3)

    # Both sides use the schema defaults: bias None, stride [1, 1],
    # padding [0, 0].
    ref_out = torch.ops.aten.thnn_conv2d(
        _reference_input(inp),
        _reference_input(weight),
        (3, 3),
        None,
        [1, 1],
        [0, 0],
    )
    res_out = flag_gems.thnn_conv2d(inp, weight, (3, 3))

    _assert_close(_comparison_output(res_out, inp.device), ref_out, dtype)


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize("shape,out_channels", _MASTER_EMPTY_SHAPES)
# The zero-extent rows only need one float dtype: the result is empty and the
# tolerance never enters the comparison.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_thnn_conv2d_empty(shape, out_channels, dtype):
    """A zero-extent input or output returns an empty plane without launching."""
    inp, weight, bias = _master_conv_operands(shape, out_channels, dtype, 3)

    res_out = flag_gems.thnn_conv2d(inp, weight, (3, 3), bias, [1, 1], [1, 1])

    assert res_out.device == inp.device
    if shape[0] == 0:
        # The native operator accepts an empty batch and returns the same plane.
        ref_out = torch.ops.aten.thnn_conv2d(
            _reference_input(inp),
            _reference_input(weight),
            (3, 3),
            _reference_input(bias),
            [1, 1],
            [1, 1],
        )
        assert res_out.shape == ref_out.shape
        assert res_out.numel() == ref_out.numel()
        return
    # A zero spatial extent is rejected by the native kernel; the candidate must
    # still return the matching empty plane without launching.
    assert res_out.numel() == 0
    assert res_out.dtype == dtype


@pytest.mark.thnn_conv2d
# The non-contiguous row only needs one float dtype: the materialisation path
# is dtype-agnostic and the values are compared with the float32 tolerance.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_thnn_conv2d_noncontiguous_input(dtype):
    """A non-contiguous input is read through its logical indices."""
    base = torch.randn((2, 16, 16, 3), dtype=dtype, device=flag_gems.device)
    inp = base.permute(0, 3, 1, 2)  # NCHW view with transposed H/W strides
    assert not inp.is_contiguous()
    weight = torch.randn((4, 3, 3, 3), dtype=dtype, device=flag_gems.device)
    bias = torch.randn(4, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.thnn_conv2d(
        _reference_input(inp),
        _reference_input(weight),
        (3, 3),
        _reference_input(bias),
        [1, 1],
        [1, 1],
    )
    res_out = flag_gems.thnn_conv2d(inp, weight, (3, 3), bias, [1, 1], [1, 1])

    _assert_close(_comparison_output(res_out, inp.device), ref_out, dtype)


@pytest.mark.thnn_conv2d
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_thnn_conv2d_does_not_mutate_operands(dtype):
    """The inputs are read-only; the result is a fresh buffer."""
    inp, weight, bias = _master_conv_operands((2, 3, 16, 16), 4, dtype, 3)
    # The snapshots must live on the oracle device: gems_assert_equal moves the
    # candidate result to the reference side and asserts the reference is
    # already there, which a CUDA-side clone would violate in quick-cpu mode.
    before = (
        _reference_input(inp.clone()),
        _reference_input(weight.clone()),
        _reference_input(bias.clone()),
    )

    res_out = flag_gems.thnn_conv2d(inp, weight, (3, 3), bias, [1, 1], [1, 1])

    utils.gems_assert_equal(
        _comparison_output(inp, inp.device), before[0], equal_nan=True
    )
    utils.gems_assert_equal(
        _comparison_output(weight, inp.device), before[1], equal_nan=True
    )
    utils.gems_assert_equal(
        _comparison_output(bias, inp.device), before[2], equal_nan=True
    )
    assert res_out.device == inp.device
    assert not res_out._is_view()


@pytest.mark.thnn_conv2d
def test_thnn_conv2d_rejects_non_tensor_weight():
    inp = torch.randn((2, 3, 8, 8), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError, AttributeError)):
        flag_gems.thnn_conv2d(inp, [[[0.0]]], (3, 3), None, [1, 1], [1, 1])


@pytest.fixture(autouse=True)
def full_precision():
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
