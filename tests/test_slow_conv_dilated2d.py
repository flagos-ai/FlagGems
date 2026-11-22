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
pytestmark = pytest.mark.slow_conv_dilated2d

# Test slow conv2d across kernel, stride, padding and dilation configurations.
# Extreme ranges retain their bounds and use the original reference dtype to preserve overflow.
# Cases: (input shape, weight shape, kernel size, stride, padding, dilation).
SLOW_CONV_DILATED2D_CASES = tu.selected_cases(
    [
        ((1, 2, 5, 5), (1, 2, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
        ((2, 3, 9, 9), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (1, 1)),
        ((2, 3, 8, 8), (5, 3, 3, 3), (3, 3), (2, 2), (1, 1), (1, 1)),
        ((2, 4, 8, 8), (6, 4, 3, 3), (3, 3), (1, 1), (2, 2), (2, 2)),
        ((1, 3, 7, 9), (4, 3, 3, 5), (3, 5), (1, 1), (1, 2), (1, 1)),
        ((2, 8, 16, 16), (16, 8, 1, 1), (1, 1), (1, 1), (0, 0), (1, 1)),
        ((1, 4, 12, 12), (8, 4, 5, 5), (5, 5), (2, 2), (2, 2), (2, 2)),
        ((2, 3, 6, 10), (5, 3, 3, 3), (3, 3), (2, 1), (1, 2), (1, 2)),
        ((2, 16, 12, 12), (8, 16, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
        ((1, 2, 4, 4), (3, 2, 2, 2), (2, 2), (1, 1), (0, 0), (2, 2)),
        # asymmetric kernel (2x3) and stride 3
        ((2, 3, 10, 10), (4, 3, 2, 3), (2, 3), (1, 1), (0, 0), (1, 1)),
        # stride 3 with asymmetric dilation (2, 1)
        ((1, 2, 9, 9), (3, 2, 3, 3), (3, 3), (3, 3), (0, 0), (2, 1)),
    ],
    quick=[
        ((1, 2, 5, 5), (1, 2, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
    ],
)

# Extra same-overload geometries covered by master. Keep the existing cases.
DILATED_CASES = SLOW_CONV_DILATED2D_CASES + tu.selected_cases(
    [
        ((2, 3, 8, 8), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (1, 1)),
        ((2, 3, 8, 8), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (2, 2)),
        ((2, 3, 8, 8), (4, 3, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
        ((2, 3, 8, 8), (4, 3, 3, 3), (3, 3), (1, 1), (1, 1), (2, 2)),
        ((2, 3, 8, 8), (4, 3, 3, 3), (3, 3), (2, 2), (0, 0), (1, 1)),
        ((2, 3, 8, 8), (4, 3, 3, 3), (3, 3), (2, 2), (0, 0), (2, 2)),
        ((2, 3, 8, 8), (4, 3, 3, 3), (3, 3), (2, 2), (1, 1), (1, 1)),
        ((2, 3, 8, 8), (4, 3, 3, 3), (3, 3), (2, 2), (1, 1), (2, 2)),
        ((1, 2, 7, 9), (3, 2, 2, 2), (2, 2), (1, 1), (0, 0), (1, 1)),
        ((1, 2, 7, 9), (3, 2, 2, 2), (2, 2), (1, 1), (0, 0), (2, 2)),
        ((1, 2, 7, 9), (3, 2, 2, 2), (2, 2), (1, 1), (1, 1), (1, 1)),
        ((1, 2, 7, 9), (3, 2, 2, 2), (2, 2), (1, 1), (1, 1), (2, 2)),
        ((1, 2, 7, 9), (3, 2, 2, 2), (2, 2), (2, 2), (0, 0), (1, 1)),
        ((1, 2, 7, 9), (3, 2, 2, 2), (2, 2), (2, 2), (0, 0), (2, 2)),
        ((1, 2, 7, 9), (3, 2, 2, 2), (2, 2), (2, 2), (1, 1), (1, 1)),
        ((1, 2, 7, 9), (3, 2, 2, 2), (2, 2), (2, 2), (1, 1), (2, 2)),
        ((2, 4, 12, 12), (6, 4, 3, 3), (3, 3), (1, 1), (0, 0), (1, 1)),
        ((2, 4, 12, 12), (6, 4, 3, 3), (3, 3), (1, 1), (0, 0), (2, 2)),
        ((2, 4, 12, 12), (6, 4, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
        ((2, 4, 12, 12), (6, 4, 3, 3), (3, 3), (1, 1), (1, 1), (2, 2)),
        ((2, 4, 12, 12), (6, 4, 3, 3), (3, 3), (2, 2), (0, 0), (1, 1)),
        ((2, 4, 12, 12), (6, 4, 3, 3), (3, 3), (2, 2), (0, 0), (2, 2)),
        ((2, 4, 12, 12), (6, 4, 3, 3), (3, 3), (2, 2), (1, 1), (1, 1)),
        ((2, 4, 12, 12), (6, 4, 3, 3), (3, 3), (2, 2), (1, 1), (2, 2)),
    ],
    quick=[],
)

BIASES = tu.selected_cases([True, False], quick=[True])

# Scale finite random inputs to limit low-precision reduction noise.
_INPUT_SCALE = 0.1

_SLOW_CONV_DILATED2D_VALUE_RANGES_CASES = tu.selected_cases(
    [
        ((1, 2, 5, 5), (2, 2, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
        ((2, 3, 8, 8), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (1, 1)),
        ((2, 4, 6, 6), (4, 4, 3, 3), (3, 3), (2, 2), (1, 1), (1, 1)),
        ((2, 3, 8, 8), (4, 3, 3, 3), (3, 3), (1, 1), (1, 1), (2, 2)),
        ((2, 2, 6, 6), (3, 2, 1, 1), (1, 1), (1, 1), (0, 0), (1, 1)),
    ],
    quick=[
        ((1, 2, 5, 5), (2, 2, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
    ],
)

_SLOW_CONV_DILATED2D_BACKWARD_CASES = tu.selected_cases(
    [
        ((1, 2, 5, 5), (3, 2, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
        ((2, 3, 6, 6), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (1, 1)),
    ],
    quick=[
        ((1, 2, 5, 5), (3, 2, 3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
    ],
)

_BACKWARD_DTYPES = tu.selected_cases(
    [torch.float16, torch.float32, torch.bfloat16, torch.float64], quick=[torch.float32]
)

# Negative padding is accepted by the reference, so it is not a negative case.
_INVALID_SLOW_CONV_DILATED2D_CASES = [
    # weight C_in disagrees with input C_in
    ((2, 3, 5, 5), (4, 5, 3, 3), (3, 3), (1, 1), (0, 0), (1, 1)),
    # weight spatial dims disagree with kernel_size
    ((2, 3, 5, 5), (4, 3, 4, 4), (3, 3), (1, 1), (0, 0), (1, 1)),
    # unbatched input has mismatched channels
    ((2, 3, 5), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (1, 1)),
    # weight is not 4-D
    ((2, 3, 5, 5), (4, 3, 3), (3, 3), (1, 1), (0, 0), (1, 1)),
    # kernel_size spatial dims disagree with the weight spatial dims
    ((2, 3, 5, 5), (4, 3, 3, 3), (2, 2), (1, 1), (0, 0), (1, 1)),
    # zero stride
    ((2, 3, 5, 5), (4, 3, 3, 3), (3, 3), (0, 0), (0, 0), (1, 1)),
    # zero dilation
    ((2, 3, 5, 5), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (0, 0)),
    # dilation so large the kernel no longer fits inside the input
    ((2, 3, 5, 5), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (10, 10)),
    # Negative padding crops farther than the input size.
    ((1, 3, 8, 8), (4, 3, 3, 3), (3, 3), (1, 1), (-5, -5), (1, 1)),
]

_UNSUPPORTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int32,
    torch.int64,
]


def _make_conv_inputs(
    inp_shape, weight_shape, with_bias, dtype, value_range=("-1", "1")
):
    inp = _INPUT_SCALE * tu.make_input(dtype, inp_shape, value_range)
    weight = _INPUT_SCALE * tu.make_input(dtype, weight_shape, value_range)
    if with_bias:
        bias = _INPUT_SCALE * tu.make_input(dtype, (weight_shape[0],), value_range)
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
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=equal_nan, atol=atol)


@pytest.mark.slow_conv_dilated2d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, dilation",
    DILATED_CASES,
)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_dilated2d(
    inp_shape, weight_shape, kernel_size, stride, padding, dilation, dtype, bias
):
    inp, weight, bias_t = _make_conv_inputs(inp_shape, weight_shape, bias, dtype)
    ref_inp = _reference_input(inp, True)
    ref_weight = _reference_input(weight, True)
    ref_bias = _reference_input(bias_t, True)

    ref_out = torch.ops.aten.slow_conv_dilated2d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    ).to(dtype)

    res_out = flag_gems.slow_conv_dilated2d(
        inp, weight, kernel_size, bias_t, stride, padding, dilation
    )

    assert res_out.shape == ref_out.shape
    _assert_close(_comparison_output(res_out, inp.device), ref_out, dtype)


@pytest.mark.slow_conv_dilated2d_out
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, dilation",
    SLOW_CONV_DILATED2D_CASES,
)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_dilated2d_out(
    inp_shape, weight_shape, kernel_size, stride, padding, dilation, dtype, bias
):
    inp, weight, bias_t = _make_conv_inputs(inp_shape, weight_shape, bias, dtype)
    ref_inp = _reference_input(inp, True)
    ref_weight = _reference_input(weight, True)
    ref_bias = _reference_input(bias_t, True)

    # The .out overload must write into the provided tensor and return it.
    ref_full = torch.ops.aten.slow_conv_dilated2d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    )
    ref_out = torch.empty_like(ref_full)
    ref_ret = torch.ops.aten.slow_conv_dilated2d.out(
        ref_inp,
        ref_weight,
        kernel_size,
        ref_bias,
        stride,
        padding,
        dilation,
        out=ref_out,
    )

    out = torch.empty(ref_full.shape, dtype=dtype, device=flag_gems.device)
    res_ret = flag_gems.slow_conv_dilated2d(
        inp, weight, kernel_size, bias_t, stride, padding, dilation, out=out
    )
    assert res_ret is out
    assert res_ret.dtype == dtype
    assert res_ret.shape == ref_ret.shape

    _assert_close(_comparison_output(res_ret, inp.device), ref_ret, dtype)


@pytest.mark.slow_conv_dilated2d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, dilation",
    _SLOW_CONV_DILATED2D_VALUE_RANGES_CASES,
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_dilated2d_value_ranges(
    inp_shape,
    weight_shape,
    kernel_size,
    stride,
    padding,
    dilation,
    value_range,
    dtype,
    bias,
):
    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, bias, dtype, value_range
    )
    ref_inp = _reference_input(inp, not tu.is_extreme_range(value_range))
    ref_weight = _reference_input(weight, not tu.is_extreme_range(value_range))
    ref_bias = _reference_input(bias_t, not tu.is_extreme_range(value_range))

    ref_out = torch.ops.aten.slow_conv_dilated2d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    ).to(dtype)

    res_out = flag_gems.slow_conv_dilated2d(
        inp, weight, kernel_size, bias_t, stride, padding, dilation
    )

    _assert_close(
        _comparison_output(res_out, inp.device), ref_out, dtype, equal_nan=True
    )


@pytest.mark.slow_conv_dilated2d_backward
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, dilation",
    _SLOW_CONV_DILATED2D_BACKWARD_CASES,
)
@pytest.mark.parametrize("dtype", tu.selected_cases(_BACKWARD_DTYPES))
def test_slow_conv_dilated2d_backward(
    inp_shape, weight_shape, kernel_size, stride, padding, dilation, dtype
):
    inp = _INPUT_SCALE * tu.make_input(dtype, inp_shape, ["-1", "1"]).requires_grad_()
    weight = (
        _INPUT_SCALE * tu.make_input(dtype, weight_shape, ["-1", "1"]).requires_grad_()
    )
    bias = (
        _INPUT_SCALE
        * tu.make_input(dtype, (weight_shape[0],), ["-1", "1"]).requires_grad_()
    )

    ref_inp = _reference_input(inp, True)
    ref_weight = _reference_input(weight, True)
    ref_bias = _reference_input(bias, True)

    ref_fwd = torch.ops.aten.slow_conv_dilated2d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    )
    ref_in_grad, ref_weight_grad, ref_bias_grad = torch.autograd.grad(
        ref_fwd.sum(), (ref_inp, ref_weight, ref_bias)
    )

    res_out = flag_gems.slow_conv_dilated2d(
        inp, weight, kernel_size, bias, stride, padding, dilation
    )
    _assert_close(_comparison_output(res_out, inp.device), ref_fwd.to(dtype), dtype)

    # grad_input contracts over C_out x kH x kW; grad_weight/grad_bias contract
    # over N x H_out x W_out. Scale atol by those counts.
    in_reduce_dim = weight_shape[0] * weight_shape[2] * weight_shape[3]
    out_reduce_dim = inp_shape[0] * ref_fwd.shape[2] * ref_fwd.shape[3]

    assert res_out.requires_grad
    res_in_grad, res_weight_grad, res_bias_grad = torch.autograd.grad(
        res_out.sum(), (inp, weight, bias)
    )
    utils.gems_assert_close(
        _comparison_output(res_in_grad, inp.device),
        ref_in_grad,
        dtype,
        reduce_dim=in_reduce_dim,
    )
    utils.gems_assert_close(
        _comparison_output(res_weight_grad, inp.device),
        ref_weight_grad,
        dtype,
        reduce_dim=out_reduce_dim,
    )
    utils.gems_assert_close(
        _comparison_output(res_bias_grad, inp.device),
        ref_bias_grad,
        dtype,
        reduce_dim=out_reduce_dim,
    )


@pytest.mark.slow_conv_dilated2d_nan_inf
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(utils.ALL_FLOAT_DTYPES))
)
@pytest.mark.parametrize("special_arg", ["inp", "weight", "bias"])
def test_slow_conv_dilated2d_nan_inf(dtype, scenario, special_arg):
    inp = torch.ones((1, 2, 4, 4), dtype=dtype, device=flag_gems.device)
    weight = torch.ones((5, 2, 1, 1), dtype=dtype, device=flag_gems.device)
    bias = torch.ones((5,), dtype=dtype, device=flag_gems.device)

    specials = tu.make_special_input(dtype, scenario)
    target = {"inp": inp, "weight": weight, "bias": bias}[special_arg]
    target.flatten()[: specials.numel()] = specials

    kernel_size = (1, 1)
    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)

    ref_inp = _reference_input(inp, True)
    ref_weight = _reference_input(weight, True)
    ref_bias = _reference_input(bias, True)
    ref_out = torch.ops.aten.slow_conv_dilated2d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    ).to(dtype)

    res_out = flag_gems.slow_conv_dilated2d(
        inp, weight, kernel_size, bias, stride, padding, dilation
    )

    tu.assert_result_close(_comparison_output(res_out, inp.device), ref_out)


@pytest.mark.slow_conv_dilated2d_negative
@pytest.mark.parametrize("case", _INVALID_SLOW_CONV_DILATED2D_CASES)
def test_slow_conv_dilated2d_negative(case):
    inp_shape, weight_shape, kernel_size, stride, padding, dilation = case
    inp, weight, bias = _make_conv_inputs(inp_shape, weight_shape, True, torch.float32)

    # The reference rejects the inconsistent configuration...
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated2d(
            inp.cpu() if _MUSA else inp,
            weight.cpu() if _MUSA else weight,
            kernel_size,
            bias.cpu() if _MUSA else bias,
            stride,
            padding,
            dilation,
        )

    # The candidate must reject the same invalid arguments.
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.slow_conv_dilated2d(
            inp, weight, kernel_size, bias, stride, padding, dilation
        )


@pytest.mark.slow_conv_dilated2d_negative
def test_slow_conv_dilated2d_rejects_kernel_larger_than_input():
    inp, weight, _ = _make_conv_inputs((1, 3, 2, 2), (4, 3, 3, 3), False, torch.float32)
    args = ([3, 3], None, [1, 1], [0, 0], [1, 1])
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated2d(
            inp.cpu() if _MUSA else inp, weight.cpu() if _MUSA else weight, *args
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.slow_conv_dilated2d(inp, weight, *args)


@pytest.mark.slow_conv_dilated2d_negative
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test_slow_conv_dilated2d_rejects_unsupported_dtype(dtype):
    inp = torch.ones((2, 3, 5, 5), dtype=dtype, device=flag_gems.device)
    weight = torch.ones((4, 3, 3, 3), dtype=dtype, device=flag_gems.device)
    bias = torch.ones((4,), dtype=dtype, device=flag_gems.device)

    if not (_MUSA and dtype in {torch.int8, torch.uint8, torch.int32, torch.int64}):
        with pytest.raises(RuntimeError):
            torch.ops.aten.slow_conv_dilated2d(
                inp.cpu() if _MUSA else inp,
                weight.cpu() if _MUSA else weight,
                (3, 3),
                bias.cpu() if _MUSA else bias,
                (1, 1),
                (0, 0),
                (1, 1),
            )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.slow_conv_dilated2d(inp, weight, (3, 3), bias, (1, 1), (0, 0), (1, 1))


@pytest.mark.slow_conv_dilated2d
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_slow_conv_dilated2d_projection(dtype):
    """1 x 1 unit-stride unpadded kernel: the specialised plain-GEMM path."""
    inp = torch.randn((3, 5, 6, 7), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((8, 5, 1, 1), dtype=dtype, device=flag_gems.device)
    ref_inp = _reference_input(inp, True)
    ref_weight = _reference_input(weight, True)

    ref_out = torch.ops.aten.slow_conv_dilated2d(
        ref_inp, ref_weight, [1, 1], None, [1, 1], [0, 0], [1, 1]
    )
    res_out = flag_gems.slow_conv_dilated2d(
        inp, weight, [1, 1], None, [1, 1], [0, 0], [1, 1]
    )
    _assert_close(_comparison_output(res_out, inp.device), ref_out, dtype)


@pytest.mark.slow_conv_dilated2d_out
@pytest.mark.slow_conv_dilated2d
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize(
    "buffer_kind",
    [
        "empty_buffer",
        "wrong_shape",
        "noncontiguous",
    ],
)
def test_slow_conv_dilated2d_master_output_layout(dtype, buffer_kind):
    """``out=`` writes in place, resizes like ATen, and keeps a non-contiguous layout.

    ATen's ``.out`` overload resizes the destination to the required shape, so
    an empty or wrongly shaped buffer is legal.  The kernels consume the
    destination strides, so a non-contiguous buffer must keep its layout.
    """
    inp = torch.randn((2, 3, 9, 9), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((4, 3, 3, 3), dtype=dtype, device=flag_gems.device)
    ref_inp = _reference_input(inp, True)
    ref_weight = _reference_input(weight, True)

    ref_out = torch.ops.aten.slow_conv_dilated2d(
        ref_inp, ref_weight, [3, 3], None, [1, 1], [1, 1], [2, 2]
    )

    if buffer_kind == "empty_buffer":
        out_buf = torch.empty(0, dtype=dtype, device=flag_gems.device)
    elif buffer_kind == "wrong_shape":
        out_buf = torch.empty((1, 1, 1, 1), dtype=dtype, device=flag_gems.device)
    else:
        out_buf = torch.empty(
            (2, 4, 7, 7), dtype=dtype, device=flag_gems.device
        ).transpose(2, 3)
        assert not out_buf.is_contiguous()
    expected_stride = out_buf.stride() if buffer_kind == "noncontiguous" else None

    res_out = flag_gems.slow_conv_dilated2d(
        inp, weight, [3, 3], None, [1, 1], [1, 1], [2, 2], out=out_buf
    )

    assert res_out is out_buf
    assert res_out.shape == (2, 4, 7, 7)
    if expected_stride is not None:
        assert res_out.stride() == expected_stride
    _assert_close(_comparison_output(res_out, inp.device), ref_out, dtype)


@pytest.mark.slow_conv_dilated2d
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_slow_conv_dilated2d_unbatched(dtype):
    """3-D input matches ATen and the 4-D call's only batch element."""
    x3 = torch.randn((3, 8, 8), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((4, 3, 3, 3), dtype=dtype, device=flag_gems.device)
    ref_x3 = _reference_input(x3, True)
    ref_weight = _reference_input(weight, True)

    ref_out = torch.ops.aten.slow_conv_dilated2d(
        ref_x3, ref_weight, [3, 3], None, [1, 1], [0, 0], [1, 1]
    )
    assert ref_out.shape == (4, 6, 6)

    res_out = flag_gems.slow_conv_dilated2d(
        x3, weight, [3, 3], None, [1, 1], [0, 0], [1, 1]
    )
    assert res_out.shape == (4, 6, 6)
    _assert_close(_comparison_output(res_out, x3.device), ref_out, dtype)

    res_batched = flag_gems.slow_conv_dilated2d(
        x3.unsqueeze(0), weight, [3, 3], None, [1, 1], [0, 0], [1, 1]
    )
    # Same kernel, same batch axis: the unbatched result is bit-identical to the
    # batch-0 slice, not merely close.
    assert torch.equal(res_out, res_batched[0])

    # A 3-D ``out=`` buffer is written in place and returned unchanged, so the
    # unbatched destination keeps the caller's rank rather than gaining a batch
    # axis internally.
    out_buf = torch.empty((4, 6, 6), dtype=dtype, device=flag_gems.device)
    out_res = flag_gems.slow_conv_dilated2d(
        x3, weight, [3, 3], None, [1, 1], [0, 0], [1, 1], out=out_buf
    )
    assert out_res is out_buf
    assert out_buf.shape == (4, 6, 6)
    _assert_close(_comparison_output(out_res, x3.device), ref_out, dtype)


@pytest.mark.slow_conv_dilated2d
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_slow_conv_dilated2d_strided_input(dtype):
    """A transposed spatial plane exercises the non-unit input strides."""
    inp = torch.randn((2, 3, 8, 9), dtype=dtype, device=flag_gems.device).transpose(
        2, 3
    )
    weight = torch.randn((4, 3, 3, 3), dtype=dtype, device=flag_gems.device)
    assert not inp.is_contiguous()
    ref_inp = _reference_input(inp, True)
    ref_weight = _reference_input(weight, True)

    ref_out = torch.ops.aten.slow_conv_dilated2d(
        ref_inp, ref_weight, [3, 3], None, [1, 1], [1, 1], [1, 1]
    )
    res_out = flag_gems.slow_conv_dilated2d(
        inp, weight, [3, 3], None, [1, 1], [1, 1], [1, 1]
    )
    _assert_close(_comparison_output(res_out, inp.device), ref_out, dtype)


@pytest.mark.slow_conv_dilated2d
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_slow_conv_dilated2d_channels_last(dtype):
    """Channels-last input keeps its NHWC strides and is still correct."""
    inp = torch.randn((2, 3, 8, 8), dtype=dtype, device=flag_gems.device).to(
        memory_format=torch.channels_last
    )
    weight = torch.randn((4, 3, 3, 3), dtype=dtype, device=flag_gems.device)
    assert inp.is_contiguous(memory_format=torch.channels_last)
    ref_inp = _reference_input(inp, True)
    ref_weight = _reference_input(weight, True)

    ref_out = torch.ops.aten.slow_conv_dilated2d(
        ref_inp, ref_weight, [3, 3], None, [1, 1], [1, 1], [1, 1]
    )
    res_out = flag_gems.slow_conv_dilated2d(
        inp, weight, [3, 3], None, [1, 1], [1, 1], [1, 1]
    )
    _assert_close(_comparison_output(res_out, inp.device), ref_out, dtype)


@pytest.mark.slow_conv_dilated2d
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_slow_conv_dilated2d_empty(dtype):
    """Empty batch and empty output-channel sets produce empty results.

    ``c_in == 0`` is deliberately *not* covered: ATen's own reference path has
    no comparable behaviour there, so there is nothing to compare against.
    """
    weight = torch.randn((4, 3, 3, 3), dtype=dtype, device=flag_gems.device)

    inp = torch.randn((0, 3, 8, 8), dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.slow_conv_dilated2d(inp, weight, [3, 3], None, [1, 1], [1, 1])
    assert res_out.shape == (0, 4, 8, 8)
    assert res_out.device == inp.device
    assert res_out.numel() == 0

    inp = torch.randn((2, 3, 8, 8), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((0, 3, 3, 3), dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.slow_conv_dilated2d(inp, weight, [3, 3], None, [1, 1], [1, 1])
    assert res_out.shape == (2, 0, 8, 8)
    assert res_out.device == inp.device
    assert res_out.numel() == 0


@pytest.mark.slow_conv_dilated2d
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_slow_conv_dilated2d_non_square_geometry(dtype):
    """Non-square kernels and per-axis stride/padding/dilation are not transposed."""
    inp = torch.randn((2, 3, 9, 11), dtype=dtype, device=flag_gems.device)
    ref_inp = _reference_input(inp, True)

    for (k_h, k_w), stride, padding, dilation in [
        ((1, 2), [1, 1], [1, 1], [1, 1]),
        ((3, 1), [1, 1], [1, 1], [1, 1]),
        ((2, 3), [1, 1], [1, 1], [1, 1]),
        ((3, 3), [1, 2], [0, 0], [1, 1]),
        ((3, 3), [2, 1], [1, 0], [1, 2]),
    ]:
        weight = torch.randn((4, 3, k_h, k_w), dtype=dtype, device=flag_gems.device)
        ref_weight = _reference_input(weight, True)
        ref_out = torch.ops.aten.slow_conv_dilated2d(
            ref_inp, ref_weight, [k_h, k_w], None, stride, padding, dilation
        )
        res_out = flag_gems.slow_conv_dilated2d(
            inp, weight, [k_h, k_w], None, stride, padding, dilation
        )
        assert res_out.shape == ref_out.shape
        _assert_close(_comparison_output(res_out, inp.device), ref_out, dtype)


@pytest.mark.slow_conv_dilated2d
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_slow_conv_dilated2d_negative_padding(dtype):
    """Negative padding crops the border instead of zero-extending it."""
    inp = torch.randn((2, 3, 8, 8), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((4, 3, 3, 3), dtype=dtype, device=flag_gems.device)
    ref_inp = _reference_input(inp, True)
    ref_weight = _reference_input(weight, True)

    ref_out = torch.ops.aten.slow_conv_dilated2d(
        ref_inp, ref_weight, [3, 3], None, [1, 1], [-1, -1], [1, 1]
    )
    res_out = flag_gems.slow_conv_dilated2d(
        inp, weight, [3, 3], None, [1, 1], [-1, -1], [1, 1]
    )

    assert res_out.shape == (2, 4, 4, 4)
    _assert_close(_comparison_output(res_out, inp.device), ref_out, dtype)


@pytest.fixture(autouse=True)
def full_precision():
    # Match native fp32 results against the fp64 reference without leaking flags.
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
