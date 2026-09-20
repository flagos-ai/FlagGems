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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Register underscore-prefixed pytest markers explicitly.
setattr(
    pytest.mark,
    "_slow_conv2d_backward",
    MarkDecorator(
        Mark("_slow_conv2d_backward", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# Test masked gradients and both output-buffer overloads of slow conv2d backward.
# Extreme ranges retain their bounds and use the original reference dtype to preserve overflow.
# Cases: (input shape, weight shape, kernel size, stride, padding).
SLOW_CONV2D_BACKWARD_CASES = tu.selected_cases(
    [
        ((16, 4, 8, 8), (4, 4, 3, 3), (3, 3), (1, 1), (0, 0)),
        ((8, 3, 16, 16), (8, 3, 3, 3), (3, 3), (1, 1), (1, 1)),
        ((32, 8, 8, 8), (32, 8, 2, 2), (2, 2), (2, 2), (0, 0)),
        ((32, 8, 8, 8), (32, 8, 2, 2), (2, 2), (1, 1), (1, 1)),
        ((4, 16, 4, 4), (16, 16, 1, 1), (1, 1), (1, 1), (0, 0)),
        ((4, 16, 4, 4), (16, 16, 1, 1), (1, 1), (2, 2), (0, 0)),
        ((2, 3, 9, 9), (4, 3, 3, 5), (3, 5), (1, 2), (1, 2)),
        ((2, 3, 4, 4), (5, 3, 3, 3), (3, 3), (1, 1), (0, 0)),
    ],
    quick=[
        ((1, 2, 5, 5), (2, 2, 3, 3), (3, 3), (1, 1), (1, 1)),
    ],
)

SLOW_CONV2D_VALUE_RANGES_CASES = SLOW_CONV2D_BACKWARD_CASES[:4]

_FULL_MASK = (True, True, True)

_MIXED_MASKS = [(True, False, True), (False, True, True)]

# Scale finite random inputs to limit low-precision reduction noise.
_INPUT_SCALE = 0.1

# Invalid (input, weight, kernel, stride, padding, grad_output) configurations.
_INVALID_SLOW_CONV2D_CASES = [
    # weight has wrong C_in vs input
    ((2, 3, 5, 5), (4, 5, 3, 3), (3, 3), (1, 1), (0, 0), (2, 4, 3, 3)),
    # weight spatial dims disagree with kernel_size -> grad_output H check fires
    ((2, 3, 5, 5), (4, 3, 4, 4), (3, 3), (1, 1), (0, 0), (2, 4, 2, 2)),
    # grad_output H_out inconsistent with (H, kernel, stride, padding)
    ((2, 3, 5, 5), (4, 3, 3, 3), (3, 3), (1, 1), (1, 1), (2, 4, 6, 6)),
    # grad_output has wrong C_out vs weight
    ((2, 3, 5, 5), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (2, 5, 3, 3)),
    # grad_output is not 4-D
    ((2, 3, 5, 5), (4, 3, 3, 3), (3, 3), (1, 1), (0, 0), (2, 4, 3)),
]

_NON_FLOAT_DTYPES = tu.selected_cases([torch.int32, torch.int64], quick=[torch.int32])

_SCALAR_PARAMS = ["kernel_size", "stride", "padding"]


def _make_inputs(
    inp_shape, weight_shape, kernel_size, stride, padding, dtype, value_range
):
    # Build input, weight and grad_output; keep extreme ranges unscaled.
    n_in, _, h_in, w_in = inp_shape
    out_c = weight_shape[0]
    k_h, k_w = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding
    h_out = (h_in + 2 * p_h - k_h) // s_h + 1
    w_out = (w_in + 2 * p_w - k_w) // s_w + 1
    inp = (1.0 if tu.is_extreme_range(value_range) else _INPUT_SCALE) * tu.make_input(
        dtype, inp_shape, value_range
    )
    weight = (
        1.0 if tu.is_extreme_range(value_range) else _INPUT_SCALE
    ) * tu.make_input(dtype, weight_shape, value_range)
    grad_output = (
        1.0 if tu.is_extreme_range(value_range) else _INPUT_SCALE
    ) * tu.make_input(dtype, (n_in, out_c, h_out, w_out), value_range)
    return inp, weight, grad_output


def _reference_output_mask(
    inp, weight, grad_output, kernel_size, stride, padding, mask, *, upcast=True
):
    ref_inp = tu.to_reference(inp, upcast)
    ref_weight = tu.to_reference(weight, upcast)
    ref_grad_output = tu.to_reference(grad_output, upcast)
    return torch.ops.aten._slow_conv2d_backward.output_mask(
        ref_grad_output,
        ref_inp,
        ref_weight,
        kernel_size,
        stride,
        padding,
        mask,
    )


def _reduction_dims(inp_shape, weight_shape, stride, padding):
    n_in, _, h_in, w_in = inp_shape
    out_c, _, k_h, k_w = weight_shape
    s_h, s_w = stride
    p_h, p_w = padding
    h_out = (h_in + 2 * p_h - k_h) // s_h + 1
    w_out = (w_in + 2 * p_w - k_w) // s_w + 1
    in_reduce_dim = out_c * k_h * k_w  # grad_input contracts over C_out x kH x kW
    out_reduce_dim = (
        n_in * h_out * w_out
    )  # grad_weight/bias contract over N x H_out x W_out
    return in_reduce_dim, out_reduce_dim


def _assert_grads_close(
    res, ref, in_reduce_dim, out_reduce_dim, dtype, equal_nan=False
):
    # Masked gradients must remain None; compare computed gradients using their reduction sizes.
    res_in_grad, res_weight_grad, res_bias_grad = res
    ref_in_grad, ref_weight_grad, ref_bias_grad = ref
    if ref_in_grad is None:
        assert res_in_grad is None
    else:
        utils.gems_assert_close(
            res_in_grad,
            ref_in_grad,
            dtype,
            reduce_dim=in_reduce_dim,
            equal_nan=equal_nan,
        )
    if ref_weight_grad is None:
        assert res_weight_grad is None
    else:
        utils.gems_assert_close(
            res_weight_grad,
            ref_weight_grad,
            dtype,
            reduce_dim=out_reduce_dim,
            equal_nan=equal_nan,
        )
    if ref_bias_grad is None:
        assert res_bias_grad is None
    else:
        utils.gems_assert_close(
            res_bias_grad,
            ref_bias_grad,
            dtype,
            reduce_dim=out_reduce_dim,
            equal_nan=equal_nan,
        )


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_VALUE_RANGES_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__slow_conv2d_backward_value_ranges(case, value_range, dtype):
    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, value_range
    )
    ref = _reference_output_mask(
        inp,
        weight,
        grad_output,
        kernel_size,
        stride,
        padding,
        _FULL_MASK,
        upcast=not tu.is_extreme_range(value_range),
    )

    res = flag_gems._slow_conv2d_backward(
        grad_output, inp, weight, kernel_size, stride, padding, _FULL_MASK
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype, equal_nan=True)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__slow_conv2d_backward_output_mask_full(case, dtype):
    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, _FULL_MASK
    )

    res = flag_gems._slow_conv2d_backward(
        grad_output, inp, weight, kernel_size, stride, padding, _FULL_MASK
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__slow_conv2d_backward_grad_input_only(case, dtype):
    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    mask = (True, False, False)
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, mask
    )

    res = flag_gems._slow_conv2d_backward(
        grad_output, inp, weight, kernel_size, stride, padding, mask
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__slow_conv2d_backward_grad_weight_only(case, dtype):
    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    mask = (False, True, False)
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, mask
    )

    res = flag_gems._slow_conv2d_backward(
        grad_output, inp, weight, kernel_size, stride, padding, mask
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__slow_conv2d_backward_grad_bias_only(case, dtype):
    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    mask = (False, False, True)
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, mask
    )

    res = flag_gems._slow_conv2d_backward(
        grad_output, inp, weight, kernel_size, stride, padding, mask
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES[:2])
@pytest.mark.parametrize("mask", _MIXED_MASKS)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__slow_conv2d_backward_mixed_mask(case, mask, dtype):
    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, mask
    )

    res = flag_gems._slow_conv2d_backward(
        grad_output, inp, weight, kernel_size, stride, padding, mask
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__slow_conv2d_backward_grad_input_out(case, dtype):
    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, _FULL_MASK
    )

    ref_grad_input = torch.full_like(inp, 7.0)
    ref_grad_weight = torch.full_like(weight, 7.0)
    ref_grad_bias = torch.full(
        (weight_shape[0],), 7.0, dtype=dtype, device=flag_gems.device
    )
    ref_ret = torch.ops.aten._slow_conv2d_backward.grad_input(
        grad_output,
        inp,
        weight,
        kernel_size,
        stride,
        padding,
        grad_input=ref_grad_input,
        grad_weight=ref_grad_weight,
        grad_bias=ref_grad_bias,
    )

    res_grad_input = torch.full_like(inp, 7.0)
    res_grad_weight = torch.full_like(weight, 7.0)
    res_grad_bias = torch.full(
        (weight_shape[0],), 7.0, dtype=dtype, device=flag_gems.device
    )
    res = flag_gems._slow_conv2d_backward(
        grad_output,
        inp,
        weight,
        kernel_size,
        stride,
        padding,
        grad_input=res_grad_input,
        grad_weight=res_grad_weight,
        grad_bias=res_grad_bias,
    )
    assert res[0] is res_grad_input
    assert res[1] is res_grad_weight
    assert res[2] is res_grad_bias

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)
    _assert_grads_close(ref_ret, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__slow_conv2d_backward_output_mask_out(case, dtype):
    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, _FULL_MASK
    )

    ref_out0 = torch.full_like(inp, 7.0)
    ref_out1 = torch.full_like(weight, 7.0)
    ref_out2 = torch.full((weight_shape[0],), 7.0, dtype=dtype, device=flag_gems.device)
    ref_ret = torch.ops.aten._slow_conv2d_backward.output_mask_out(
        grad_output,
        inp,
        weight,
        kernel_size,
        stride,
        padding,
        _FULL_MASK,
        out0=ref_out0,
        out1=ref_out1,
        out2=ref_out2,
    )

    res_out0 = torch.full_like(inp, 7.0)
    res_out1 = torch.full_like(weight, 7.0)
    res_out2 = torch.full((weight_shape[0],), 7.0, dtype=dtype, device=flag_gems.device)
    res = flag_gems._slow_conv2d_backward(
        grad_output,
        inp,
        weight,
        kernel_size,
        stride,
        padding,
        _FULL_MASK,
        out0=res_out0,
        out1=res_out1,
        out2=res_out2,
    )
    assert res[0] is res_out0
    assert res[1] is res_out1
    assert res[2] is res_out2

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)
    _assert_grads_close(ref_ret, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", SLOW_CONV2D_BACKWARD_CASES[:3])
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test__slow_conv2d_backward_backward(case, dtype):
    inp_shape, weight_shape, kernel_size, stride, padding = case
    inp, weight, grad_output = _make_inputs(
        inp_shape, weight_shape, kernel_size, stride, padding, dtype, ["-1", "1"]
    )
    bias = _INPUT_SCALE * tu.make_input(dtype, (weight_shape[0],), ["-1", "1"])

    # Differentiate sum(forward * grad_output) w.r.t. (input, weight, bias)
    # through autograd on the fp64 upcast reference and compare against the
    # candidate's three gradients. This validates the candidate against the true
    # gradient through an independent computation path.
    ref_inp = tu.to_reference(inp, True).requires_grad_(True)
    ref_weight = tu.to_reference(weight, True).requires_grad_(True)
    ref_bias = tu.to_reference(bias, True).requires_grad_(True)
    ref_grad_output = tu.to_reference(grad_output, True)

    fwd = torch.ops.aten._slow_conv2d_forward(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding
    )
    ref = torch.autograd.grad(
        (fwd * ref_grad_output).sum(),
        (ref_inp, ref_weight, ref_bias),
        allow_unused=True,
    )

    res = flag_gems._slow_conv2d_backward(
        grad_output, inp, weight, kernel_size, stride, padding, _FULL_MASK
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        inp_shape, weight_shape, stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(utils.ALL_FLOAT_DTYPES))
)
@pytest.mark.parametrize("special_arg", ["inp", "weight", "grad_output"])
def test__slow_conv2d_backward_nan_inf(dtype, scenario, special_arg):
    inp = torch.ones((2, 3, 5, 5), dtype=dtype, device=flag_gems.device)
    weight = torch.ones((2, 3, 3, 3), dtype=dtype, device=flag_gems.device)
    grad_output = torch.ones((2, 2, 5, 5), dtype=dtype, device=flag_gems.device)

    specials = tu.make_special_input(dtype, scenario)
    target = {"inp": inp, "weight": weight, "grad_output": grad_output}[special_arg]
    target.flatten()[: specials.numel()] = specials

    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    ref = _reference_output_mask(
        inp, weight, grad_output, kernel_size, stride, padding, _FULL_MASK
    )

    res = flag_gems._slow_conv2d_backward(
        grad_output, inp, weight, kernel_size, stride, padding, _FULL_MASK
    )

    in_reduce_dim, out_reduce_dim = _reduction_dims(
        (2, 3, 5, 5), (2, 3, 3, 3), stride, padding
    )
    _assert_grads_close(res, ref, in_reduce_dim, out_reduce_dim, dtype, equal_nan=True)


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("case", _INVALID_SLOW_CONV2D_CASES)
def test__slow_conv2d_backward_negative_invalid_config(case):
    inp_shape, weight_shape, kernel_size, stride, padding, grad_output_shape = case
    inp = _INPUT_SCALE * tu.make_input(torch.float32, inp_shape, ["-1", "1"])
    weight = _INPUT_SCALE * tu.make_input(torch.float32, weight_shape, ["-1", "1"])
    grad_output = _INPUT_SCALE * tu.make_input(
        torch.float32, grad_output_shape, ["-1", "1"]
    )

    # The reference rejects the inconsistent configuration...
    with pytest.raises(RuntimeError):
        torch.ops.aten._slow_conv2d_backward.output_mask(
            grad_output, inp, weight, kernel_size, stride, padding, _FULL_MASK
        )

    # The candidate must reject the same invalid arguments.
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._slow_conv2d_backward(
            grad_output, inp, weight, kernel_size, stride, padding, _FULL_MASK
        )


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("dtype", _NON_FLOAT_DTYPES)
def test__slow_conv2d_backward_negative_non_float_dtype(dtype):
    inp = tu.make_input(dtype, (2, 3, 5, 5), ["0", "1"])
    weight = tu.make_input(dtype, (4, 3, 3, 3), ["0", "1"])
    grad_output = tu.make_input(dtype, (2, 4, 3, 3), ["0", "1"])

    with pytest.raises(RuntimeError):
        torch.ops.aten._slow_conv2d_backward.output_mask(
            grad_output, inp, weight, (3, 3), (1, 1), (0, 0), _FULL_MASK
        )

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._slow_conv2d_backward(
            grad_output, inp, weight, (3, 3), (1, 1), (0, 0), _FULL_MASK
        )


@pytest.mark._slow_conv2d_backward
def test__slow_conv2d_backward_negative_non_4d_grad_output():
    inp = _INPUT_SCALE * tu.make_input(torch.float32, (2, 3, 5, 5), ["-1", "1"])
    weight = _INPUT_SCALE * tu.make_input(torch.float32, (4, 3, 3, 3), ["-1", "1"])
    grad_output = _INPUT_SCALE * tu.make_input(torch.float32, (2, 4, 3), ["-1", "1"])

    with pytest.raises(RuntimeError):
        torch.ops.aten._slow_conv2d_backward.output_mask(
            grad_output, inp, weight, (3, 3), (1, 1), (0, 0), _FULL_MASK
        )

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._slow_conv2d_backward(
            grad_output, inp, weight, (3, 3), (1, 1), (0, 0), _FULL_MASK
        )


@pytest.mark._slow_conv2d_backward
@pytest.mark.parametrize("scalar_param", _SCALAR_PARAMS)
def test__slow_conv2d_backward_negative_scalar_param(scalar_param):
    inp_shape, weight_shape, kernel_size, stride, padding = SLOW_CONV2D_BACKWARD_CASES[
        0
    ]
    inp = _INPUT_SCALE * tu.make_input(torch.float32, inp_shape, ["-1", "1"])
    weight = _INPUT_SCALE * tu.make_input(torch.float32, weight_shape, ["-1", "1"])
    n_in, _, h_in, w_in = inp_shape
    out_c = weight_shape[0]
    k_h, k_w = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding
    h_out = (h_in + 2 * p_h - k_h) // s_h + 1
    w_out = (w_in + 2 * p_w - k_w) // s_w + 1
    grad_output = _INPUT_SCALE * tu.make_input(
        torch.float32, (n_in, out_c, h_out, w_out), ["-1", "1"]
    )

    args = [kernel_size, stride, padding]
    args[_SCALAR_PARAMS.index(scalar_param)] = 3

    with pytest.raises(RuntimeError):
        torch.ops.aten._slow_conv2d_backward.output_mask(
            grad_output, inp, weight, args[0], args[1], args[2], _FULL_MASK
        )

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._slow_conv2d_backward(
            grad_output, inp, weight, args[0], args[1], args[2], _FULL_MASK
        )


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
