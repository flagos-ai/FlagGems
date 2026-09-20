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

# Test transposed conv2d; weight dimensions are (C_in, C_out, kH, kW).
# Extreme ranges retain their bounds and use the original reference dtype to preserve overflow.
# Cases: (input shape, weight shape, kernel size, stride, padding, output padding, dilation).
SLOW_CONV_TRANSPOSE2D_CASES = tu.selected_cases(
    [
        (
            (1, 2, 5, 5),
            (2, 3, 3, 3),
            (3, 3),
            (1, 1),
            (0, 0),
            (0, 0),
            (1, 1),
        ),
        (
            (2, 4, 8, 8),
            (4, 6, 3, 3),
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            (1, 1),
        ),
        (
            (2, 8, 12, 12),
            (8, 8, 3, 3),
            (3, 3),
            (2, 1),
            (1, 2),
            (1, 0),
            (2, 1),
        ),
        (
            (1, 3, 7, 9),
            (3, 4, 3, 5),
            (3, 5),
            (1, 1),
            (1, 2),
            (0, 0),
            (1, 1),
        ),
        (
            (2, 4, 6, 6),
            (4, 8, 2, 2),
            (2, 2),
            (1, 1),
            (1, 1),
            (1, 1),
            (2, 2),
        ),
        (
            (1, 16, 8, 8),
            (16, 8, 1, 1),
            (1, 1),
            (1, 1),
            (0, 0),
            (0, 0),
            (1, 1),
        ),
        (
            (2, 3, 5, 4),
            (3, 5, 3, 3),
            (3, 3),
            (2, 2),
            (0, 1),
            (1, 0),
            (1, 1),
        ),
        (
            (1, 4, 10, 10),
            (4, 8, 5, 5),
            (5, 5),
            (2, 2),
            (2, 2),
            (1, 1),
            (1, 1),
        ),
        (
            (2, 8, 16, 16),
            (8, 16, 3, 3),
            (3, 3),
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
        ),
        (
            (1, 2, 4, 4),
            (2, 3, 2, 2),
            (2, 2),
            (1, 1),
            (0, 0),
            (0, 0),
            (2, 2),
        ),
        (
            (1, 2, 5, 5),
            (2, 3, 3, 3),
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            (2, 2),
        ),
        (
            (2, 4, 5),
            (2, 3, 3, 3),
            (3, 3),
            (2, 1),
            (1, 0),
            (1, 0),
            (1, 1),
        ),
    ],
    quick=[
        (
            (1, 2, 5, 5),
            (2, 3, 3, 3),
            (3, 3),
            (1, 1),
            (0, 0),
            (0, 0),
            (1, 1),
        ),
    ],
)

BIASES = tu.selected_cases([True, False], quick=[True])

# Scale finite random inputs to limit low-precision reduction noise.
_INPUT_SCALE = 0.1

_BACKWARD_CASES = SLOW_CONV_TRANSPOSE2D_CASES[:3]

_GEMS_ERRORS = (TypeError, ValueError, RuntimeError, AttributeError)


def _conv_output_shape(
    inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation
):
    h_in = inp_shape[-2]
    w_in = inp_shape[-1]
    out_c = weight_shape[1]
    h_out = (
        (h_in - 1) * stride[0]
        - 2 * padding[0]
        + dilation[0] * (kernel_size[0] - 1)
        + output_padding[0]
        + 1
    )
    w_out = (
        (w_in - 1) * stride[1]
        - 2 * padding[1]
        + dilation[1] * (kernel_size[1] - 1)
        + output_padding[1]
        + 1
    )
    if len(inp_shape) == 3:
        return (out_c, h_out, w_out)
    return (inp_shape[0], out_c, h_out, w_out)


def _make_conv_inputs(inp_shape, weight_shape, with_bias, dtype, value_range):
    inp = tu.make_input(dtype, inp_shape, value_range)
    # Transposed conv weight is (C_in, C_out, kH, kW): the bias length is the
    # second (output-channel) dim.
    weight = tu.make_input(dtype, weight_shape, value_range)
    if with_bias:
        bias = tu.make_input(dtype, (weight_shape[1],), value_range)
    else:
        bias = None
    return inp, weight, bias


def _assert_close(res_out, ref_out, dtype, equal_nan=False):
    # Supplement dtype-relative tolerance for the fp64 reference with absolute reduction-error bounds.
    if dtype == torch.bfloat16:
        atol = 4e-1
    elif dtype == torch.float16:
        atol = 4e-2
    else:
        atol = 1e-4
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol, equal_nan=equal_nan)


@pytest.mark.slow_conv_transpose2d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation",
    SLOW_CONV_TRANSPOSE2D_CASES,
)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_transpose2d(
    inp_shape,
    weight_shape,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
    dtype,
    bias,
):
    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, bias, dtype, ["-1", "1"]
    )
    ref_inp = tu.to_reference(inp, True)
    ref_weight = tu.to_reference(weight, True)
    ref_bias = tu.to_reference(bias_t, True)

    ref_out = torch.ops.aten.slow_conv_transpose2d(
        ref_inp,
        ref_weight,
        kernel_size,
        ref_bias,
        stride,
        padding,
        output_padding,
        dilation,
    ).to(dtype)

    # Native CUDA adds a leading batch dimension to the unbatched input.
    # Independent reference storage keeps that change local to each call.
    res_out = flag_gems.slow_conv_transpose2d(
        inp, weight, kernel_size, bias_t, stride, padding, output_padding, dilation
    )

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    _assert_close(res_out, ref_out, dtype)


@pytest.mark.slow_conv_transpose2d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation",
    SLOW_CONV_TRANSPOSE2D_CASES[:2],
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_slow_conv_transpose2d_value_ranges(
    inp_shape,
    weight_shape,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
    value_range,
    dtype,
):
    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, True, dtype, value_range
    )
    ref_inp = tu.to_reference(inp, not tu.is_extreme_range(value_range))
    ref_weight = tu.to_reference(weight, not tu.is_extreme_range(value_range))
    ref_bias = tu.to_reference(bias_t, not tu.is_extreme_range(value_range))

    ref_out = torch.ops.aten.slow_conv_transpose2d(
        ref_inp,
        ref_weight,
        kernel_size,
        ref_bias,
        stride,
        padding,
        output_padding,
        dilation,
    ).to(dtype)

    res_out = flag_gems.slow_conv_transpose2d(
        inp, weight, kernel_size, bias_t, stride, padding, output_padding, dilation
    )

    _assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.slow_conv_transpose2d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation",
    _BACKWARD_CASES,
)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("bias", tu.selected_cases(BIASES))
def test_slow_conv_transpose2d_backward(
    inp_shape,
    weight_shape,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
    dtype,
    bias,
):
    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, bias, dtype, ["-1", "1"]
    )
    inp = (_INPUT_SCALE * inp).requires_grad_()
    weight = (_INPUT_SCALE * weight).requires_grad_()
    if bias_t is not None:
        bias_t = (_INPUT_SCALE * bias_t).requires_grad_()

    out_shape = _conv_output_shape(
        inp_shape,
        weight_shape,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
    )
    grad_out = tu.make_input(dtype, out_shape, ["-1", "1"])

    ref_inp = tu.to_reference(inp, True)
    ref_weight = tu.to_reference(weight, True)
    ref_bias = tu.to_reference(bias_t, True)
    ref_grad_out = tu.to_reference(grad_out, True)
    ref_out = torch.ops.aten.slow_conv_transpose2d(
        ref_inp,
        ref_weight,
        kernel_size,
        ref_bias,
        stride,
        padding,
        output_padding,
        dilation,
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

    res_out = flag_gems.slow_conv_transpose2d(
        inp, weight, kernel_size, bias_t, stride, padding, output_padding, dilation
    )
    _assert_close(res_out, ref_out.to(dtype), dtype)

    assert res_out.requires_grad, "candidate must preserve autograd"
    if bias_t is None:
        res_gi, res_gw = torch.autograd.grad(res_out, (inp, weight), grad_out)
        res_gb = None
    else:
        res_gi, res_gw, res_gb = torch.autograd.grad(
            res_out, (inp, weight, bias_t), grad_out
        )
    # grad_input reduces over C_out*kH*kW terms, grad_weight/grad_bias over
    # N*H_out*W_out terms; the fp64 reference is exact for the scaled inputs, so
    # the candidate only needs to match the native precision.
    in_reduce_dim = weight_shape[1] * weight_shape[2] * weight_shape[3]
    out_reduce_dim = inp_shape[0] * out_shape[2] * out_shape[3]
    for res_g, ref_g, reduce_dim in zip(
        (res_gi, res_gw, res_gb),
        (ref_gi, ref_gw, ref_gb),
        (in_reduce_dim, out_reduce_dim, out_reduce_dim),
    ):
        if ref_g is None:
            assert res_g is None
        else:
            utils.gems_assert_close(
                res_g, ref_g.to(dtype), dtype, reduce_dim=reduce_dim
            )


@pytest.mark.slow_conv_transpose2d
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(utils.ALL_FLOAT_DTYPES))
)
@pytest.mark.parametrize("special_arg", ["inp", "weight", "bias"])
def test_slow_conv_transpose2d_nan_inf(dtype, scenario, special_arg):
    (
        inp_shape,
        weight_shape,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
    ) = ((1, 2, 5, 5), (2, 5, 3, 3), (3, 3), (1, 1), (0, 0), (0, 0), (1, 1))
    inp = torch.ones(inp_shape, dtype=dtype, device=flag_gems.device)
    weight = torch.ones(weight_shape, dtype=dtype, device=flag_gems.device)
    bias = torch.ones((weight_shape[1],), dtype=dtype, device=flag_gems.device)

    specials = tu.make_special_input(dtype, scenario)
    target = {"inp": inp, "weight": weight, "bias": bias}[special_arg]
    target.flatten()[: specials.numel()] = specials

    ref_inp = tu.to_reference(inp, True)
    ref_weight = tu.to_reference(weight, True)
    ref_bias = tu.to_reference(bias, True)
    ref_out = torch.ops.aten.slow_conv_transpose2d(
        ref_inp,
        ref_weight,
        kernel_size,
        ref_bias,
        stride,
        padding,
        output_padding,
        dilation,
    ).to(dtype)

    res_out = flag_gems.slow_conv_transpose2d(
        inp, weight, kernel_size, bias, stride, padding, output_padding, dilation
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.slow_conv_transpose2d_out
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, output_padding, dilation",
    SLOW_CONV_TRANSPOSE2D_CASES,
)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_transpose2d_out(
    inp_shape,
    weight_shape,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
    dtype,
    bias,
):
    inp, weight, bias_t = _make_conv_inputs(
        inp_shape, weight_shape, bias, dtype, ["-1", "1"]
    )
    ref_inp = tu.to_reference(inp, True)
    ref_weight = tu.to_reference(weight, True)
    ref_bias = tu.to_reference(bias_t, True)

    # The native .out overload is callable on this backend (probed), so call it
    # directly for the reference -- never simulate it with default()+copy_.
    ref_full = torch.ops.aten.slow_conv_transpose2d(
        tu.to_reference(ref_inp),
        ref_weight,
        kernel_size,
        ref_bias,
        stride,
        padding,
        output_padding,
        dilation,
    )
    ref_out = torch.empty_like(ref_full)
    ref_ret = torch.ops.aten.slow_conv_transpose2d.out(
        ref_inp,
        ref_weight,
        kernel_size,
        ref_bias,
        stride,
        padding,
        output_padding,
        dilation,
        out=ref_out,
    )

    out = torch.empty(ref_full.shape, dtype=dtype, device=flag_gems.device)
    res_ret = flag_gems.slow_conv_transpose2d(
        inp,
        weight,
        kernel_size,
        bias_t,
        stride,
        padding,
        output_padding,
        dilation,
        out=out,
    )
    assert res_ret is out
    _assert_close(res_ret, ref_ret, dtype)


@pytest.mark.slow_conv_transpose2d
def test_slow_conv_transpose2d_rejects_channel_mismatch():
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (3, 3, 3, 3), ["-1", "1"])
    args = (inp, weight, (3, 3), None, (1, 1), (0, 0), (0, 0), (1, 1))
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose2d(*args)
    with pytest.raises(_GEMS_ERRORS):
        flag_gems.slow_conv_transpose2d(*args)


@pytest.mark.slow_conv_transpose2d
@pytest.mark.parametrize("bad_shape", [(2, 5), (1, 2, 5, 5, 5)])
def test_slow_conv_transpose2d_rejects_invalid_input_rank(bad_shape):
    inp = tu.make_input(torch.float32, bad_shape, ["-1", "1"])
    weight = tu.make_input(torch.float32, (2, 3, 3, 3), ["-1", "1"])
    args = (inp, weight, (3, 3), None, (1, 1), (0, 0), (0, 0), (1, 1))
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose2d(*args)
    with pytest.raises(_GEMS_ERRORS):
        flag_gems.slow_conv_transpose2d(*args)


@pytest.mark.slow_conv_transpose2d
def test_slow_conv_transpose2d_rejects_invalid_weight_rank():
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (2, 3, 3), ["-1", "1"])
    args = (inp, weight, (3, 3), None, (1, 1), (0, 0), (0, 0), (1, 1))
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose2d(*args)
    with pytest.raises(_GEMS_ERRORS):
        flag_gems.slow_conv_transpose2d(*args)


@pytest.mark.slow_conv_transpose2d
@pytest.mark.parametrize(
    "stride, output_padding",
    [
        ((2, 2), (2, 2)),
        ((1, 1), (1, 1)),
    ],
)
def test_slow_conv_transpose2d_rejects_invalid_output_padding(stride, output_padding):
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (2, 3, 3, 3), ["-1", "1"])
    args = (
        inp,
        weight,
        (3, 3),
        None,
        stride,
        (0, 0),
        output_padding,
        (1, 1),
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose2d(*args)
    with pytest.raises(_GEMS_ERRORS):
        flag_gems.slow_conv_transpose2d(*args)


@pytest.mark.slow_conv_transpose2d
@pytest.mark.parametrize("stride", [(0, 0), (-1, 1)])
def test_slow_conv_transpose2d_rejects_nonpositive_stride(stride):
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (2, 3, 3, 3), ["-1", "1"])
    args = (inp, weight, (3, 3), None, stride, (0, 0), (0, 0), (1, 1))
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose2d(*args)
    with pytest.raises(_GEMS_ERRORS):
        flag_gems.slow_conv_transpose2d(*args)


@pytest.mark.slow_conv_transpose2d
@pytest.mark.parametrize("dilation", [(0, 0), (-1, -1), (1, -2)])
def test_slow_conv_transpose2d_rejects_nonpositive_dilation(dilation):
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (2, 3, 3, 3), ["-1", "1"])
    args = (inp, weight, (3, 3), None, (1, 1), (0, 0), (0, 0), dilation)
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose2d(*args)
    with pytest.raises(_GEMS_ERRORS):
        flag_gems.slow_conv_transpose2d(*args)


@pytest.mark.slow_conv_transpose2d
@pytest.mark.parametrize(
    "scalar_param, scalar_value",
    [
        ("kernel_size", 3),
        ("stride", 1),
        ("padding", 0),
        ("output_padding", 0),
        ("dilation", 1),
    ],
)
def test_slow_conv_transpose2d_rejects_scalar_params(scalar_param, scalar_value):
    inp = tu.make_input(torch.float32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (2, 3, 3, 3), ["-1", "1"])
    args = [inp, weight, (3, 3), None, (1, 1), (0, 0), (0, 0), (1, 1)]
    index = {
        "kernel_size": 2,
        "stride": 4,
        "padding": 5,
        "output_padding": 6,
        "dilation": 7,
    }[scalar_param]
    args[index] = scalar_value
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose2d(*args)
    with pytest.raises(_GEMS_ERRORS):
        flag_gems.slow_conv_transpose2d(*args)


@pytest.mark.slow_conv_transpose2d
def test_slow_conv_transpose2d_rejects_non_float_dtype():
    inp = tu.make_input(torch.int32, (1, 2, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.int32, (2, 3, 3, 3), ["-1", "1"])
    args = (inp, weight, (3, 3), None, (1, 1), (0, 0), (0, 0), (1, 1))
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_transpose2d(*args)
    with pytest.raises(_GEMS_ERRORS):
        flag_gems.slow_conv_transpose2d(*args)


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
