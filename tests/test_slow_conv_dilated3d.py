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

# Test batched and unbatched slow conv3d with dilation.
# Extreme ranges retain their bounds and use the original reference dtype to preserve overflow.
# Cases: (input shape, weight shape, kernel size, stride, padding, dilation).
SLOW_CONV_DILATED3D_CASES = tu.selected_cases(
    [
        ((1, 2, 5, 5, 5), (1, 2, 3, 3, 3), (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        ((2, 3, 6, 6, 6), (4, 3, 3, 3, 3), (3, 3, 3), (1, 1, 1), (0, 0, 0), (1, 1, 1)),
        ((1, 3, 8, 8, 8), (4, 3, 3, 3, 3), (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1)),
        ((2, 4, 6, 6, 6), (6, 4, 3, 3, 3), (3, 3, 3), (1, 1, 1), (1, 1, 1), (2, 2, 2)),
        ((1, 2, 7, 7, 7), (3, 2, 3, 3, 3), (3, 3, 3), (1, 1, 1), (2, 2, 2), (1, 1, 1)),
        ((2, 3, 5, 5, 5), (5, 3, 1, 1, 1), (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)),
        ((2, 4, 5, 5, 5), (3, 4, 3, 3, 3), (3, 3, 3), (2, 1, 1), (1, 1, 0), (1, 1, 1)),
        ((1, 2, 4, 4, 4), (3, 2, 2, 2, 2), (2, 2, 2), (1, 1, 1), (0, 0, 0), (2, 2, 2)),
        ((2, 8, 4, 4, 4), (4, 8, 2, 2, 2), (2, 2, 2), (1, 1, 1), (0, 0, 0), (1, 1, 1)),
        ((1, 3, 9, 9, 9), (2, 3, 3, 3, 3), (3, 3, 3), (2, 2, 2), (2, 2, 2), (1, 1, 1)),
        ((2, 2, 6, 5, 7), (3, 2, 3, 3, 3), (3, 3, 3), (1, 2, 1), (1, 1, 2), (1, 1, 1)),
        ((1, 2, 6, 6, 6), (4, 2, 3, 3, 3), (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ],
    quick=[
        ((1, 2, 5, 5, 5), (1, 2, 3, 3, 3), (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ],
)

BIASES = tu.selected_cases([True, False], quick=[True])

# Drop the leading batch dimension. Bias is excluded: the reference is unreliable for 4-D + bias.
_UNBATCHED_CASES = []
for i in tu.selected_cases([0, 5, 4], quick=[0]):
    shape, *params = SLOW_CONV_DILATED3D_CASES[i]
    _UNBATCHED_CASES.append((shape[1:], *params))

# 3x3x3, pointwise 1x1x1 and dilation-2 paths.
_VALUE_RANGE_CASES = [
    SLOW_CONV_DILATED3D_CASES[i] for i in tu.selected_cases([0, 5, 3], quick=[0])
]

# Padding 1, no padding and dilation 2.
_BACKWARD_CASES = [
    SLOW_CONV_DILATED3D_CASES[i] for i in tu.selected_cases([0, 1, 3], quick=[0])
]

_BACKWARD_DTYPES = tu.selected_cases(
    [torch.float16, torch.float32, torch.bfloat16, torch.float64], quick=[torch.float32]
)


def _conv_output_shape(inp_shape, weight_shape, stride, padding, dilation):
    def _out_size(in_size, k, s, p, d):
        return (in_size + 2 * p - d * (k - 1) - 1) // s + 1

    # A 4-D input is the unbatched route: the spatial dims start one offset
    # earlier and the output drops the batch dim too.
    if len(inp_shape) == 4:
        spatial_offset, lead = 1, (weight_shape[0],)
    else:
        spatial_offset, lead = 2, (inp_shape[0], weight_shape[0])

    spatial = tuple(
        _out_size(
            inp_shape[spatial_offset + i],
            weight_shape[2 + i],
            stride[i],
            padding[i],
            dilation[i],
        )
        for i in range(3)
    )
    return lead + spatial


def _make_conv_inputs(
    inp_shape, weight_shape, with_bias, dtype, value_range=("-1", "1")
):
    inp = tu.make_input(dtype, inp_shape, value_range)
    weight = tu.make_input(dtype, weight_shape, value_range)
    if with_bias:
        # bias has one element per output channel (the first weight dim).
        bias = tu.make_input(dtype, (weight_shape[0],), value_range)
    else:
        bias = None
    return inp, weight, bias


def _assert_close(res_out, ref_out, dtype, equal_nan=False):
    # Supplement dtype-relative tolerance for the fp64 reference with absolute reduction-error bounds.
    if dtype == torch.bfloat16:
        atol = 2e-1
    elif dtype == torch.float16:
        atol = 2e-2
    else:
        atol = 1e-4
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=equal_nan, atol=atol)


@pytest.mark.slow_conv_dilated3d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, dilation",
    SLOW_CONV_DILATED3D_CASES,
)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_dilated3d(
    inp_shape, weight_shape, kernel_size, stride, padding, dilation, dtype, bias
):
    inp, weight, bias_t = _make_conv_inputs(inp_shape, weight_shape, bias, dtype)
    ref_inp = tu.to_reference(inp, True)
    ref_weight = tu.to_reference(weight, True)
    ref_bias = tu.to_reference(bias_t, True)

    ref_out = torch.ops.aten.slow_conv_dilated3d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    ).to(dtype)

    res_out = flag_gems.slow_conv_dilated3d(
        inp, weight, kernel_size, bias_t, stride, padding, dilation
    )

    _assert_close(res_out, ref_out, dtype)


@pytest.mark.slow_conv_dilated3d
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, dilation",
    _UNBATCHED_CASES,
)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_slow_conv_dilated3d_unbatched(
    inp_shape, weight_shape, kernel_size, stride, padding, dilation, dtype
):
    inp, weight, bias_t = _make_conv_inputs(inp_shape, weight_shape, False, dtype)
    ref_inp = tu.to_reference(inp, True)
    ref_weight = tu.to_reference(weight, True)

    ref_out = torch.ops.aten.slow_conv_dilated3d(
        ref_inp, ref_weight, kernel_size, None, stride, padding, dilation
    ).to(dtype)

    res_out = flag_gems.slow_conv_dilated3d(
        inp, weight, kernel_size, bias_t, stride, padding, dilation
    )

    # The output must also drop the batch dim, not silently keep a leading 1.
    assert tuple(res_out.shape) == _conv_output_shape(
        inp_shape, weight_shape, stride, padding, dilation
    )
    _assert_close(res_out, ref_out, dtype)


@pytest.mark.slow_conv_dilated3d
@pytest.mark.parametrize("case", _VALUE_RANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_dilated3d_value_ranges(case, value_range, dtype, bias):
    inp_shape, weight_shape, kernel_size, stride, padding, dilation = case
    inp = tu.make_input(dtype, inp_shape, value_range)
    weight = tu.make_input(dtype, weight_shape, value_range)
    bias_t = tu.make_input(dtype, (weight_shape[0],), value_range) if bias else None
    ref_inp = tu.to_reference(inp, not tu.is_extreme_range(value_range))
    ref_weight = tu.to_reference(weight, not tu.is_extreme_range(value_range))
    ref_bias = tu.to_reference(bias_t, not tu.is_extreme_range(value_range))

    ref_out = torch.ops.aten.slow_conv_dilated3d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    ).to(dtype)

    res_out = flag_gems.slow_conv_dilated3d(
        inp, weight, kernel_size, bias_t, stride, padding, dilation
    )

    _assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.slow_conv_dilated3d_out
@pytest.mark.parametrize(
    "inp_shape, weight_shape, kernel_size, stride, padding, dilation",
    SLOW_CONV_DILATED3D_CASES,
)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("bias", BIASES)
def test_slow_conv_dilated3d_out(
    inp_shape, weight_shape, kernel_size, stride, padding, dilation, dtype, bias
):
    inp, weight, bias_t = _make_conv_inputs(inp_shape, weight_shape, bias, dtype)
    ref_inp = tu.to_reference(inp, True)
    ref_weight = tu.to_reference(weight, True)
    ref_bias = tu.to_reference(bias_t, True)

    # The .out overload must write into the provided tensor and return it.
    ref_full = torch.ops.aten.slow_conv_dilated3d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    )
    ref_out = torch.empty_like(ref_full)
    ref_ret = torch.ops.aten.slow_conv_dilated3d.out(
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
    res_ret = flag_gems.slow_conv_dilated3d(
        inp, weight, kernel_size, bias_t, stride, padding, dilation, out=out
    )
    assert res_ret is out

    _assert_close(res_ret, ref_ret, dtype)


@pytest.mark.slow_conv_dilated3d_backward
@pytest.mark.parametrize("case", _BACKWARD_CASES)
@pytest.mark.parametrize("dtype", tu.selected_cases(_BACKWARD_DTYPES))
def test_slow_conv_dilated3d_backward(case, dtype):
    inp_shape, weight_shape, kernel_size, stride, padding, dilation = case
    out_shape = _conv_output_shape(inp_shape, weight_shape, stride, padding, dilation)

    inp = tu.make_input(dtype, inp_shape, ["-1", "1"]).requires_grad_()
    weight = tu.make_input(dtype, weight_shape, ["-1", "1"]).requires_grad_()
    bias = tu.make_input(dtype, (weight_shape[0],), ["-1", "1"]).requires_grad_()
    grad_out = tu.make_input(dtype, out_shape, ["-1", "1"])

    # Reference graph on the fp64-upcast inputs.
    ref_inp = tu.to_reference(inp, True).requires_grad_()
    ref_weight = tu.to_reference(weight, True).requires_grad_()
    ref_bias = tu.to_reference(bias, True).requires_grad_()
    ref_grad_out = tu.to_reference(grad_out, True)

    ref_out = torch.ops.aten.slow_conv_dilated3d(
        ref_inp, ref_weight, kernel_size, ref_bias, stride, padding, dilation
    )
    ref_gi, ref_gw, ref_gb = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), grad_outputs=ref_grad_out
    )

    # The candidate forward must match the fp64 reference...
    res_out = flag_gems.slow_conv_dilated3d(
        inp, weight, kernel_size, bias, stride, padding, dilation
    )
    _assert_close(res_out, ref_out.to(dtype), dtype)

    # ...and, if the candidate kernel is autograd-aware, its gradients must
    # match the reference gradients too.
    assert res_out.requires_grad
    res_gi, res_gw, res_gb = torch.autograd.grad(
        res_out, (inp, weight, bias), grad_outputs=grad_out
    )
    _assert_close(res_gi, ref_gi.to(dtype), dtype)
    _assert_close(res_gw, ref_gw.to(dtype), dtype)
    _assert_close(res_gb, ref_gb.to(dtype), dtype)


@pytest.mark.slow_conv_dilated3d_nan_inf
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(utils.ALL_FLOAT_DTYPES))
)
@pytest.mark.parametrize("special_arg", ["inp", "weight", "bias"])
def test_slow_conv_dilated3d_nan_inf(dtype, scenario, special_arg):
    inp = torch.ones((1, 1, 4, 4, 4), dtype=dtype, device=flag_gems.device)
    weight = torch.ones((5, 1, 2, 2, 2), dtype=dtype, device=flag_gems.device)
    bias = torch.ones((5,), dtype=dtype, device=flag_gems.device)

    specials = tu.make_special_input(dtype, scenario)
    target = {"inp": inp, "weight": weight, "bias": bias}[special_arg]
    target.flatten()[: specials.numel()] = specials
    kernel_size = (2, 2, 2)

    ref_inp = tu.to_reference(inp, True)
    ref_weight = tu.to_reference(weight, True)
    ref_bias = tu.to_reference(bias, True)
    ref_out = torch.ops.aten.slow_conv_dilated3d(
        ref_inp, ref_weight, kernel_size, ref_bias, (1, 1, 1), (0, 0, 0), (1, 1, 1)
    ).to(dtype)

    res_out = flag_gems.slow_conv_dilated3d(
        inp, weight, kernel_size, bias, (1, 1, 1), (0, 0, 0), (1, 1, 1)
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.slow_conv_dilated3d_negative
def test_slow_conv_dilated3d_rejects_wrong_kernel_size():
    inp, weight, _ = _make_conv_inputs(
        (1, 2, 5, 5, 5), (1, 2, 3, 3, 3), False, torch.float32
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated3d(
            inp, weight, (2, 2, 2), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.slow_conv_dilated3d(
            inp, weight, (2, 2, 2), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )


@pytest.mark.slow_conv_dilated3d_negative
def test_slow_conv_dilated3d_rejects_channel_mismatch():
    inp = tu.make_input(torch.float32, (1, 2, 5, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 3, 3, 3, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )


@pytest.mark.slow_conv_dilated3d_negative
def test_slow_conv_dilated3d_rejects_int_dtype():
    for bad_dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn, torch.int32):
        inp = tu.make_input(bad_dtype, (1, 2, 5, 5, 5), ["-1", "1"])
        weight = tu.make_input(bad_dtype, (1, 2, 3, 3, 3), ["-1", "1"])
        with pytest.raises(RuntimeError):
            torch.ops.aten.slow_conv_dilated3d(
                inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
            )
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            flag_gems.slow_conv_dilated3d(
                inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
            )


@pytest.mark.slow_conv_dilated3d_negative
def test_slow_conv_dilated3d_rejects_6d_input():
    inp = tu.make_input(torch.float32, (1, 2, 2, 5, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3, 3, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )


@pytest.mark.slow_conv_dilated3d_negative
def test_slow_conv_dilated3d_rejects_wrong_weight_rank():
    inp = tu.make_input(torch.float32, (1, 2, 5, 5, 5), ["-1", "1"])
    weight = tu.make_input(torch.float32, (1, 2, 3, 3), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )


@pytest.mark.slow_conv_dilated3d_negative
def test_slow_conv_dilated3d_rejects_negative_stride():
    inp, weight, _ = _make_conv_inputs(
        (1, 2, 5, 5, 5), (1, 2, 3, 3, 3), False, torch.float32
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (-1, 1, 1), (1, 1, 1), (1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (-1, 1, 1), (1, 1, 1), (1, 1, 1)
        )


@pytest.mark.slow_conv_dilated3d_negative
def test_slow_conv_dilated3d_rejects_negative_dilation():
    inp, weight, _ = _make_conv_inputs(
        (1, 2, 5, 5, 5), (1, 2, 3, 3, 3), False, torch.float32
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (-1, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (-1, 1, 1)
        )


@pytest.mark.slow_conv_dilated3d_negative
def test_slow_conv_dilated3d_rejects_output_size_too_small():
    inp, weight, _ = _make_conv_inputs(
        (1, 2, 5, 5, 5), (1, 2, 3, 3, 3), False, torch.float32
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (5, 1, 1)
        )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.slow_conv_dilated3d(
            inp, weight, (3, 3, 3), None, (1, 1, 1), (1, 1, 1), (5, 1, 1)
        )


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
