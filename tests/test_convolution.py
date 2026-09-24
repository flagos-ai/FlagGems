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


@pytest.fixture(autouse=True)
def ieee_precision(monkeypatch):
    # Keep native FP32 arithmetic comparable across reference devices.
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


# Convolution needs batched input and weight tensors of equal rank (3–5).
_BASIC_CASES = [
    ((20, 320, 15), (32, 320, 3), (1,), (1,), (1,)),
    ((16, 128, 64, 60), (32, 128, 3, 3), (1, 1), (1, 1), (1, 1)),
    ((16, 7, 57, 32, 29), (8, 7, 3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
]
_QUICK_CASES = [((2, 19, 7), (4, 19, 3), (1,), (1,), (1,))]
CONV_CASES = tu.selected_cases(_BASIC_CASES, quick=_QUICK_CASES)

CONV_DTYPES = [torch.float32, torch.bfloat16, torch.float16]
if utils.fp64_is_supported:
    CONV_DTYPES.append(torch.float64)

_PARAM_INPUT_SHAPE = (20, 320, 15)
_PARAM_DTYPE = torch.bfloat16
_PARAM_CASES = tu.selected_cases(
    [
        ((32, 320, 3), 1, 1, 1, 1),
        ((32, 320, 3), 2, 1, 1, 1),
        ((32, 320, 3), 3, 1, 1, 1),
        ((32, 320, 3), 1, 0, 1, 1),
        ((32, 320, 3), 1, 2, 1, 1),
        ((32, 320, 3), 1, 1, 2, 1),
        ((32, 320, 3), 1, 1, 3, 1),
        ((32, 160, 3), 1, 1, 1, 2),
        ((32, 80, 3), 1, 1, 1, 4),
        ((320, 1, 3), 1, 1, 1, 320),
        ((32, 320, 1), 1, 0, 1, 1),
    ],
    quick=[],
)

_TRANSPOSED_CASES = tu.selected_cases(
    [
        ((20, 320, 15), (320, 32, 3), (1,), (1,), (0,), 1),
        ((20, 320, 15), (320, 32, 3), (2,), (1,), (1,), 1),
        ((20, 320, 15), (320, 32, 3), (3,), (1,), (2,), 1),
        ((20, 320, 15), (320, 16, 3), (2,), (1,), (1,), 2),
        ((16, 128, 64, 60), (128, 8, 3, 3), (1, 1), (1, 1), (0, 0), 1),
        ((16, 7, 57, 32, 29), (7, 8, 3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 0, 0), 1),
    ],
    quick=[],
)

_BACKWARD_CASES = tu.selected_cases(
    [
        ((2, 6, 12), (3, 6, 3), (1,), (1,), (1,), 1),
        ((2, 4, 8, 8), (5, 4, 3, 3), (2, 2), (1, 1), (1, 1), 1),
        ((2, 4, 5, 6, 7), (6, 2, 3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), 2),
    ],
    quick=[],
)

_LAYOUT_CASES = tu.selected_cases(
    [
        ("non_contiguous", (4, 4, 3, 3)),
        ("channels_last", (4, 8, 3, 3)),
    ],
    quick=[],
)


@pytest.mark.convolution
@pytest.mark.parametrize("input_shape,weight_shape,stride,padding,dilation", CONV_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", CONV_DTYPES)
@pytest.mark.parametrize("with_bias", tu.selected_cases([False, True], quick=[False]))
def test_convolution_value_ranges(
    input_shape, weight_shape, stride, padding, dilation, value_range, dtype, with_bias
):
    inp = tu.make_input(dtype, input_shape, value_range)
    weight = tu.make_input(dtype, weight_shape, value_range)
    bias = tu.make_input(dtype, (weight_shape[0],), value_range) if with_bias else None
    ref_bias = tu.to_reference(bias)
    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)

    ref_out = torch.ops.aten.convolution(
        ref_inp,
        ref_weight,
        ref_bias,
        stride,
        padding,
        dilation,
        False,
        (0,) * len(stride),
        1,
    )
    res_out = flag_gems.convolution(
        inp, weight, bias, stride, padding, dilation, False, (0,) * len(stride), 1
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.convolution
@pytest.mark.parametrize("weight_shape,stride,padding,dilation,groups", _PARAM_CASES)
def test_convolution_parameters(weight_shape, stride, padding, dilation, groups):
    inp = tu.make_input(_PARAM_DTYPE, _PARAM_INPUT_SHAPE, ["-1", "1"])
    weight = tu.make_input(_PARAM_DTYPE, weight_shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)

    ref_out = torch.ops.aten.convolution(
        ref_inp,
        ref_weight,
        None,
        (stride,),
        (padding,),
        (dilation,),
        False,
        (0,),
        groups,
    )
    res_out = flag_gems.convolution(
        inp, weight, None, (stride,), (padding,), (dilation,), False, (0,), groups
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.convolution
@pytest.mark.parametrize(
    "input_shape,weight_shape,stride,padding,output_padding,groups", _TRANSPOSED_CASES
)
@pytest.mark.parametrize("dtype", CONV_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_convolution_transposed(
    input_shape,
    weight_shape,
    stride,
    padding,
    output_padding,
    groups,
    dtype,
    value_range,
):
    inp = tu.make_input(dtype, input_shape, value_range)
    weight = tu.make_input(dtype, weight_shape, value_range)
    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    dilation = (1,) * len(stride)

    ref_out = torch.ops.aten.convolution(
        ref_inp,
        ref_weight,
        None,
        stride,
        padding,
        dilation,
        True,
        output_padding,
        groups,
    )
    res_out = flag_gems.convolution(
        inp, weight, None, stride, padding, dilation, True, output_padding, groups
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.convolution
@pytest.mark.parametrize("layout,weight_shape", _LAYOUT_CASES)
@pytest.mark.parametrize("dtype", CONV_DTYPES)
def test_convolution_layouts(layout, weight_shape, dtype):
    base_inp = tu.make_input(dtype, (2, 8, 16, 16), ["-1", "1"])
    weight = tu.make_input(dtype, weight_shape, ["-1", "1"])
    ref_base = tu.to_reference(base_inp)
    ref_weight = tu.to_reference(weight)

    if layout == "non_contiguous":
        inp = base_inp[:, ::2]
        ref_inp = ref_base[:, ::2]
    else:
        inp = base_inp.to(memory_format=torch.channels_last)
        ref_inp = ref_base.to(memory_format=torch.channels_last)

    ref_out = torch.ops.aten.convolution(
        ref_inp, ref_weight, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
    )
    res_out = flag_gems.convolution(
        inp, weight, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
    )

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.convolution
@pytest.mark.parametrize("dtype", CONV_DTYPES)
@pytest.mark.parametrize(
    "input_shape", tu.selected_cases([(2, 6, 12)], quick=[(2, 19, 7)])
)
def test_convolution_out(dtype, input_shape):
    inp = tu.make_input(dtype, input_shape, ["-1", "1"])
    weight = tu.make_input(dtype, (4, input_shape[1], 3), ["-1", "1"])
    bias = tu.make_input(dtype, (4,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    ref_bias = tu.to_reference(bias)

    res_out = torch.empty(
        (input_shape[0], 4, input_shape[2]), dtype=dtype, device=flag_gems.device
    )
    ref_out = tu.to_reference(res_out)
    torch.ops.aten.convolution.out(
        ref_inp, ref_weight, ref_bias, (1,), (1,), (1,), False, (0,), 1, out=ref_out
    )
    res_ret = flag_gems.convolution(
        inp, weight, bias, (1,), (1,), (1,), False, (0,), 1, out=res_out
    )

    assert res_ret is res_out
    tu.assert_result_close(res_ret, ref_out)


@pytest.mark.convolution
@pytest.mark.parametrize(
    "input_shape,weight_shape,stride,padding,dilation,groups", _BACKWARD_CASES
)
@pytest.mark.parametrize("dtype", CONV_DTYPES)
@pytest.mark.parametrize("transposed", [False, True])
def test_convolution_backward(
    input_shape, weight_shape, stride, padding, dilation, groups, dtype, transposed
):
    channels_out = weight_shape[0]
    if transposed:
        weight_shape = (input_shape[1], channels_out // groups, *weight_shape[2:])
    inp = tu.make_input(dtype, input_shape, ["-1", "1"]).requires_grad_()
    weight = tu.make_input(dtype, weight_shape, ["-1", "1"]).requires_grad_()
    bias = tu.make_input(dtype, (channels_out,), ["-1", "1"]).requires_grad_()
    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)
    ref_bias = tu.to_reference(bias)

    ref_out = torch.ops.aten.convolution(
        ref_inp,
        ref_weight,
        ref_bias,
        stride,
        padding,
        dilation,
        transposed,
        (0,) * len(stride),
        groups,
    )
    upstream = tu.make_input(dtype, tuple(ref_out.shape), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)
    res_out = flag_gems.convolution(
        inp,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        (0,) * len(stride),
        groups,
    )

    tu.assert_result_close(res_out, ref_out)

    ref_grads = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), grad_outputs=ref_upstream
    )
    res_grads = torch.autograd.grad(res_out, (inp, weight, bias), grad_outputs=upstream)
    for res_grad, ref_grad in zip(res_grads, ref_grads):
        tu.assert_result_close(res_grad, ref_grad)


@pytest.mark.convolution
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(CONV_DTYPES), quick=[])
)
def test_convolution_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario).reshape(1, 1, 5)
    weight = torch.ones(1, 1, 1, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)
    ref_weight = tu.to_reference(weight)

    ref_out = torch.ops.aten.convolution(
        ref_inp, ref_weight, None, (1,), (0,), (1,), False, (0,), 1
    )
    res_out = flag_gems.convolution(inp, weight, None, (1,), (0,), (1,), False, (0,), 1)

    tu.assert_result_close(res_out, ref_out)


def _conv_inputs():
    inp = tu.make_input(torch.float32, (2, 4, 12), ["-1", "1"])
    weight = tu.make_input(torch.float32, (6, 4, 3), ["-1", "1"])
    return inp, weight


@pytest.mark.convolution
@pytest.mark.parametrize("padding", [-1, -2])
def test_convolution_negative_padding(padding):
    inp, weight = _conv_inputs()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.convolution(inp, weight, None, (1,), (padding,), (1,), False, (0,), 1)


@pytest.mark.convolution
@pytest.mark.parametrize("stride", [0, -1])
def test_convolution_non_positive_stride(stride):
    inp, weight = _conv_inputs()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.convolution(inp, weight, None, (stride,), (1,), (1,), False, (0,), 1)


@pytest.mark.convolution
@pytest.mark.parametrize("dilation", [0, -1])
def test_convolution_non_positive_dilation(dilation):
    inp, weight = _conv_inputs()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.convolution(
            inp, weight, None, (1,), (1,), (dilation,), False, (0,), 1
        )


@pytest.mark.convolution
@pytest.mark.parametrize("groups", [0, -1])
def test_convolution_non_positive_groups(groups):
    inp, weight = _conv_inputs()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.convolution(inp, weight, None, (1,), (1,), (1,), False, (0,), groups)


@pytest.mark.convolution
@pytest.mark.parametrize("input_shape", [(4, 12), (4,)])
def test_convolution_invalid_rank(input_shape):
    inp = tu.make_input(torch.float32, input_shape, ["-1", "1"])
    weight = tu.make_input(torch.float32, (6, 4, 3), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.convolution(inp, weight, None, (1,), (1,), (1,), False, (0,), 1)


@pytest.mark.convolution
def test_convolution_channel_mismatch():
    inp = tu.make_input(torch.float32, (2, 4, 12), ["-1", "1"])
    weight = tu.make_input(torch.float32, (6, 3, 3), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.convolution(inp, weight, None, (1,), (1,), (1,), False, (0,), 1)


@pytest.mark.convolution
def test_convolution_groups_not_divisible():
    inp = tu.make_input(torch.float32, (2, 4, 12), ["-1", "1"])
    weight = tu.make_input(torch.float32, (6, 3, 3), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.convolution(inp, weight, None, (1,), (1,), (1,), False, (0,), 3)


@pytest.mark.convolution
def test_convolution_kernel_larger_than_input():
    inp = tu.make_input(torch.float32, (2, 4, 4), ["-1", "1"])
    weight = tu.make_input(torch.float32, (6, 4, 8), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.convolution(inp, weight, None, (1,), (0,), (1,), False, (0,), 1)


@pytest.mark.convolution
def test_convolution_output_padding_not_smaller_than_stride():
    inp = tu.make_input(torch.float32, (2, 4, 12), ["-1", "1"])
    weight = tu.make_input(torch.float32, (4, 6, 3), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.convolution(inp, weight, None, (1,), (1,), (1,), True, (2,), 1)


@pytest.mark.convolution
def test_convolution_bias_length_mismatch():
    inp = tu.make_input(torch.float32, (2, 4, 12), ["-1", "1"])
    weight = tu.make_input(torch.float32, (6, 4, 3), ["-1", "1"])
    bias = tu.make_input(torch.float32, (5,), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.convolution(inp, weight, bias, (1,), (1,), (1,), False, (0,), 1)


@pytest.mark.convolution
def test_convolution_non_tensor_input():
    inp, weight = _conv_inputs()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.convolution(
            inp.tolist(), weight, None, (1,), (1,), (1,), False, (0,), 1
        )


@pytest.mark.convolution
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.uint8,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ],
)
def test_convolution_unsupported_dtype(dtype):
    inp = torch.zeros((2, 4, 12), dtype=dtype, device=flag_gems.device)
    weight = torch.zeros((6, 4, 3), dtype=dtype, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.convolution(inp, weight, None, (1,), (1,), (1,), False, (0,), 1)
