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

@pytest.fixture(autouse=True)
def _native_convolution_precision():
    if flag_gems.vendor_name != "mthreads":
        yield
        return
    previous = torch.backends.mudnn.allow_tf32
    try:
        torch.backends.mudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.mudnn.allow_tf32 = previous


FLOAT_DTYPES = [torch.float16, torch.float32]
SHAPES = [
    ((1, 2, 5, 5), (1, 2, 3, 3), 1),
    ((2, 3, 9, 9), (4, 3, 3, 3), 1),
    ((2, 4, 7, 7), (6, 2, 3, 3), 2),
]
OUTPUT_MASKS = [
    [True, True, True],
    [True, False, False],
    [False, True, False],
    [False, False, True],
]


@pytest.mark.convolution_backward_overrideable
@pytest.mark.parametrize("shape,weight_shape,groups", SHAPES)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("output_mask", OUTPUT_MASKS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_convolution_backward_overrideable(
    shape, weight_shape, groups, stride, padding, output_mask, dtype
):
    if (
        flag_gems.vendor_name == "ascend"
        and dtype == torch.float32
        and any(output_mask[:2])
        and not utils.TO_CPU
    ):
        pytest.skip("Ascend native float32 input/weight gradients do not meet the reference precision; CPU reference remains covered")
    torch.backends.cudnn.allow_tf32 = False
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(weight_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    stride_t = [stride, stride]
    padding_t = [padding, padding]
    dilation_t = [1, 1]
    output_padding = [0, 0]
    ref_forward = torch.nn.functional.conv2d(
        ref_inp, ref_weight, stride=stride, padding=padding, groups=groups
    )
    grad_output = torch.randn(ref_forward.shape, dtype=dtype, device=flag_gems.device)
    ref_grad_output = utils.to_reference(grad_output, True)
    ref_outputs = torch.ops.aten.convolution_backward(
        ref_grad_output,
        ref_inp,
        ref_weight,
        [weight.shape[0]],
        stride_t,
        padding_t,
        dilation_t,
        False,
        output_padding,
        groups,
        output_mask,
    )
    outputs = flag_gems.convolution_backward_overrideable(
        grad_output,
        inp,
        weight,
        stride_t,
        padding_t,
        dilation_t,
        False,
        output_padding,
        groups,
        output_mask,
    )
    for enabled, result, reference in zip(output_mask, outputs, ref_outputs):
        if enabled:
            assert result.device == inp.device
            assert result.dtype == inp.dtype
            assert result.shape == reference.shape
            assert result is not inp
            assert result is not weight
            utils.gems_assert_close(result, reference, dtype, reduce_dim=weight.shape[2])
        else:
            assert result is None
