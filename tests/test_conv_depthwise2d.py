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

SHAPE_DEPTHWISE = [
    ((32, 4, 8, 8), (32, 1, 2, 2), (2, 2)),
    ((18, 16, 4, 4), (16, 1, 2, 2), (2, 2)),
    # ((9, 32, 4, 4), (128, 1, 2, 2), (2, 2)),
    # ((32, 16, 8, 8), (32, 1, 4, 4), (4, 4)),
    # ((18, 8, 4, 4), (16, 1, 2, 2), (2, 2)),
    # ((9, 4, 4, 4), (128, 1, 2, 2), (2, 2)),
    # ((32, 4, 8, 8), (32, 1, 3, 3), (3, 3)),
    # ((18, 16, 13, 13), (16, 1, 5, 5), (5, 5)),
    # ((9, 32, 8, 8), (128, 1, 3, 3), (3, 3)),
    # ((32, 16, 9, 9), (32, 1, 5, 5), (5, 5)),
    # ((18, 8, 7, 7), (16, 1, 3, 3), (3, 3)),
    # ((9, 4, 6, 6), (128, 1, 3, 3), (3, 3)),
]


def _reference_depthwise2d(inp, weight, kernel_size, bias, stride, padding, dilation):
    if inp.device.type == "cpu":
        if tuple(weight.shape[-2:]) != tuple(kernel_size):
            raise RuntimeError("kernel_size must match the weight's spatial dimensions")
        return torch.nn.functional.conv2d(
            inp, weight, bias, stride, padding, dilation, groups=inp.shape[1]
        )
    return torch.ops.aten._conv_depthwise2d(
        inp, weight, kernel_size, bias, stride, padding, dilation
    )


@pytest.mark.conv_depthwise2d
@pytest.mark.underscore_conv_depthwise2d
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
@pytest.mark.parametrize("shape_input, shape_weight, kernel", SHAPE_DEPTHWISE)
@pytest.mark.parametrize("stride", [[1, 1], [2, 2]])
@pytest.mark.parametrize("padding", [[0, 0], [1, 1]])
@pytest.mark.parametrize("dilation", [[1, 1]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("bias", [True, False])
def test_conv_depthwise2d(
    shape_input, shape_weight, kernel, stride, padding, dilation, dtype, bias
):
    torch.manual_seed(0)
    inp = torch.randn(shape_input, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(shape_weight, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)

    if bias:
        bias_tensor = torch.randn(shape_weight[0], dtype=dtype, device=flag_gems.device)
        ref_bias = utils.to_reference(bias_tensor, True)
    else:
        bias_tensor = None
        ref_bias = None

    ref_out = _reference_depthwise2d(
        ref_inp,
        ref_weight,
        kernel,
        ref_bias,
        stride,
        padding,
        dilation,
    )

    res_out = flag_gems._conv_depthwise2d(
        inp, weight, kernel, bias_tensor, stride, padding, dilation
    )
    assert res_out.device == inp.device
    assert res_out.dtype == dtype
    assert res_out.shape == ref_out.shape
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv_depthwise2d
@pytest.mark.underscore_conv_depthwise2d
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_conv_depthwise2d_noncontiguous(dtype):
    torch.manual_seed(0)
    base = torch.randn((2, 8, 9, 11), dtype=dtype, device=flag_gems.device)
    inp = base[:, ::2, :, :]
    weight = torch.randn((4, 1, 3, 3), dtype=dtype, device=flag_gems.device)
    ref = _reference_depthwise2d(
        utils.to_reference(inp, True), utils.to_reference(weight, True),
        [3, 3], None, [2, 2], [1, 1], [1, 1]
    )
    result = flag_gems._conv_depthwise2d(
        inp, weight, [3, 3], None, [2, 2], [1, 1], [1, 1]
    )
    assert result.device == inp.device
    utils.gems_assert_close(result, ref, dtype, reduce_dim=9)
    assert result.shape == ref.shape
    assert result.dtype == dtype


@pytest.mark.conv_depthwise2d
@pytest.mark.underscore_conv_depthwise2d
def test_conv_depthwise2d_invalid_kernel_size():
    torch.manual_seed(0)
    inp = torch.randn((1, 4, 8, 8), device=flag_gems.device)
    weight = torch.randn((4, 1, 3, 3), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._conv_depthwise2d(
            inp, weight, [2, 2], None, [1, 1], [0, 0], [1, 1]
        )
