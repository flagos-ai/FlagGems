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
from .conftest import QUICK_MODE

vendor_name = flag_gems.vendor_name

if QUICK_MODE:
    SHAPE_CONV3D = [
        ((1, 2, 5, 5, 5), (1, 2, 3, 3, 3), 1),
    ]
    FLOAT_DTYPES = [torch.float32]
    STRIDES = [1]
    INT_PADDINGS = [1]
    STR_PADDINGS = ["same"]
    DILATIONS = [1]
    BIASES = [True]
else:
    SHAPE_CONV3D = [
        ((1, 2, 5, 5, 5), (1, 2, 3, 3, 3), 1),
        ((2, 3, 9, 9, 9), (1, 3, 3, 3, 3), 1),
        # ((2, 2, 3, 3, 3), (1, 2, 2, 2, 2), 1),
        # ((32, 8, 8, 8, 8), (32, 8, 2, 2, 2), 1),
        # ((18, 16, 4, 4, 4), (16, 16, 2, 2, 2), 1),
        ((9, 16, 4, 4, 4), (128, 4, 2, 2, 2), 4),
        # ((32, 16, 8, 8, 8), (32, 4, 4, 4, 4), 4),
        # ((18, 16, 4, 4, 4), (16, 8, 2, 2, 2), 2),
        # ((9, 16, 4, 4, 4), (128, 8, 2, 2, 2), 2),
        ((32, 8, 8, 8, 8), (32, 8, 3, 3, 3), 1),
        ((18, 16, 5, 5, 5), (16, 16, 3, 3, 3), 1),
        # ((9, 16, 7, 7, 7), (128, 4, 3, 3, 3), 4),
        # ((32, 16, 9, 9, 9), (32, 4, 5, 5, 5), 4),
        # ((18, 16, 11, 11, 11), (16, 8, 3, 3, 3), 2),
        # ((9, 16, 6, 6, 6), (128, 8, 3, 3, 3), 2),
    ]
    FLOAT_DTYPES = [torch.float16, torch.float32]
    STRIDES = [1, 2]
    INT_PADDINGS = [0, 1]
    STR_PADDINGS = ["valid", "same"]
    DILATIONS = [1, 2]
    BIASES = [True, False]


@pytest.mark.conv3d
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV3D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", INT_PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dilation", DILATIONS)
@pytest.mark.parametrize("bias", BIASES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_conv3d(
    monkeypatch, shape, kernel, stride, padding, groups, dtype, dilation, bias
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = utils.to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    if bias is True:
        bias = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=False
        )
        bias_ref = utils.to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = utils.to_reference(weight, True)
    ref_out = torch.nn.functional.conv3d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv3d(
        inp,
        weight,
        bias=bias,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv3d
def test_conv3d_large_n_offsets():
    """Batch offsets past INT32_MAX must not wrap (Issue #6588).

    `conv3d()` pads `weight_c` up to 16 before launching the kernel, so the
    input row pitch for this patch-embedding shape is 16 * 2 * 14 * 14 = 6272
    elements. The last of 522240 rows then starts at 3.27e9 elements, beyond
    INT32_MAX; the 32-bit address arithmetic used to fault during autotuning.
    """
    if flag_gems.device != "cuda":
        pytest.skip("requires a CUDA-compatible device")

    n, out_c = 522240, 1280
    needed = (n * 16 * 2 * 14 * 14 + n * 3 * 2 * 14 * 14 + n * out_c) * 2
    if torch.cuda.mem_get_info()[0] < needed + 2**30:
        pytest.skip(f"requires ~{needed / 2**30:.0f} GiB of free device memory")

    inp = torch.ones((n, 3, 2, 14, 14), dtype=torch.bfloat16, device=flag_gems.device)
    weight = torch.ones(
        (out_c, 3, 2, 14, 14), dtype=torch.bfloat16, device=flag_gems.device
    )

    out = flag_gems.conv3d(
        inp, weight, stride=(2, 14, 14), padding=0, dilation=1, groups=1
    )

    assert out.shape == (n, out_c, 1, 1, 1)
    # Each output element sums one 3 x 2 x 14 x 14 patch of ones.
    patch_sum = 3 * 2 * 14 * 14
    assert out.min().item() == patch_sum
    assert out.max().item() == patch_sum


@pytest.mark.conv3d_padding
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #3023")
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV3D)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", STR_PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dilation", DILATIONS)
@pytest.mark.parametrize("bias", BIASES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_conv3d_padding(
    monkeypatch, shape, kernel, stride, padding, groups, dtype, dilation, bias
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = utils.to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    if bias is True:
        bias = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=False
        )
        bias_ref = utils.to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = utils.to_reference(weight, True)
    ref_out = torch.nn.functional.conv3d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv3d(
        inp,
        weight,
        bias=bias,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
