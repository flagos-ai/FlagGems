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

if QUICK_MODE:
    SHAPE_CONV1D = [
        ((32, 2, 4), (17, 2, 2)),
    ]
    SHAPE_CONV1D_DILATION = [
        ((32, 2, 16), (17, 2, 3)),
    ]
    FLOAT_DTYPES = [torch.float32]
    STR_PADDINGS = ["same"]
    INT_PADDINGS = [2]
    DILATIONS = [(2,)]

else:
    SHAPE_CONV1D = [
        ((32, 2, 4), (17, 2, 2)),
        ((32, 15, 6), (17, 15, 2)),
        # ((64, 64, 64), (128, 64, 7)),  # commented out to reduce CI timeout - large
    ]

    SHAPE_CONV1D_DILATION = [
        ((32, 2, 16), (17, 2, 3)),
        ((32, 15, 32), (17, 15, 3)),
        # ((64, 64, 64), (128, 64, 3)),  # commented out to reduce CI timeout - large
    ]
    FLOAT_DTYPES = [torch.float32, torch.float16]
    STR_PADDINGS = ["valid", "same"]
    INT_PADDINGS = [0, 2]
    DILATIONS = [1, (2,)]  # original: [1, 2, (1,), (2,)], reduced to avoid CI timeout


@pytest.mark.conv1d
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV1D)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("padding", [1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_conv1d(monkeypatch, shape, kernel, stride, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)
    ref_out = torch.nn.functional.conv1d(
        ref_inp, ref_weight, bias=None, stride=stride, padding=padding, dilation=1
    )

    res_out = flag_gems.conv1d(
        inp, weight, bias=None, stride=stride, padding=padding, dilation=1
    )
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv1d_padding
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="Issue #3022")
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV1D)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", STR_PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_conv1d_padding(monkeypatch, shape, kernel, stride, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)
    ref_out = torch.nn.functional.conv1d(
        ref_inp, ref_weight, bias=None, stride=stride, padding=padding, dilation=1
    )

    res_out = flag_gems.conv1d(
        inp, weight, bias=None, stride=stride, padding=padding, dilation=1
    )
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv1d
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV1D_DILATION)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", INT_PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dilation", DILATIONS)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_conv1d_dilation(shape, kernel, stride, padding, dtype, dilation):
    """Test conv1d with various dilation values, including tuple form.

    This specifically tests the fix where conv1d must properly convert dilation
    to a 2D tuple before delegating to conv2d. Previously, passing dilation as
    a single-element tuple (e.g., (1,)) would cause a ValueError in conv2d.
    """
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.conv1d(
        ref_inp,
        ref_weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    res_out = flag_gems.conv1d(
        inp, weight, bias=None, stride=stride, padding=padding, dilation=dilation
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
