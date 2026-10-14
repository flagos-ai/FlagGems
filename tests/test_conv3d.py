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

import importlib

import pytest
import torch
import torch.nn.functional as F

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


@pytest.mark.conv3d
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Ascend padding and conv3d implementations required",
)
@pytest.mark.parametrize("case", ["vision_bf16", "grouped", "same_even", "forced_grid"])
def test_ascend_conv3d_regressions(case):
    assert flag_gems.conv3d.__module__.endswith("_ascend.ops.conv3d")
    dtype = torch.bfloat16 if case == "vision_bf16" else torch.float32
    shape = (
        (1, 3, 2, 16, 16)
        if case == "vision_bf16"
        else (1, 6, 5, 5, 5) if case in ("grouped", "forced_grid") else (1, 2, 5, 6, 7)
    )
    wshape = (
        (1152, 3, 2, 16, 16)
        if case == "vision_bf16"
        else (
            (66, 3, 2, 2, 2)
            if case == "forced_grid"
            else (8, 3, 2, 2, 2) if case == "grouped" else (3, 2, 2, 3, 4)
        )
    )
    inp = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    weight = torch.randn(wshape, device=inp.device, dtype=dtype)
    bias = torch.randn(wshape[0] * 2, device=inp.device, dtype=dtype)[::2]
    stride = (2, 16, 16) if case == "vision_bf16" else 1
    groups = 2 if case in ("grouped", "forced_grid") else 1
    padding = "same" if case == "same_even" else 0
    previous_hf32 = torch.npu.conv.allow_hf32
    try:
        if dtype == torch.float32:
            torch.npu.conv.allow_hf32 = False
        reference = F.conv3d(inp, weight, bias, stride, padding, groups=groups)
    finally:
        torch.npu.conv.allow_hf32 = previous_hf32
    module = importlib.import_module(flag_gems.conv3d.__module__)
    cap = module._CONV3D_MAX_GRID
    try:
        if case == "forced_grid":
            module._CONV3D_MAX_GRID = 2
        previous_registrar = flag_gems.current_work_registrar
        lib = torch.library.Library("aten", "IMPL")
        try:
            flag_gems.only_enable(lib=lib, include=["conv3d"])
            actual = F.conv3d(inp, weight, bias, stride, padding, groups=groups)
        finally:
            lib._destroy()
            flag_gems.current_work_registrar = previous_registrar
    finally:
        module._CONV3D_MAX_GRID = cap
    torch.testing.assert_close(
        actual,
        reference,
        rtol=1e-2 if dtype == torch.bfloat16 else 2e-3,
        atol=0.125 if dtype == torch.bfloat16 else 2e-4,
    )
