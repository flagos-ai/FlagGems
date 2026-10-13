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

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Ascend padding and conv3d implementations required",
)


@pytest.mark.parametrize(
    "case", ["empty", "rank3_strided", "rank5", "rank6", "negative_reflect"]
)
def test_ascend_pad_regressions(case):
    assert flag_gems.pad.__module__.endswith("_ascend.ops.pad")
    assert flag_gems.constant_pad_nd.__module__.endswith("_ascend.ops.pad")
    shape = {
        "empty": (2, 0),
        "rank3_strided": (2, 3, 4),
        "rank5": (1, 3, 2, 4, 4),
        "rank6": (1, 2, 3, 2, 4, 4),
        "negative_reflect": (2, 3, 7),
    }[case]
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    if case == "rank3_strided":
        inp = inp.transpose(1, 2)
    padding = (
        (0, 0, 0, 0, 0, 0, 0, 13)
        if case in ("rank5", "rank6")
        else (-1, 2) if case == "negative_reflect" else (1, 2)
    )
    mode = "reflect" if case == "negative_reflect" else "constant"
    value = None if mode == "reflect" else 3.5
    # TorchNPU does not support negative ReflectionPad1d; CPU is only the oracle.
    reference = F.pad(inp.cpu(), padding, mode, value).to(inp.device)
    with flag_gems.use_gems(include=["pad", "constant_pad_nd"]):
        actual = F.pad(inp, padding, mode, value)
    assert torch.equal(actual, reference)
    if mode == "constant":
        with flag_gems.use_gems(include=["constant_pad_nd"]):
            actual = torch.ops.aten.constant_pad_nd(inp, padding, value)
        assert torch.equal(actual, reference)


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
        with flag_gems.use_gems(include=["conv3d"]):
            actual = F.conv3d(inp, weight, bias, stride, padding, groups=groups)
    finally:
        module._CONV3D_MAX_GRID = cap
    torch.testing.assert_close(
        actual,
        reference,
        rtol=1e-2 if dtype == torch.bfloat16 else 2e-3,
        atol=0.125 if dtype == torch.bfloat16 else 2e-4,
    )


@pytest.mark.pad
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend addressing")
def test_ascend_pad_strided_constant_safe_masked_addresses():
    generator = torch.Generator(device="cpu").manual_seed(20261014)
    base = torch.randn((1, 64, 112, 112), dtype=torch.float32, generator=generator)
    x_cpu = base[::2, ::2]
    x = base.to(flag_gems.device)[::2, ::2]
    assert not x.is_contiguous()
    padding = (3, 5, 4, 2, 1, 7, 2, 3)
    expected = torch.nn.functional.pad(x_cpu, padding, mode="constant", value=27.0)
    actual = flag_gems.pad(x, padding, mode="constant", value=27.0)
    assert torch.equal(actual.cpu(), expected)
