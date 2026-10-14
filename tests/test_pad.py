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

import random
import time

import pytest
import torch
import torch.nn.functional as F

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_equal, to_reference

random.seed(time.time() // 100)

device = flag_gems.device


@pytest.mark.pad
@pytest.mark.parametrize(
    "shape",
    [[1024, 1024], [64, 64, 64, 64], [1, 64, 112, 112], [4, 64, 128]],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("pad_mode", ["constant", "reflect", "replicate", "circular"])
@pytest.mark.parametrize("contiguous", [True, False])
def test_pad(shape, dtype, pad_mode, contiguous):
    rank = len(shape)
    if pad_mode != "constant" and rank < 3:
        # Invalid combination: PyTorch non-constant padding requires 3D+ input tensors
        return
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.bfloat16:
        pytest.skip("Issue #3794: not working")

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    if not contiguous:
        # BUG #2835
        if flag_gems.vendor_name == "kunlunxin":
            x = x.cpu()[::2, ::2].to(flag_gems.device)
        else:
            x = x[::2, ::2]

    ref_x = to_reference(x)
    if ref_x.dtype == torch.float16:
        ref_x = ref_x.to(torch.float32)

    rank = x.ndim
    if pad_mode == "constant":
        num_pad = rank * 2
    else:
        # Non-constant modes only pad last (rank-1) dims, up to 3 dims max.
        # For 2D: pad last 1 dim (2 values); 3D: pad last 2 dims (4 values);
        # 4D+: pad last 3 dims (6 values).
        num_pad = min(rank - 1, 3) * 2
    pad_params = torch.randint(0, 10, (num_pad,), dtype=torch.int32, device="cpu")
    pad_value = float(torch.randint(0, 1024, (1,), dtype=torch.int32, device="cpu"))

    if pad_mode != "constant":
        # Clamp each pad value to be valid for reflect (< dim) / circular (<= dim).
        for i in range(num_pad // 2):
            dim_size = x.shape[rank - 1 - i]
            max_pad = dim_size - 1 if pad_mode == "reflect" else dim_size
            pad_params[2 * i] = int(pad_params[2 * i]) % max(max_pad, 1)
            pad_params[2 * i + 1] = int(pad_params[2 * i + 1]) % max(max_pad, 1)
        pad_value = None

    # Convert pad_params to list of Python ints for torch.nn.functional.pad
    pad_params_list = [int(pad_params[i]) for i in range(pad_params.shape[0])]

    ref_out = torch.nn.functional.pad(ref_x, pad_params_list, pad_mode, pad_value)
    res_out = flag_gems.pad(x, pad_params_list, pad_mode, pad_value)

    if ref_out.dtype != res_out.dtype:
        ref_out = ref_out.to(res_out.dtype)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend kernel regression cases"
)
@pytest.mark.pad
@pytest.mark.parametrize("shape", [(0, 5), (2, 0)])
def test_ascend_constant_pad_empty_input(shape):
    x = torch.empty(shape, device=flag_gems.device)
    expected = torch.nn.functional.pad(x, (2, 3, 1, 1), value=2.5)
    actual = flag_gems.pad(x, (2, 3, 1, 1), value=2.5)
    assert torch.equal(actual, expected)


@pytest.mark.pad
@pytest.mark.skipif(
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


@pytest.mark.pad
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend addressing")
@pytest.mark.parametrize("mode", ["replicate", "reflect", "circular"])
def test_ascend_pad_address_span_guard_and_wide_fallback(monkeypatch, mode):
    import importlib

    module = importlib.import_module(flag_gems.pad.__module__)
    guard = module._pad_can_use_int32
    assert guard((2, 3), (3, 1), (4, 5), (1, 1), (1, 1), 20, 1024)
    assert not guard((2,), (2**31,), (2,), (0,), (0,), 2, 1024)
    assert not guard((2**31 - 1,), (1,), (2**31 - 1,), (0,), (0,), 2**31 - 1, 1024)
    monkeypatch.setattr(module, "_pad_can_use_int32", lambda *args: False)
    x = torch.arange(24, device=flag_gems.device).reshape(2, 3, 4).transpose(1, 2)
    expected = torch.nn.functional.pad(x.cpu(), (-1, 2), mode=mode)
    assert torch.equal(flag_gems.pad(x, (-1, 2), mode=mode).cpu(), expected)


@pytest.mark.pad
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend addressing")
def test_ascend_pad_large_reflect_exact_coordinates():
    generator = torch.Generator(device="cpu").manual_seed(20261014)
    x_cpu = torch.randn((64, 64, 64, 64), dtype=torch.float16, generator=generator)
    padding = (3, 5, 4, 2, 1, 7)
    expected = F.pad(x_cpu.float(), padding, mode="reflect").to(x_cpu.dtype)
    assert expected.numel() > 2**24
    actual = flag_gems.pad(x_cpu.to(flag_gems.device), padding, mode="reflect")
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.pad
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend addressing")
def test_ascend_pad_strided_constant_safe_masked_addresses():
    generator = torch.Generator(device="cpu").manual_seed(20261014)
    base = torch.randn((1, 64, 112, 112), dtype=torch.float32, generator=generator)
    x_cpu = base[::2, ::2]
    x = base.to(flag_gems.device)[::2, ::2]
    assert not x.is_contiguous()
    padding = (3, 5, 4, 2, 1, 7, 2, 3)
    expected = F.pad(x_cpu, padding, mode="constant", value=27.0)
    actual = flag_gems.pad(x, padding, mode="constant", value=27.0)
    assert torch.equal(actual.cpu(), expected)
