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


def _function_from(module):
    def fn():
        pass

    fn.__module__ = module
    return fn


@pytest.mark.parametrize(
    "module, expected",
    [
        ("flag_gems.ops.add", "GEMS"),
        ("_test_vendor.ops.add", "GEMS_TEST_VENDOR"),
        ("flag_gems.runtime.backend._test_vendor.ops.add", "GEMS_TEST_VENDOR"),
        ("_test_vendor.fused.nested.add", "GEMS_TEST_VENDOR"),
        ("_test_vendor_extra.ops.add", "GEMS"),
    ],
)
def test_gems_log_prefix(monkeypatch, module, expected):
    monkeypatch.setattr(flag_gems, "vendor_name", "test_vendor")
    assert utils.gems_log_prefix(_function_from(module)) == expected


@pytest.mark.parametrize("arch", ["gcu300", "gcu400"])
def test_gems_log_prefix_arch_module(monkeypatch, arch):
    # Architecture packages are imported without the vendor module component.
    monkeypatch.setattr(flag_gems, "vendor_name", "enflame")
    assert (
        utils.gems_log_prefix(_function_from(f"{arch}.ops.native_layer_norm")) == "GEMS"
    )


@pytest.mark.parametrize(
    "module",
    [
        "flag_gems.ops.add",
        "_test_vendor.ops.nested.add",
        "flag_gems.runtime.backend._test_vendor.ops.add",
    ],
)
def test_gems_log_logger(module):
    assert utils.gems_log_logger(_function_from(module)) == module


@pytest.mark.parametrize("supports_fp64", [False, True])
@pytest.mark.parametrize(
    "dtype, mode, numerator, denominator, expected",
    [
        (torch.float16, "floor", 9.8984375, 0.01512908935546875, 654.0),
        (torch.float16, "trunc", 8.3125, 1.6630859375, 4.0),
        (torch.bfloat16, "floor", 8.6875, -0.036376953125, -239.0),
        (torch.bfloat16, "trunc", -7.9375, 0.0771484375, -102.0),
    ],
)
def test_ref_div_rounding_low_precision(
    monkeypatch, supports_fp64, dtype, mode, numerator, denominator, expected
):
    monkeypatch.setattr(flag_gems, "vendor_name", "test_vendor")
    monkeypatch.setattr(utils, "fp64_is_supported", supports_fp64)
    a = torch.tensor([numerator], dtype=dtype)
    b = torch.tensor([denominator], dtype=dtype)

    result = utils.ref_div_rounding(a, b, mode)

    assert result.dtype == dtype
    assert result.device.type == "cpu"
    assert result.item() == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
@pytest.mark.parametrize("mode", [None, "floor", "trunc"])
def test_ref_div_rounding_regular_dtypes(monkeypatch, dtype, mode):
    monkeypatch.setattr(flag_gems, "vendor_name", "test_vendor")
    a = torch.tensor([7, -7, 0], dtype=dtype)
    b = torch.tensor([2, 2, 2], dtype=dtype)

    torch.testing.assert_close(
        utils.ref_div_rounding(a, b, mode), torch.div(a, b, rounding_mode=mode)
    )


@pytest.mark.parametrize("scalar_first", [False, True])
@pytest.mark.parametrize("mode", [None, "floor", "trunc"])
def test_ref_div_rounding_scalar_operands(monkeypatch, scalar_first, mode):
    monkeypatch.setattr(flag_gems, "vendor_name", "test_vendor")
    tensor = torch.tensor([7.0, -7.0], dtype=torch.float16)
    a, b = (2.0, tensor) if scalar_first else (tensor, 2.0)
    wide_a, wide_b = (2.0, tensor.double()) if scalar_first else (tensor.double(), 2.0)
    expected = torch.div(wide_a, wide_b, rounding_mode=mode).to(torch.float16)

    torch.testing.assert_close(utils.ref_div_rounding(a, b, mode), expected)


@pytest.mark.parametrize("to_cpu", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ref_div_rounding_kunlunxin_uses_cpu_semantics(monkeypatch, to_cpu, dtype):
    monkeypatch.setattr(flag_gems, "vendor_name", "kunlunxin")
    monkeypatch.setattr(utils, "TO_CPU", to_cpu)
    a = torch.tensor([9.8984375, -7.9375], dtype=dtype)
    b = torch.tensor([0.01512908935546875, 0.0771484375], dtype=dtype)

    torch.testing.assert_close(
        utils.ref_div_rounding(a, b, "floor"), torch.div(a, b, rounding_mode="floor")
    )
