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


def _reference(inp, exponent):
    if flag_gems.vendor_name == "enflame":
        # The target native ldexp DispatchStub has no GCU kernel. Preserve
        # the chosen reference precision while evaluating the same ATen CPU API.
        inp, exponent = inp.cpu(), exponent.cpu()
    return torch.ops.aten.ldexp.Tensor(inp, exponent).to(inp.dtype)


@pytest.mark.ldexp_
@pytest.mark.parametrize("shape", [*utils.POINTWISE_SHAPES, (2, 0, 3)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ldexp_(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    exponent = torch.randint(-8, 9, shape, dtype=torch.int32, device=inp.device)
    original_pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone(), True)
    reference = _reference(ref_inp, utils.to_reference(exponent, False))

    expected_device = inp.device
    result = flag_gems.ldexp_(inp, exponent)

    assert result is inp
    assert result.device == expected_device
    assert result.data_ptr() == original_pointer
    assert result.dtype == dtype
    utils.gems_assert_close(result.cpu(), reference.cpu(), dtype)


@pytest.mark.ldexp_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ldexp_noncontiguous_broadcast_(dtype):
    torch.manual_seed(0)
    inp = torch.randn((19, 7), dtype=dtype, device=flag_gems.device).T
    exponent = torch.randint(-4, 5, (19,), dtype=torch.int64, device=inp.device)
    original_pointer = inp.data_ptr()
    original_stride = inp.stride()
    reference = _reference(
        utils.to_reference(inp.clone(), True), utils.to_reference(exponent, False)
    )

    expected_device = inp.device
    result = flag_gems.ldexp_(inp, exponent)

    assert result is inp
    assert result.device == expected_device
    assert result.data_ptr() == original_pointer
    assert result.stride() == original_stride
    utils.gems_assert_close(result.cpu(), reference.cpu(), dtype)


@pytest.mark.ldexp_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ldexp_special_values_(dtype):
    inp = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    exponent = torch.tensor([128, -128, 0, 1, -1, 0, 2], device=inp.device)
    reference = _reference(
        utils.to_reference(inp.clone(), True), utils.to_reference(exponent, False)
    )

    expected_device = inp.device
    result = flag_gems.ldexp_(inp, exponent)

    assert result is inp
    assert result.device == expected_device
    utils.gems_assert_close(result.cpu(), reference.cpu(), dtype, equal_nan=True)


@pytest.mark.ldexp_
@pytest.mark.parametrize("self_dtype", [torch.bool, torch.int32])
def test_ldexp_rejects_integer_self_(self_dtype):
    inp = torch.ones((17,), dtype=self_dtype, device=flag_gems.device)
    exponent = torch.ones((17,), dtype=torch.int32, device=inp.device)
    with pytest.raises(RuntimeError):
        flag_gems.ldexp_(inp, exponent)


@pytest.mark.ldexp_
def test_ldexp_rejects_expanding_self_():
    inp = torch.ones((1, 7), device=flag_gems.device)
    exponent = torch.ones((3, 7), dtype=torch.int32, device=inp.device)
    with pytest.raises(RuntimeError):
        flag_gems.ldexp_(inp, exponent)
