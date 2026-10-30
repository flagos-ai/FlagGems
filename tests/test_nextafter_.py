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

# Kernel uses generic int bitcast based on input dtype bitwidth;
# supports fp16, bf16, fp32, and fp64 on devices with FP64 support.
_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
if flag_gems.runtime.device.support_fp64:
    _DTYPES.append(torch.float64)


@pytest.mark.nextafter_
@pytest.mark.parametrize("shape", [*utils.POINTWISE_SHAPES, (2, 0, 3)])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_nextafter_(shape, dtype):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and not utils.TO_CPU
        and dtype == torch.float16
        and 0 not in shape
    ):
        pytest.skip("Kunlunxin native Half nextafter loses original-dtype ULPs")
    if flag_gems.vendor_name == "cambricon" and dtype == torch.float64:
        pytest.skip("Issue #5253: FP64 nextafter is not supported")
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=inp.device)
    original_pointer = inp.data_ptr()
    reference = utils.to_reference(inp.clone(), False)
    reference.nextafter_(utils.to_reference(other, False))

    result = flag_gems.nextafter_(inp, other)

    assert result is inp
    assert result.data_ptr() == original_pointer
    utils.gems_assert_equal(result, reference)


@pytest.mark.nextafter_
@pytest.mark.parametrize("dtype", _DTYPES)
def test_nextafter_noncontiguous_broadcast_(dtype):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and not utils.TO_CPU
        and dtype == torch.float16
    ):
        pytest.skip("Kunlunxin native Half nextafter loses original-dtype ULPs")
    if flag_gems.vendor_name == "cambricon" and dtype == torch.float64:
        pytest.skip("Issue #5253: FP64 nextafter is not supported")
    torch.manual_seed(0)
    inp = torch.randn((19, 7), dtype=dtype, device=flag_gems.device).T
    other = torch.randn((19,), dtype=dtype, device=inp.device)
    original_pointer = inp.data_ptr()
    original_stride = inp.stride()
    reference = utils.to_reference(inp.clone(), False)
    reference.nextafter_(utils.to_reference(other, False))

    result = flag_gems.nextafter_(inp, other)

    assert result is inp
    assert result.data_ptr() == original_pointer
    assert result.stride() == original_stride
    utils.gems_assert_equal(result, reference)


@pytest.mark.nextafter_
@pytest.mark.parametrize("dtype", _DTYPES)
def test_nextafter_boundaries_(dtype):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and not utils.TO_CPU
        and dtype == torch.float16
    ):
        pytest.skip("Kunlunxin native Half nextafter loses original-dtype ULPs")
    if flag_gems.vendor_name == "cambricon" and dtype == torch.float64:
        pytest.skip("Issue #5253: FP64 nextafter is not supported")
    inp = torch.tensor(
        [0.0, -0.0, 1.0, float("inf"), -float("inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    other = torch.tensor(
        [-1.0, 1.0, float("inf"), 0.0, 0.0, 1.0],
        dtype=dtype,
        device=inp.device,
    )
    reference = utils.to_reference(inp.clone(), False)
    reference.nextafter_(utils.to_reference(other, False))

    result = flag_gems.nextafter_(inp, other)

    assert result is inp
    utils.gems_assert_equal(result, reference, equal_nan=True)


@pytest.mark.nextafter_
def test_nextafter_rejects_expanding_self_():
    inp = torch.ones((1, 7), device=flag_gems.device)
    other = torch.ones((3, 7), device=inp.device)
    with pytest.raises(RuntimeError):
        flag_gems.nextafter_(inp, other)
