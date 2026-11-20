# Copyright 2026, The FlagOS Contributors.
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

# CI derives the pytest marker from the implementation file name via the
# #6359 convention; the stripped name also needs a marker, so declare both.
pytestmark = [
    pytest.mark.make_dual_copy,
]


def _ref(primal, tangent, level=0, out=None):
    """Native reference; it only runs on inference tensors, so build them here."""
    with torch.inference_mode():
        p = primal.clone(memory_format=torch.contiguous_format)
        t = tangent.clone(memory_format=torch.contiguous_format)
        if out is None:
            return torch.ops.aten._make_dual_copy(p, t, level)
        dst = torch.empty_like(p)
        torch.ops.aten._make_dual_copy.out(p, t, level, out=dst)
        return dst


@pytest.mark.make_dual_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_make_dual_copy(shape, dtype):
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    primal = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    tangent = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = _ref(primal, tangent)
    res_out = flag_gems._make_dual_copy(primal, tangent, 0)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    # The copy must not alias the input.
    assert res_out.data_ptr() != primal.data_ptr()
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.make_dual_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_make_dual_copy_int(shape, dtype):
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    primal = torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
    tangent = torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)

    ref_out = _ref(primal, tangent)
    res_out = flag_gems._make_dual_copy(primal, tangent, 0)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.make_dual_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_make_dual_copy_complex(shape, dtype):
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    primal = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    tangent = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = _ref(primal, tangent)
    res_out = flag_gems._make_dual_copy(primal, tangent, 0)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.make_dual_copy
def test_make_dual_copy_noncontiguous():
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    primal = torch.randn((16, 32), device=flag_gems.device)[:, ::2]
    tangent = torch.randn((16, 32), device=flag_gems.device)[:, ::2]

    ref_out = _ref(primal, tangent)
    res_out = flag_gems._make_dual_copy(primal, tangent, 0)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.make_dual_copy
def test_make_dual_copy_empty():
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    primal = torch.empty((0, 4), device=flag_gems.device)
    tangent = torch.empty((0, 4), device=flag_gems.device)

    res_out = flag_gems._make_dual_copy(primal, tangent, 0)

    assert tuple(res_out.shape) == (0, 4)
    assert res_out.numel() == 0


@pytest.mark.make_dual_copy_out
def test_make_dual_copy_out():
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    primal = torch.randn((16, 16), device=flag_gems.device)
    tangent = torch.randn((16, 16), device=flag_gems.device)

    ref_out = _ref(primal, tangent)
    res_out = torch.empty_like(primal)
    res_ret = flag_gems._make_dual_copy_out(primal, tangent, 0, out=res_out)

    assert res_ret is res_out
    utils.gems_assert_close(res_out, ref_out, torch.float32)
