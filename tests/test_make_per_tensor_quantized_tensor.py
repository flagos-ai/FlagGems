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
    pytest.mark.make_per_tensor_quantized_tensor,
]

_QINT = [torch.uint8, torch.int8, torch.int32]


def _ref(inp, scale, zero_point):
    return torch.ops.aten._make_per_tensor_quantized_tensor(inp, scale, zero_point)


def _rand_int(shape, dtype):
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-128, 128, shape, dtype=dtype, device=flag_gems.device)


@pytest.mark.make_per_tensor_quantized_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", _QINT)
def test_make_per_tensor_quantized_tensor(shape, dtype):
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _rand_int(shape, dtype)

    ref_out = _ref(inp, 0.25, 7)
    res_out = flag_gems._make_per_tensor_quantized_tensor(inp, 0.25, 7)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.qscheme() == ref_out.qscheme()
    assert res_out.q_scale() == pytest.approx(ref_out.q_scale())
    assert res_out.q_zero_point() == ref_out.q_zero_point()
    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())


@pytest.mark.make_per_tensor_quantized_tensor
def test_make_per_tensor_quantized_tensor_noncontiguous():
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _rand_int((16, 32), torch.uint8)[:, ::2]

    ref_out = _ref(inp, 1.5, 0)
    res_out = flag_gems._make_per_tensor_quantized_tensor(inp, 1.5, 0)

    assert res_out.shape == ref_out.shape
    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())


@pytest.mark.make_per_tensor_quantized_tensor
def test_make_per_tensor_quantized_tensor_empty():
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = torch.empty((0, 4), dtype=torch.uint8, device=flag_gems.device)

    res_out = flag_gems._make_per_tensor_quantized_tensor(inp, 2.0, 5)

    assert tuple(res_out.shape) == (0, 4)
    assert res_out.dtype == torch.quint8
    assert res_out.q_scale() == pytest.approx(2.0)
    assert res_out.q_zero_point() == 5


@pytest.mark.make_per_tensor_quantized_tensor_out
def test_make_per_tensor_quantized_tensor_out():
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    inp = _rand_int((16, 16), torch.int8)

    ref_out = _ref(inp, 0.5, 3)
    res_out = torch._empty_affine_quantized(
        inp.shape,
        dtype=torch.qint8,
        device=flag_gems.device,
        scale=0.5,
        zero_point=3,
    )
    res_ret = flag_gems._make_per_tensor_quantized_tensor_out(inp, 0.5, 3, out=res_out)

    assert res_ret is res_out
    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())
