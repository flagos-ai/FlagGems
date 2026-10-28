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


_EQUAL_SHAPES = [*utils.POINTWISE_SHAPES, (257,), (2, 0, 3)]
_BROADCAST_CASES = [
    pytest.param((7, 13), (1, 13), id="leading-dimension"),
    pytest.param((2, 3, 5), (1, 3, 1), id="multiple-dimensions"),
    pytest.param((2, 0, 3), (1, 0, 1), id="zero-size"),
]


def _make_positive(shape, dtype):
    return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(1.0, 3.0)


def _assert_inplace_result(result, mutated_input, original_pointer, reference, dtype):
    assert result is mutated_input
    assert result.data_ptr() == original_pointer
    assert result.shape == reference.shape
    assert result.dtype == dtype
    utils.gems_assert_close(
        result, reference, dtype, equal_nan=True, atol=1e-3
    )


@pytest.mark.inplace
@pytest.mark.igamma_
@pytest.mark.parametrize("shape", _EQUAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_igamma_equal_shape_(shape, dtype):
    torch.manual_seed(0)
    inp = _make_positive(shape, dtype)
    other = _make_positive(shape, dtype)
    other_before = other.clone()
    original_pointer = inp.data_ptr()

    ref_inp = utils.to_reference(inp.clone(), True)
    ref_other = utils.to_reference(other, True)
    ref_out = ref_inp.igamma_(ref_other)

    result = flag_gems.igamma_(inp, other)

    _assert_inplace_result(result, inp, original_pointer, ref_out, dtype)
    utils.gems_assert_equal(other, utils.to_reference(other_before))


@pytest.mark.inplace
@pytest.mark.igamma_
@pytest.mark.parametrize("self_shape,other_shape", _BROADCAST_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_igamma_broadcast_(self_shape, other_shape, dtype):
    torch.manual_seed(0)
    inp = _make_positive(self_shape, dtype)
    other = _make_positive(other_shape, dtype)
    original_pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_other = utils.to_reference(other, True)
    ref_out = ref_inp.igamma_(ref_other)

    result = flag_gems.igamma_(inp, other)

    _assert_inplace_result(result, inp, original_pointer, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.igamma_
def test_igamma_noncontiguous_():
    torch.manual_seed(0)
    inp = _make_positive((4, 7, 9), torch.float32).transpose(0, 2)
    other = _make_positive(inp.shape, torch.float32)
    original_pointer = inp.data_ptr()
    original_stride = inp.stride()
    assert not inp.is_contiguous()

    ref_inp = utils.to_reference(inp.clone(), True)
    ref_other = utils.to_reference(other, True)
    ref_out = ref_inp.igamma_(ref_other)

    result = flag_gems.igamma_(inp, other)

    assert result.stride() == original_stride
    _assert_inplace_result(
        result, inp, original_pointer, ref_out, torch.float32
    )


@pytest.mark.inplace
@pytest.mark.igamma_
def test_igamma_boundary_values_():
    inp = torch.tensor(
        [0.5, 1.0, 2.0, 4.0], dtype=torch.float32, device=flag_gems.device
    )
    other = torch.tensor(
        [0.0, float("inf"), 1.0, 8.0],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    original_pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_other = utils.to_reference(other, True)
    ref_out = ref_inp.igamma_(ref_other)

    result = flag_gems.igamma_(inp, other)

    _assert_inplace_result(
        result, inp, original_pointer, ref_out, torch.float32
    )


@pytest.mark.inplace
@pytest.mark.igamma_
def test_igamma_rejects_nonbroadcastable_shape_():
    inp = torch.ones((2, 3), device=flag_gems.device)
    other = torch.ones((4,), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.igamma_(inp, other)
