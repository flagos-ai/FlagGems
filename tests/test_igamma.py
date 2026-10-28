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
    pytest.param((2, 3, 5), (), id="scalar-tensor"),
    pytest.param((2, 0, 3), (1, 0, 1), id="zero-size"),
]


def _make_positive(shape, dtype, *, offset=0.1):
    return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
        offset, 2.0 + offset
    )


@pytest.mark.igamma
@pytest.mark.parametrize("shape", _EQUAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_igamma_equal_shape(shape, dtype):
    torch.manual_seed(0)
    inp = _make_positive(shape, dtype)
    other = _make_positive(shape, dtype)
    inp_before = inp.clone()
    other_before = other.clone()

    ref_inp = utils.to_reference(inp, True)
    ref_other = utils.to_reference(other, True)
    ref_out = torch.igamma(ref_inp, ref_other)

    result = flag_gems.igamma(inp, other)

    assert result.shape == torch.Size(shape)
    assert result.dtype == dtype
    utils.gems_assert_close(result, ref_out, dtype, equal_nan=True, atol=1e-3)
    utils.gems_assert_equal(inp, utils.to_reference(inp_before))
    utils.gems_assert_equal(other, utils.to_reference(other_before))


@pytest.mark.igamma
@pytest.mark.parametrize("self_shape,other_shape", _BROADCAST_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_igamma_broadcast(self_shape, other_shape, dtype):
    torch.manual_seed(0)
    inp = _make_positive(self_shape, dtype)
    other = _make_positive(other_shape, dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_other = utils.to_reference(other, True)
    ref_out = torch.igamma(ref_inp, ref_other)

    result = flag_gems.igamma(inp, other)

    assert result.shape == ref_out.shape
    utils.gems_assert_close(result, ref_out, dtype, equal_nan=True, atol=1e-3)


@pytest.mark.igamma
def test_igamma_noncontiguous():
    torch.manual_seed(0)
    inp = _make_positive((4, 7, 9), torch.float32).transpose(0, 2)
    other = _make_positive(inp.shape, torch.float32)
    assert not inp.is_contiguous()
    ref_out = torch.igamma(
        utils.to_reference(inp, True), utils.to_reference(other, True)
    )

    result = flag_gems.igamma(inp, other)

    utils.gems_assert_close(result, ref_out, torch.float32, atol=1e-3)


@pytest.mark.igamma
def test_igamma_boundary_values():
    inp = torch.tensor(
        [0.5, 1.0, 2.0, 4.0], dtype=torch.float32, device=flag_gems.device
    )
    other = torch.tensor(
        [0.0, float("inf"), 1.0, 8.0],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref_out = torch.igamma(
        utils.to_reference(inp, True), utils.to_reference(other, True)
    )

    result = flag_gems.igamma(inp, other)

    utils.gems_assert_close(
        result, ref_out, torch.float32, equal_nan=True, atol=1e-3
    )


@pytest.mark.igamma
def test_igamma_rejects_nonbroadcastable_shape():
    inp = torch.ones((2, 3), device=flag_gems.device)
    other = torch.ones((4,), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.igamma(inp, other)
