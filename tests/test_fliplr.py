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


@pytest.mark.fliplr
@pytest.mark.parametrize(
    "shape",
    [(0, 0), (0, 3), (1, 1), (2, 3), (3, 2), (32, 64), (4, 3, 5), (2, 3, 4, 5)],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_fliplr(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)
    original = utils.to_reference(inp.clone(), False)

    expected = torch.fliplr(ref_inp)
    with flag_gems.use_gems():
        result = torch.fliplr(inp)

    utils.gems_assert_equal(result, expected)
    utils.gems_assert_equal(inp, original)
    assert result.shape == inp.shape
    assert result.stride() == expected.stride()
    if inp.numel() > 0:
        assert result.data_ptr() != inp.data_ptr()


@pytest.mark.fliplr
@pytest.mark.parametrize(
    ("dtype", "high"),
    [(torch.int32, 17), (torch.int64, 17), (torch.bool, 2)],
)
def test_accuracy_fliplr_non_float(dtype, high):
    inp = torch.randint(0, high, (7, 11), dtype=dtype, device=flag_gems.device)
    expected = torch.fliplr(utils.to_reference(inp, False))

    with flag_gems.use_gems():
        result = torch.fliplr(inp)

    utils.gems_assert_equal(result, expected)


def _make_transposed(dtype):
    return torch.randn((5, 7), dtype=dtype, device=flag_gems.device).T


def _make_sliced(dtype):
    return torch.randn((12, 18), dtype=dtype, device=flag_gems.device)[::2, ::3]


def _make_expanded(dtype):
    return torch.randn((6, 1), dtype=dtype, device=flag_gems.device).expand(6, 9)


@pytest.mark.fliplr
@pytest.mark.parametrize("make_input", [_make_transposed, _make_sliced, _make_expanded])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_fliplr_noncontiguous(make_input, dtype):
    inp = make_input(dtype)
    ref_inp = utils.to_reference(inp, False)
    expected = torch.fliplr(ref_inp)

    with flag_gems.use_gems():
        result = torch.fliplr(inp)

    utils.gems_assert_equal(result, expected)
    assert result.stride() == expected.stride()


@pytest.mark.fliplr
@pytest.mark.parametrize("shape", [(), (0,), (8,)])
def test_fliplr_rejects_inputs_with_fewer_than_two_dims(shape):
    inp = torch.empty(shape, device=flag_gems.device)
    with (
        flag_gems.use_gems(),
        pytest.raises(RuntimeError, match="Input must be >= 2-d"),
    ):
        torch.fliplr(inp)
