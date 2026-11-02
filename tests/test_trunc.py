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

# Values that sit exactly on a truncation boundary or are not finite.
SPECIAL_VALUES = [
    -float("inf"),
    -2.5,
    -1.5,
    -1.0,
    -0.5,
    -0.0,
    0.0,
    0.5,
    1.0,
    1.5,
    2.5,
    float("inf"),
    float("nan"),
]


@pytest.mark.trunc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trunc(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.trunc(utils.to_reference(inp))
    res_out = flag_gems.trunc(inp)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == inp.dtype


@pytest.mark.trunc_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trunc_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.trunc_()
    res_out = flag_gems.trunc_(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.trunc
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trunc_special_values(dtype):
    """Boundary and non-finite inputs must pass through untouched."""
    inp = torch.tensor(SPECIAL_VALUES, dtype=dtype, device=flag_gems.device)

    ref_out = torch.trunc(utils.to_reference(inp))
    res_out = flag_gems.trunc(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.trunc
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trunc_always_rounds_towards_zero(dtype):
    """The sign of a truncated value never flips, and -0.0 stays negative."""
    inp = torch.tensor([-0.5, -0.0, 0.0, 0.5], dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.trunc(inp)

    assert (res_out == 0).all()
    # -0.5 -> -0.0, not +0.0: truncation is towards zero, never across it.
    assert torch.signbit(res_out[0]).item()
    assert torch.signbit(res_out[1]).item()
    assert not torch.signbit(res_out[3]).item()


@pytest.mark.trunc
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_trunc_integer_input(dtype):
    """Integer inputs are already integral, so they come back unchanged."""
    inp = torch.randint(-100, 100, (32, 17), dtype=dtype, device=flag_gems.device)

    ref_out = torch.trunc(utils.to_reference(inp))
    res_out = flag_gems.trunc(inp)

    assert res_out.dtype == dtype
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.trunc
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trunc_non_contiguous(dtype):
    """The pointwise kernel has to read strided input correctly."""
    inp = torch.randn(16, 32, dtype=dtype, device=flag_gems.device).transpose(0, 1)
    assert not inp.is_contiguous()

    ref_out = torch.trunc(utils.to_reference(inp))
    res_out = flag_gems.trunc(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.trunc
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trunc_empty(dtype):
    inp = torch.empty(0, dtype=dtype, device=flag_gems.device)

    ref_out = torch.trunc(utils.to_reference(inp))
    res_out = flag_gems.trunc(inp)

    assert res_out.shape == (0,)
    assert res_out.dtype == dtype
    utils.gems_assert_equal(res_out, ref_out)
