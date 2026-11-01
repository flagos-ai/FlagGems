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


@pytest.mark.eq
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_eq(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.eq(ref_inp1, ref_inp2)
    res_out = flag_gems.eq(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.eq_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_eq_scalar(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    inp2 = 0
    ref_inp1 = utils.to_reference(inp1)

    ref_out = torch.eq(ref_inp1, inp2)
    res_out = flag_gems.eq_scalar(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.eq
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_eq_integer_precision(dtype):
    # Issue #7584: distinct integers beyond the fp32 mantissa (2**24) must
    # not collapse to the same fp32 value and compare equal.
    base = torch.arange(2**24 - 2, 2**24 + 3, device=flag_gems.device, dtype=dtype)
    inp1 = torch.cat([base, -base], dim=0).unsqueeze(1)
    inp2 = inp1 + 1

    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.eq(ref_inp1, ref_inp2)
    res_out = flag_gems.eq(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)
    assert not res_out.any().item()


@pytest.mark.eq_scalar
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_eq_scalar_integer_precision(dtype):
    inp = torch.arange(2**24, 2**24 + 4, device=flag_gems.device, dtype=dtype).unsqueeze(0)
    scalar = 2**24 + 1  # collides with 2**24 after an fp32 cast

    ref_inp = utils.to_reference(inp)
    ref_out = torch.eq(ref_inp, scalar)
    res_out = flag_gems.eq_scalar(inp, scalar)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.sum().item() == 1
