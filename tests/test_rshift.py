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
from flag_gems.experimental_ops.__rshift__ import (
    rshift_scalar,
    rshift_scalar_,
    rshift_tensor,
    rshift_tensor_,
)

from . import accuracy_utils as utils


@pytest.mark.rshift
# Covers one-, two-, and three-dimensional pointwise inputs.
@pytest.mark.parametrize("shape", [(1024,), (7, 13), (2, 3, 5)])
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES + [torch.uint8])
def test_rshift_tensor_and_inplace(dtype, shape):
    value = torch.randint(0, 100, shape, dtype=dtype, device=flag_gems.device)
    shift = torch.randint(0, 7, shape, dtype=dtype, device=flag_gems.device)
    expected = utils.to_reference(value) >> utils.to_reference(shift)

    utils.gems_assert_equal(rshift_tensor(value, shift), expected)

    actual = value.clone()
    result = rshift_tensor_(actual, shift)
    assert result is actual
    utils.gems_assert_equal(actual, expected)


@pytest.mark.rshift
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES + [torch.uint8])
def test_rshift_scalar_and_inplace(dtype):
    value = torch.randint(0, 100, (11, 17), dtype=dtype, device=flag_gems.device)
    expected = utils.to_reference(value) >> 3

    utils.gems_assert_equal(rshift_scalar(value, 3), expected)

    actual = value.clone()
    result = rshift_scalar_(actual, 3)
    assert result is actual
    utils.gems_assert_equal(actual, expected)
