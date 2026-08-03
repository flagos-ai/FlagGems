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

RSHIFT_DTYPES = utils.ALL_INT_DTYPES + [torch.uint8]


@pytest.mark.rshift
# Covers one-, two-, and three-dimensional pointwise inputs.
@pytest.mark.parametrize("shape", [(1024,), (7, 13), (2, 3, 5)])
@pytest.mark.parametrize("dtype", RSHIFT_DTYPES)
def test_rshift_tensor(dtype, shape):
    value = torch.randint(0, 100, shape, dtype=dtype, device=flag_gems.device)
    shift = torch.randint(0, 7, shape, dtype=dtype, device=flag_gems.device)
    expected = torch.ops.aten.__rshift__.Tensor(
        utils.to_reference(value), utils.to_reference(shift)
    )

    with flag_gems.use_gems():
        actual = torch.ops.aten.__rshift__.Tensor(value, shift)

    utils.gems_assert_equal(actual, expected)


@pytest.mark.rshift
@pytest.mark.parametrize("dtype", RSHIFT_DTYPES)
def test_rshift_scalar(dtype):
    value = torch.randint(0, 100, (11, 17), dtype=dtype, device=flag_gems.device)
    expected = torch.ops.aten.__rshift__.Scalar(utils.to_reference(value), 3)

    with flag_gems.use_gems():
        actual = torch.ops.aten.__rshift__.Scalar(value, 3)

    utils.gems_assert_equal(actual, expected)


@pytest.mark.rshift
@pytest.mark.parametrize("dtype", RSHIFT_DTYPES)
def test_rshift_output_overloads(dtype):
    value = torch.randint(0, 100, (9, 13), dtype=dtype, device=flag_gems.device)
    shift = torch.randint(0, 7, value.shape, dtype=dtype, device=flag_gems.device)
    expected_tensor = utils.to_reference(value) >> utils.to_reference(shift)
    expected_scalar = utils.to_reference(value) >> 2
    tensor_out = torch.empty_like(value)
    scalar_out = torch.empty_like(value)

    with flag_gems.use_gems():
        tensor_result = torch.ops.aten.__rshift__.Tensor_out(
            value, shift, out=tensor_out
        )
        scalar_result = torch.ops.aten.__rshift__.Scalar_out(value, 2, out=scalar_out)

    assert tensor_result is tensor_out
    assert scalar_result is scalar_out
    utils.gems_assert_equal(tensor_out, expected_tensor)
    utils.gems_assert_equal(scalar_out, expected_scalar)
