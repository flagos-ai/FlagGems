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


@pytest.mark.xlogy_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # keep ``other`` positive so ``log`` stays finite for a clean comparison
    y = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 5.0 + 0.01

    ref_x = utils.to_reference(x.clone(), True)
    ref_y = utils.to_reference(y, True)
    ref_out = ref_x.xlogy_(ref_y)

    with flag_gems.use_gems():
        res_out = x.xlogy_(y)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(x, ref_x, dtype)
    assert res_out is x


@pytest.mark.xlogy_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_special_values_(dtype):
    # Exercise the PyTorch precedence: NaN(other) -> NaN; x == 0 -> 0; else x*log(y)
    x = torch.tensor([0.0, 0.0, 2.0, 3.0, 0.0], dtype=dtype, device=flag_gems.device)
    y = torch.tensor(
        [5.0, 0.0, 4.0, float("nan"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_x = utils.to_reference(x.clone(), True)
    ref_y = utils.to_reference(y, True)
    ref_out = ref_x.xlogy_(ref_y)

    with flag_gems.use_gems():
        res_out = x.xlogy_(y)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_close(x, ref_x, dtype, equal_nan=True)
    assert res_out is x


@pytest.mark.xlogy_tensor_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_tensor_scalar_(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    scalar = 3.5

    ref_x = utils.to_reference(x.clone(), True)
    ref_out = ref_x.xlogy_(scalar)

    with flag_gems.use_gems():
        res_out = x.xlogy_(scalar)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(x, ref_x, dtype)
    assert res_out is x
