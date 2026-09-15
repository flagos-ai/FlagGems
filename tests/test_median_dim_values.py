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


@pytest.mark.median_dim_values
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_median_dim_values(dim, keepdim, dtype):
    shape = (3, 17)
    inp = torch.arange(
        torch.Size(shape).numel(), device=flag_gems.device, dtype=torch.int64
    )
    inp = inp.reshape(shape).to(dtype)
    ref = torch.median(utils.to_reference(inp), dim=dim, keepdim=keepdim)
    values = torch.empty(ref.values.shape, dtype=dtype, device=flag_gems.device)
    indices = torch.empty(ref.indices.shape, dtype=torch.int64, device=flag_gems.device)

    result = flag_gems.median_dim_values(
        inp, dim=dim, keepdim=keepdim, values=values, indices=indices
    )

    assert result.values is values
    assert result.indices is indices
    utils.gems_assert_equal(result.values, ref.values)
    utils.gems_assert_equal(result.indices, ref.indices)
