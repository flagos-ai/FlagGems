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


@pytest.mark.median_out
@pytest.mark.parametrize("shape", [(17,), (3, 17)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_median_out(shape, dtype):
    inp = torch.arange(
        torch.Size(shape).numel(), device=flag_gems.device, dtype=torch.int64
    )
    inp = inp.reshape(shape).to(dtype)
    ref = torch.median(utils.to_reference(inp))
    out = torch.empty((), dtype=dtype, device=flag_gems.device)

    result = flag_gems.median_out(inp, out=out)

    assert result is out
    utils.gems_assert_equal(result, ref)
