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


@pytest.mark.lshift
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_lshift(shape, dtype):
    inp1 = torch.randint(low=0, high=0x00FF, size=shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    inp2 = torch.randint(low=0, high=8, size=shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    # Reference: using bitwise_left_shift which is equivalent to __lshift__
    ref_out = torch.bitwise_left_shift(ref_inp1, ref_inp2)
    # FlagGems: using << operator which dispatches to aten.__lshift__
    with flag_gems.use_gems():
        res_out = inp1 << inp2

    utils.gems_assert_equal(res_out, ref_out)
