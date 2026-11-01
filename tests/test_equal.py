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


@pytest.mark.equal
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_equal(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    inp2 = inp1.clone()

    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.equal(ref_inp1, ref_inp2)
    res_out = flag_gems.equal(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)

    ref_out = torch.equal(ref_inp1 + 1, ref_inp2)
    res_out = flag_gems.equal(inp1 + 1, inp2)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.equal
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_equal_integer_precision(dtype):
    # Issue #7584: equal() inherits eq(), so distinct integers beyond the
    # fp32 mantissa (2**24) must still compare unequal. A pair whose every
    # element collides after an fp32 cast (2**24 vs 2**24 + 1) is the case
    # where the bug flips the final boolean answer.
    inp1 = torch.arange(2**24, 2**24 + 2, device=flag_gems.device, dtype=dtype).unsqueeze(0)
    inp2 = inp1.clone()
    inp3 = inp1 + 1

    assert flag_gems.equal(inp1, inp2) is True
    assert flag_gems.equal(inp1, inp3) is False
