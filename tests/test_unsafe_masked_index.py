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


@pytest.mark.unsafe_masked_index
@pytest.mark.parametrize("shape", utils.UT_SHAPES_1D)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unsafe_masked_index(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) > 0.3
    indices = torch.randint(0, max(inp.numel(), 1), shape, device=flag_gems.device)
    fill = 0.0

    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)
    ref_indices = utils.to_reference(indices)

    op = torch._unsafe_masked_index
    ref_out = op(ref_inp, ref_mask, [ref_indices], fill)
    with flag_gems.use_gems():
        res_out = op(inp, mask, [indices], fill)

    utils.gems_assert_close(res_out, ref_out, dtype)
