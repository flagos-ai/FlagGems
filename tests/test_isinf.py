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


@pytest.mark.isinf
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_isinf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.masked_fill(inp, inp > 1.0, -float("inf"))
    ref_inp = utils.to_reference(inp)

    ref_out = torch.isinf(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.isinf(inp)

    utils.gems_assert_equal(res_out, ref_out)
