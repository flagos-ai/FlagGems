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

import itertools

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    M_VALUES = [1, 64]
    TOP_KS = [2]
    K_VALUES = [128]
else:
    M_VALUES = [1, 33, 64, 222]
    TOP_KS = [2, 6]
    K_VALUES = [128, 511, 1024]
MOE_SHAPES = list(itertools.product(M_VALUES, TOP_KS, K_VALUES))


@pytest.mark.moe_sum
@pytest.mark.parametrize("shape", MOE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_moe_sum(shape, dtype):
    m, topk, k = shape
    inp1 = torch.randn((m, topk, k), dtype=dtype, device=flag_gems.device)
    res_out = torch.empty((m, k), dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_out = torch.sum(ref_inp1, dim=1)

    with flag_gems.use_gems():
        flag_gems.moe_sum(inp1, res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)
