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
from . import conftest as cfg

# rot90 test shapes - only 2D for now since the kernel only handles 2D case
if cfg.QUICK_MODE:
    ROT90_SHAPES_2D = [(2, 3), (100, 128)]
    ROT90_K_VALUES = [0, 1, -1]
else:
    ROT90_SHAPES_2D = [(2, 3), (5, 7), (100, 128)]
    ROT90_K_VALUES = [0, 1, 2, 3, -1, -2]


@pytest.mark.rot90
@pytest.mark.parametrize("shape", ROT90_SHAPES_2D)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("k", ROT90_K_VALUES)
def test_rot90(shape, dtype, k):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.rot90(ref_inp, k, [0, 1])
    with flag_gems.use_gems():
        res_out = torch.rot90(inp, k, [0, 1])

    utils.gems_assert_equal(res_out, ref_out)
