# Copyright 2026, The FlagOS Contributors.
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
from flag_gems.ops._pad_circular import _pad_circular

from . import accuracy_utils as utils

# (shape, pad) cases exercising circular padding of the last 1/2/3 dims.
# Constraints of aten._pad_circular are respected: at least one leading
# (non-padded) dim, and |pad| <= dim size (wrap at most once). Negative pad crops.
SHAPE_PAD_CASES = [
    # 1D pad (last dim)
    ((1, 8), [2, 2]),
    ((4, 16), [3, 5]),
    ((2, 3, 32), [8, 0]),
    ((8, 64, 20), [5, 5]),
    # negative pad (crop) on last dim
    ((1, 10), [-2, -3]),
    # 2D pad (last two dims)
    ((2, 3, 8, 8), [1, 1, 1, 1]),
    ((4, 16, 32), [4, 4, 2, 2]),
    ((8, 6, 20), [5, 5, 3, 3]),
    # 3D pad (last three dims)
    ((2, 4, 4, 4), [1, 1, 1, 1, 1, 1]),
    ((2, 3, 5, 6, 7), [2, 2, 1, 1, 3, 3]),
]


@pytest.mark.pad_circular
@pytest.mark.parametrize("shape, pad", SHAPE_PAD_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_pad_circular(shape, pad, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten._pad_circular(ref_inp, pad)

    res_out = _pad_circular(inp, pad)

    utils.gems_assert_equal(res_out, ref_out)
