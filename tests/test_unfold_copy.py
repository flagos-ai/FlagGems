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


@pytest.mark.unfold_copy
@pytest.mark.parametrize(
    "shape, dim, size, step",
    [
        # 2D case (B, D) -> (B, L, size)
        ((4, 8), 1, 3, 1),
        ((16, 32), 1, 8, 2),
        ((8, 15), 1, 4, 3),
        # 3D case with dim=1: (B, D, C) -> (B, L, C, size)
        ((2, 6, 8), 1, 3, 1),
        ((4, 8, 16), 1, 4, 2),
        # 3D case with dim=2: (B, D, C) -> (B, D, L, size)
        ((2, 6, 8), 2, 3, 1),
        ((4, 8, 16), 2, 4, 2),
        ((2, 6, 8), 2, 3, 2),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unfold_copy(shape, dim, size, step, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.unfold_copy(ref_inp, dimension=dim, size=size, step=step)

    with flag_gems.use_gems():
        res_out = torch.unfold_copy(inp, dimension=dim, size=size, step=step)

    utils.gems_assert_close(res_out, ref_out, dtype)
