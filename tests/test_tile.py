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

TILE_DIMS = [(0,), (2,), (2, 0), (0, 2), (2, 2), (2, 2, 2), (2, 2, 2, 2)]


@pytest.mark.tile
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dims", TILE_DIMS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_tile(shape, dims, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.tile(ref_inp, dims)
    res_out = flag_gems.tile(inp, dims)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.tile
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tile_noncontiguous(dtype):
    inp = torch.randn((17, 33), dtype=dtype, device=flag_gems.device).T
    ref = torch.tile(utils.to_reference(inp), (2, 3))
    result = flag_gems.tile(inp, (2, 3))
    utils.gems_assert_equal(result, ref)
