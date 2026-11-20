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

device = flag_gems.device

# Shapes with a zero dimension: numel() == 0, which is not covered by
# POINTWISE_SHAPES (its "()" entry is a 0-dim tensor with one element).
EMPTY_SHAPES = [(0,), (3, 0), (2, 0, 4)]


@pytest.mark.ones_like
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ones_like(shape, dtype):
    inp = torch.empty(size=shape, dtype=dtype, device=device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ones_like(ref_inp)
    res_out = flag_gems.ones_like(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.ones_like
@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ones_like_empty(shape, dtype):
    # Regression test: with numel() == 0 the Ascend backend derived a block
    # size of 0 and the kernel failed to compile ("arange's end argument must
    # be greater than the start argument").
    inp = torch.empty(size=shape, dtype=dtype, device=device)

    ref_out = torch.ones_like(utils.to_reference(inp))
    res_out = flag_gems.ones_like(inp)

    utils.gems_assert_equal(res_out, ref_out)
