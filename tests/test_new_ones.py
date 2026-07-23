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


@pytest.mark.new_ones
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_new_ones(shape, dtype):
    inp = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = ref_inp.new_ones(shape)
    with flag_gems.use_gems():
        res_out = inp.new_ones(shape)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.new_ones
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_new_ones_different_size(shape, dtype):
    inp = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    size = (2, 3)
    ref_out = ref_inp.new_ones(size)
    with flag_gems.use_gems():
        res_out = inp.new_ones(size)
    utils.gems_assert_equal(res_out, ref_out)
