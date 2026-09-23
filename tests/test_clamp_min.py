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


@pytest.mark.clamp_min
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_min(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mini = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_mini = utils.to_reference(mini)

    ref_out = torch.clamp_min(ref_inp, min=ref_mini)
    with flag_gems.use_gems():
        res_out = torch.clamp_min(inp, min=mini)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.clamp_min_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_min_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mini = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    ref_mini = utils.to_reference(mini)

    ref_out = torch.clamp_min_(ref_inp, min=ref_mini)
    with flag_gems.use_gems():
        res_out = torch.clamp_min_(inp, min=mini)

    utils.gems_assert_equal(res_out, ref_out)
