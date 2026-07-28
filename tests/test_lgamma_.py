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


@pytest.mark.lgamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lgamma(shape, dtype):
    torch.manual_seed(0)
    inp = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    )  # lgamma requires positive values
    ref_inp = utils.to_reference(inp)
    ref_out = ref_inp.lgamma()
    with flag_gems.use_gems():
        res_out = inp.lgamma()
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lgamma_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lgamma_(shape, dtype):
    torch.manual_seed(0)
    inp = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    )  # lgamma requires positive values
    ref_inp = utils.to_reference(inp.clone())
    ref_out = ref_inp.lgamma_()
    with flag_gems.use_gems():
        res_out = inp.lgamma_()
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp, ref_inp, dtype)
