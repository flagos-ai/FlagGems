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


@pytest.mark.celu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_celu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    alpha = torch.rand(1).item()

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.nn.functional.celu(ref_inp, alpha)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.celu(inp, alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.celu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_celu_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    alpha = torch.rand(1).item()

    res_inp = inp.clone().to(flag_gems.device)
    inp_clone = inp.clone()
    ref_inp = utils.to_reference(inp_clone, True)
    torch.nn.functional.celu_(ref_inp, alpha)

    with flag_gems.use_gems():
        torch.nn.functional.celu_(res_inp, alpha)

    utils.gems_assert_close(res_inp, ref_inp, dtype)
