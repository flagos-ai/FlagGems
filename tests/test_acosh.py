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


@pytest.mark.acosh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh(shape, dtype):
    # acosh domain is [1, inf), so generate input in [1, 2]
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(1.0, 2.0)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.acosh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.acosh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.acosh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_acosh_(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(1.0, 2.0)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.acosh_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.acosh_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
