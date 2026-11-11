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


@pytest.mark.arccosh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arccosh_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # arccosh requires input >= 1
    inp = inp.abs() + 1.0
    ref_inp = utils.to_reference(inp, True)
    act_inp = inp.clone()

    ref_inp.arccosh_()
    flag_gems.arccosh_(act_inp)

    utils.gems_assert_close(act_inp, ref_inp, dtype)
