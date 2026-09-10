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


@pytest.mark.fmod_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# fmod only supports float32 due to integer division semantics
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fmod_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.where(inp2 == 0, torch.ones_like(inp2), inp2)
    ref_inp1 = utils.to_reference(inp.clone())
    ref_inp2 = utils.to_reference(inp2)

    ref_out = ref_inp1.fmod_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp.fmod_(inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=2.0)
    utils.gems_assert_close(inp, ref_out, dtype, atol=2.0)

    ref_inp1 = utils.to_reference(inp.clone(), False)
    for d in inp2.flatten()[:2]:
        ref_d = utils.to_reference(d, False)
        ref_out = ref_inp1.fmod_(ref_d)
        with flag_gems.use_gems():
            res_out = inp.clone().fmod_(d)
        utils.gems_assert_close(res_out, ref_out, dtype, atol=2.0)
