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


@pytest.mark.special_airy_ai
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_airy_ai(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Use float32 for reference since PyTorch doesn't support airy_ai on float16
    ref_out = torch.special.airy_ai(ref_inp.float()).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.special.airy_ai(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.special_airy_ai_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_airy_ai_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Use float32 for reference since PyTorch doesn't support airy_ai on float16
    out_ref = torch.empty_like(ref_inp, dtype=torch.float32)
    ref_out = torch.special.airy_ai(ref_inp.float(), out=out_ref)
    ref_out = out_ref.to(dtype)

    out_act = torch.empty_like(inp)
    with flag_gems.use_gems():
        act_out = torch.special.airy_ai(inp, out=out_act)

    utils.gems_assert_close(act_out, ref_out, dtype, atol=1e-3)
    utils.gems_assert_close(out_act, ref_out, dtype, atol=1e-3)
