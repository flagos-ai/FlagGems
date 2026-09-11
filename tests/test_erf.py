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


@pytest.mark.special_erf
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_erf(shape, dtype, caplog):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.ops.aten.special_erf(ref_x.float()).to(dtype)
    else:
        ref_out = torch.ops.aten.special_erf(ref_x)
    with caplog.at_level("DEBUG", logger="flag_gems.ops.special_erf"):
        with flag_gems.use_gems():
            act_out = torch.ops.aten.special_erf(x)
    assert "GEMS SPECIAL_ERF" in caplog.text
    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.erf
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.erf(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.erf(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.erf_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erf_(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.erf_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.erf_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
