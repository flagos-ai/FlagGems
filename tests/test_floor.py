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


@pytest.mark.floor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_floor(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.floor(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.floor(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.floor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_floor_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.floor_()
    with flag_gems.use_gems():
        res_out = inp.floor_()

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.floor_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_floor_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)

    torch.floor(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        torch.floor(inp, out=out)

    utils.gems_assert_equal(out, ref_out)
