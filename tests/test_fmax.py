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


@pytest.mark.fmax
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fmax(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.fmax(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.fmax(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.fmax
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fmax_with_nan(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Inject NaN values into both inputs at different positions
    nan_mask1 = torch.rand(shape, device=flag_gems.device) < 0.2
    nan_mask2 = torch.rand(shape, device=flag_gems.device) < 0.2
    inp1[nan_mask1] = float("nan")
    inp2[nan_mask2] = float("nan")
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.fmax(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.fmax(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.fmax_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fmax_out(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.empty_like(ref_inp1)
    torch.fmax(ref_inp1, ref_inp2, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp1)
        torch.fmax(inp1, inp2, out=res_out)

    utils.gems_assert_equal(res_out, ref_out)
