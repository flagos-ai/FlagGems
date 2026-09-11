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


@pytest.mark.arccosh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arccosh(shape, dtype):
    # arccosh domain is [1, inf); shift inputs into the valid range.
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    ref_inp = utils.to_reference(inp)
    ref_out = torch.arccosh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.arccosh(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.arccosh
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arccosh_special_values(dtype):
    # Values below 1 must produce NaN, +inf maps to +inf, matching PyTorch.
    inp = torch.tensor(
        [1.0, 2.0, 0.5, 0.0, -1.0, float("inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)
    ref_out = torch.arccosh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.arccosh(inp)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.arccosh_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arccosh_out(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    ref_inp = utils.to_reference(inp)

    ref_out = torch.empty_like(ref_inp)
    torch.arccosh(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.arccosh(inp, out=res_out)
    utils.gems_assert_close(res_out, ref_out, dtype)
