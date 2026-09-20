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
from . import conftest as cfg

if cfg.QUICK_MODE:
    ROUND_DECIMALS_SHAPES = [(2, 3)]
    ROUND_HALF_SHAPES = [(2, 3)]
    ROUND_DECIMALS = [0]
else:
    ROUND_DECIMALS_SHAPES = [(2, 3), (128, 256), (4, 8, 16)]
    ROUND_HALF_SHAPES = [(2, 3), (4, 8)]
    ROUND_DECIMALS = [-2, -1, 0, 1, 2]


@pytest.mark.round
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_round(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.round(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.round(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.round_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_round_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.round_(ref_inp)
    with flag_gems.use_gems():
        res_out = inp.round_()

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.round_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_round_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)

    torch.round(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        torch.round(inp, out=out)

    utils.gems_assert_equal(out, ref_out)


@pytest.mark.round
@pytest.mark.parametrize("shape", ROUND_DECIMALS_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("decimals", ROUND_DECIMALS)
def test_round_decimals(shape, dtype, decimals):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 100

    # When demical≠0 and input is float16/bfloat16,
    # compute result is difference between CUDA and CPU in Pytorch itself because of precision error
    # so compare the result between FlagGems version and Pytorch CUDA version
    ref_out = torch.round(inp, decimals=decimals)
    ref_out = ref_out.to("cpu")

    with flag_gems.use_gems():
        res_out = torch.round(inp, decimals=decimals)
        res_out = res_out.to("cpu")

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.round
@pytest.mark.parametrize("shape", ROUND_HALF_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_round_half_to_even(shape, dtype):
    # Test round half to even: 2.5->2, 3.5->4, -2.5->-2, -3.5->-4
    inp = torch.tensor(
        [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.round(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.round(inp)

    utils.gems_assert_equal(res_out, ref_out)
