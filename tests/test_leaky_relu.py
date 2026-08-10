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


@pytest.mark.leaky_relu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    negative_slope = 0.01
    ref_out = torch.nn.functional.leaky_relu(ref_inp, negative_slope=negative_slope)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu(res_inp, negative_slope=negative_slope)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.leaky_relu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp.clone(), True)

    negative_slope = 0.01
    ref_out = torch.nn.functional.leaky_relu_(ref_inp, negative_slope=negative_slope)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu_(
            res_inp, negative_slope=negative_slope
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.leaky_relu_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    negative_slope = 0.01

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.empty(shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten.leaky_relu.out(ref_inp, negative_slope=negative_slope, out=ref_out)

    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.leaky_relu.out(inp, negative_slope=negative_slope, out=out)

    utils.gems_assert_close(out, ref_out, dtype)
