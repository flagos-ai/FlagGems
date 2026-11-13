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


@pytest.mark.clamp_max
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("max", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max(shape, max, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.clamp_max(ref_inp, max)
    res_out = flag_gems.clamp_max(inp, max)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.clamp_max_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("max", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max_(shape, max, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.clamp_max_(max)
    res_out = flag_gems.clamp_max_(inp, max)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.clamp_max
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("max", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES + utils.FP8_DTYPES)
def test_clamp_max_low_precision(shape, max, dtype):
    if dtype in utils.FP8_DTYPES:
        # Native CUDA clamp_max_scalar is not implemented for fp8;
        # FlagGems supports it, so the reference runs through float32.
        inp = (
            torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
            .clamp(-2, 2)
            .to(dtype)
        )
        ref_inp = utils.to_reference(inp.float())
        ref_out = torch.clamp_max(ref_inp, max).to(dtype)
        res_out = flag_gems.clamp_max(inp, max)
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        # An integral tensor with a float scalar promotes the result to
        # float32, matching torch's type promotion rules.
        inp = torch.randint(0, 8, shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp)
        ref_out = torch.clamp_max(ref_inp, max)
        res_out = flag_gems.clamp_max(inp, max)
        utils.gems_assert_close(res_out, ref_out, res_out.dtype)


@pytest.mark.clamp_max
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("max", [-100, -3, -1, 0, 100])
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES)
def test_clamp_max_uint8_int_scalar(shape, max, dtype):
    # torch converts an integral scalar to the tensor's dtype first: negative
    # scalars wrap modulo 256 for uint8 and are rejected outside [-255, 255].
    if dtype == torch.uint8:
        inp = torch.randint(0, 8, shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp)
        ref_out = torch.clamp_max(ref_inp, max)
        res_out = flag_gems.clamp_max(inp, max)
        utils.gems_assert_close(res_out, ref_out, torch.uint8)
    else:
        inp = torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp)
        ref_out = torch.clamp_max(ref_inp, max)
        res_out = flag_gems.clamp_max(inp, max)
        utils.gems_assert_close(res_out, ref_out, torch.int8)
