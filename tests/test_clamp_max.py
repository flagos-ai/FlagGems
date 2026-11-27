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
        if flag_gems.vendor_name == "enflame":
            # torch >= 2.10 rejects any non-zero rtol/atol on 1-byte floats
            # ("not supported for bitwise comparison of low dimensional
            # floats"), so the comparison would raise before checking the
            # values.  An fp8 value is exactly representable in fp32, so
            # promote both sides and compare there; other vendors keep their
            # existing comparison.
            utils.gems_assert_close(res_out.float(), ref_out.float(), torch.float32)
        else:
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


@pytest.mark.clamp_max
def test_clamp_max_int8_overflow_scalar():
    # Values outside the dtype range are rejected before dispatch, matching
    # native "value cannot be converted to type int8_t without overflow".
    inp = torch.tensor([1, 2, 3], dtype=torch.int8, device=flag_gems.device)
    if flag_gems.vendor_name == "enflame":
        # torch_gcu's native clamp_max does not validate an integral scalar;
        # only the FlagGems entry point raises the ATen error here.
        with pytest.raises(RuntimeError):
            flag_gems.clamp_max(inp, 200)
        return
    with pytest.raises(RuntimeError):
        torch.clamp_max(inp, 200)
    with pytest.raises(RuntimeError):
        flag_gems.clamp_max(inp, 200)


@pytest.mark.clamp_max
def test_clamp_max_uint8_overflow_scalar():
    # Negative scalars outside [-255, 255] are rejected for uint8 as well.
    inp = torch.tensor([1, 2, 3], dtype=torch.uint8, device=flag_gems.device)
    if flag_gems.vendor_name == "enflame":
        # Same native validation gap as the int8 case above.
        with pytest.raises(RuntimeError):
            flag_gems.clamp_max(inp, -256)
        return
    with pytest.raises(RuntimeError):
        torch.clamp_max(inp, -256)
    with pytest.raises(RuntimeError):
        flag_gems.clamp_max(inp, -256)


@pytest.mark.clamp_max
def test_clamp_max_uint8_boundary_scalar():
    # The accepted range endpoints for uint8 integral scalars.
    inp = torch.tensor([1, 2, 3], dtype=torch.uint8, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.clamp_max(ref_inp, 255)
    res_out = flag_gems.clamp_max(inp, 255)
    utils.gems_assert_close(res_out, ref_out, torch.uint8)
