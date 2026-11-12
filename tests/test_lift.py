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


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


@pytest.mark.lift
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lift(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.lift(ref_inp)
    res_out = flag_gems.lift(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lift_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lift_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)
    torch.ops.aten.lift.out(ref_inp, out=ref_out)
    res_out = torch.empty_like(inp)
    flag_gems.lift_out(inp, out=res_out)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lift
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES + utils.FP8_DTYPES)
def test_lift_low_precision(shape, dtype):
    # lift is a metadata-only identity: dtype, shape and values must survive
    # bit exactly, including float8 inputs.
    if dtype in utils.FP8_DTYPES:
        raw = (torch.arange(1, _numel(shape) + 1, device=flag_gems.device) % 256).to(
            torch.uint8
        )
        inp = raw.reshape(shape).view(dtype)
    else:
        inp = torch.randint(0, 8, shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.lift(ref_inp)
    res_out = flag_gems.lift(inp.clone())
    if ref_out.device != res_out.device:
        res_out = res_out.cpu()
    assert res_out.dtype == ref_out.dtype
    # Bitwise compare: NaNs in fp8 inputs break float32 closeness checks.
    assert torch.equal(res_out.view(torch.uint8), ref_out.view(torch.uint8))


@pytest.mark.lift_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES + utils.FP8_DTYPES)
def test_lift_out_low_precision(shape, dtype):
    if dtype in utils.FP8_DTYPES:
        raw = (torch.arange(1, _numel(shape) + 1, device=flag_gems.device) % 256).to(
            torch.uint8
        )
        inp = raw.reshape(shape).view(dtype)
    else:
        inp = torch.randint(0, 8, shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.lift.out(ref_inp, out=torch.empty_like(ref_inp))
    res_out = flag_gems.lift_out(inp.clone(), out=torch.empty_like(inp))
    if ref_out.device != res_out.device:
        res_out = res_out.cpu()
    assert res_out.dtype == ref_out.dtype
    assert torch.equal(res_out.view(torch.uint8), ref_out.view(torch.uint8))
