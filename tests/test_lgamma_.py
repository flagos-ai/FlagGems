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
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.triton_lang_helper import _fallback_lgamma, _patch_missing_symbols

from . import accuracy_utils as utils


@triton.jit
def fallback_lgamma_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, _fallback_lgamma(x), mask=mask)


def test_missing_lgamma_uses_fallback():
    class EmptyLibdevice:
        pass

    libdevice = _patch_missing_symbols(EmptyLibdevice(), ("lgamma",))
    assert libdevice.lgamma is _fallback_lgamma


def test_fallback_lgamma():
    inp = torch.tensor(
        [
            -float("inf"),
            -4.0,
            -3.75,
            -2.5,
            -1.25,
            -0.5,
            -0.0,
            0.0,
            0.1,
            0.5,
            1.0,
            1.5,
            3.0,
            8.0,
            32.0,
            float("inf"),
            float("nan"),
        ],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    expected = torch.lgamma(inp)
    actual = torch.empty_like(inp)
    block_size = triton.next_power_of_2(inp.numel())
    with torch_device_fn.device(inp.device):
        fallback_lgamma_kernel[(1,)](inp, actual, inp.numel(), BLOCK_SIZE=block_size)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5, equal_nan=True)


@pytest.mark.lgamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lgamma(shape, dtype):
    torch.manual_seed(0)
    inp = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    )  # lgamma requires positive values
    ref_inp = utils.to_reference(inp)
    ref_out = ref_inp.lgamma()
    with flag_gems.use_gems():
        res_out = inp.lgamma()
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lgamma_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lgamma_(shape, dtype):
    torch.manual_seed(0)
    inp = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    )  # lgamma requires positive values
    ref_inp = utils.to_reference(inp.clone())
    ref_out = ref_inp.lgamma_()
    with flag_gems.use_gems():
        res_out = inp.lgamma_()
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp, ref_inp, dtype)
