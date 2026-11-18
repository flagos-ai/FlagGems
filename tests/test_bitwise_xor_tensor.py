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

# Full dtype set for bitwise ops (integer + bool only; ATen raises for
# float/complex inputs, so no fp32/bf16/fp16/fp8 cases exist).
XOR_DTYPES = utils.SMALL_INT_DTYPES + utils.INT_DTYPES + utils.ALL_INT_DTYPES

# Tensor value pairs (self, other) in six fixed slots per dtype, so every
# dtype exercises its own min/max instead of inheriting int64-sized constants
# that would be out of range and get skipped for int8/uint8/int16.
# slot 0: (min, min)  slot 1: (max, max)  slot 2: (min, 0)
# slot 3: (0, min)    slot 4: (max, 0)    slot 5: (min, max)
XOR_EXTREME_SLOTS = [0, 1, 2, 3, 4, 5]


def _extreme_pair(dtype, slot):
    """Return the value pair for one boundary slot of `dtype`."""
    info = torch.iinfo(dtype)
    return [
        (info.min, info.min),
        (info.max, info.max),
        (info.min, 0),
        (0, info.min),
        (info.max, 0),
        (info.min, info.max),
    ][slot]


@pytest.mark.bitwise_xor
@pytest.mark.bitwise_xor_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.SMALL_INT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
def test_bitwise_xor_tensor(shape, dtype):
    if dtype in utils.BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        inp1 = utils.randint_in_range(shape, dtype).to(flag_gems.device)
        inp2 = utils.randint_in_range(shape, dtype).to(flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.bitwise_xor(ref_inp1, ref_inp2)
    res_out = flag_gems.bitwise_xor_tensor(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_xor
@pytest.mark.bitwise_xor_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", XOR_DTYPES)
@pytest.mark.parametrize("slot", XOR_EXTREME_SLOTS)
def test_bitwise_xor_tensor_extremes(shape, dtype, slot):
    """Tensor-Tensor XOR at this dtype's own min/max boundaries.

    Each dtype is driven by its own iinfo bounds, so int8 gets (-128, 127)
    rather than inheriting int64 constants it cannot represent.
    """
    a_val, b_val = _extreme_pair(dtype, slot)
    inp1 = torch.full(shape, a_val, dtype=dtype, device="cpu").to(flag_gems.device)
    inp2 = torch.full(shape, b_val, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.bitwise_xor(ref_inp1, ref_inp2)
    res_out = flag_gems.bitwise_xor_tensor(inp1, inp2)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_xor_
@pytest.mark.bitwise_xor_tensor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.SMALL_INT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
def test_bitwise_xor_tensor_(shape, dtype):
    if dtype in utils.BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
    else:
        inp1 = utils.randint_in_range(shape, dtype).to(flag_gems.device)
        inp2 = utils.randint_in_range(shape, dtype).to(flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone())
    ref_inp2 = utils.to_reference(inp2)

    ref_out = ref_inp1.bitwise_xor_(ref_inp2)
    res_out = flag_gems.bitwise_xor_tensor_(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_xor_
@pytest.mark.bitwise_xor_tensor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", XOR_DTYPES)
@pytest.mark.parametrize("slot", XOR_EXTREME_SLOTS)
def test_bitwise_xor_tensor__extremes(shape, dtype, slot):
    """Inplace tensor XOR at this dtype's own min/max boundaries."""
    a_val, b_val = _extreme_pair(dtype, slot)
    inp1 = torch.full(shape, a_val, dtype=dtype, device=flag_gems.device)
    inp2 = torch.full(shape, b_val, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone())
    ref_inp2 = utils.to_reference(inp2)

    ref_out = ref_inp1.bitwise_xor_(ref_inp2)
    res_out = flag_gems.bitwise_xor_tensor_(inp1, inp2)
    utils.gems_assert_equal(res_out, ref_out)
