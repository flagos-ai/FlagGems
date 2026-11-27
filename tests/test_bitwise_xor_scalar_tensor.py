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

import random

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Full dtype set for bitwise ops (integer + bool only; ATen raises for
# float/complex inputs, so no fp32/bf16/fp16/fp8 cases exist).
XOR_DTYPES = utils.SMALL_INT_DTYPES + utils.INT_DTYPES + utils.ALL_INT_DTYPES

# Scalar values: 0, ±1, ±255, ±2**15, ±2**31-1, ±2**63-1 plus several large
# primes and negative counterparts — 12 values total.
XOR_SCALARS = [
    0,
    1,
    -1,
    0x00FF,
    -0x00FF,
    0x7FFF,
    -0x8000,
    0x7FFFFFFF,
    -0x80000000,
    0x7FFFFFFFFFFFFFFF,
    -0x8000000000000000,
    123456789,
]

# Tensor value pairs (self, other) at dtype boundaries.
XOR_VALUE_PAIRS = [
    (0, 0),
    (1, 1),
    (-1, -1),
    (0x00FF, 0x00FF),
    (0x7FFF, 0x7FFF),
    (0x7FFFFFFF, 0x7FFFFFFF),
    (0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF),
    (-0x8000000000000000, -0x8000000000000000),
    (0, 0x00FF),
    (0, -1),
    (0x00FF, 0x7FFFFFFF),
    (-1, 0x7FFF),
]


@pytest.mark.bitwise_xor
@pytest.mark.bitwise_xor_scalar_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.SMALL_INT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES
)
def test_bitwise_xor_scalar_tensor(shape, dtype):
    if dtype in utils.BOOL_TYPES:
        inp1 = bool(random.randint(0, 1))
        inp2 = torch.randint(0, 2, size=shape, dtype=torch.bool, device="cpu").to(
            flag_gems.device
        )
    else:
        inp1 = 0x00FF
        inp2 = utils.randint_in_range(shape, dtype).to(flag_gems.device)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.bitwise_xor(inp1, ref_inp2)
    res_out = flag_gems.bitwise_xor_scalar_tensor(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_xor
@pytest.mark.bitwise_xor_scalar_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", XOR_DTYPES)
@pytest.mark.parametrize("inp1", XOR_SCALARS)
def test_bitwise_xor_scalar_tensor_extremes(shape, dtype, inp1):
    """Scalar_Tensor XOR with a variety of scalar values.

    Scalar_Tensor forwards its Scalar `self` unchanged, so extreme scalar
    values must round-trip through the kernel unchanged.
    """
    inp2 = utils.randint_in_range(shape, dtype).to(flag_gems.device)
    ref_inp2 = utils.to_reference(inp2)
    ref_out = torch.bitwise_xor(inp1, ref_inp2)
    res_out = flag_gems.bitwise_xor_scalar_tensor(inp1, inp2)
    utils.gems_assert_equal(res_out, ref_out)


# Tensor-side boundary slots: the random inputs above never reach this dtype's
# own min/max (randint is upper-exclusive), so pin them explicitly.
XOR_TENSOR_SLOTS = [0, 1, 2]


def _tensor_boundary(dtype, slot):
    """Return min, max or zero for `slot`, in this dtype's own range."""
    info = torch.iinfo(dtype)
    return [info.min, info.max, 0][slot]


@pytest.mark.bitwise_xor
@pytest.mark.bitwise_xor_scalar_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", XOR_DTYPES)
@pytest.mark.parametrize("slot", XOR_TENSOR_SLOTS)
@pytest.mark.parametrize("inp1", XOR_SCALARS)
def test_bitwise_xor_scalar_tensor_bounds(shape, dtype, slot, inp1):
    """Scalar XOR with a dtype-boundary-filled tensor.

    Combines extreme scalars with this dtype's own min/max tensor values.
    """
    val = _tensor_boundary(dtype, slot)
    inp2 = torch.full(shape, val, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp2 = utils.to_reference(inp2)
    ref_out = torch.bitwise_xor(inp1, ref_inp2)
    res_out = flag_gems.bitwise_xor_scalar_tensor(inp1, inp2)
    utils.gems_assert_equal(res_out, ref_out)
