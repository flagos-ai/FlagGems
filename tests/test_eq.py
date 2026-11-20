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


@pytest.mark.eq
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_eq(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.eq(ref_inp1, ref_inp2)
    res_out = flag_gems.eq(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.eq_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_eq_scalar(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    inp2 = 0
    ref_inp1 = utils.to_reference(inp1)

    ref_out = torch.eq(ref_inp1, inp2)
    res_out = flag_gems.eq_scalar(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.eq
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_eq_integer_precision(dtype, strided):
    # FP32 rounds 2**24 and 2**24 + 1 to the same value, so comparing in FP32
    # would report distinct integer ids as equal.
    inp1_cpu = torch.tensor([2**24, 2**24 + 1, 2**24 + 2], dtype=dtype)
    inp2_cpu = torch.full_like(inp1_cpu, 2**24 + 1)
    inp1 = inp1_cpu.to(flag_gems.device)
    inp2 = inp2_cpu.to(flag_gems.device)
    if strided:
        storage = torch.empty(2 * inp1.numel(), dtype=dtype, device=flag_gems.device)
        storage[::2].copy_(inp1)
        inp1 = storage[::2]

    ref_out = torch.eq(inp1_cpu, inp2_cpu)
    res_out = flag_gems.eq(inp1, inp2).cpu()

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.eq_scalar
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_eq_scalar_integer_precision(dtype):
    inp1_cpu = torch.tensor([2**24, 2**24 + 1, 2**24 + 2], dtype=dtype)
    inp1 = inp1_cpu.to(flag_gems.device)

    ref_out = torch.eq(inp1_cpu, 2**24 + 1)
    res_out = flag_gems.eq_scalar(inp1, 2**24 + 1).cpu()

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.eq_scalar
def test_eq_scalar_float_scalar():
    # A floating-point scalar must not be truncated to an integer.
    inp1_cpu = torch.tensor([1, 2, 3], dtype=torch.int64)
    inp1 = inp1_cpu.to(flag_gems.device)

    ref_out = torch.eq(inp1_cpu, 2.5)
    res_out = flag_gems.eq_scalar(inp1, 2.5).cpu()

    utils.gems_assert_equal(res_out, ref_out)
