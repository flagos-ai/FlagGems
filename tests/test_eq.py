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
@pytest.mark.eq_scalar
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("value", [2**24, -(2**24), 2**31 - 2])
def test_eq_integer_precision(dtype, strided, value):
    inp = torch.tensor([value, value + 1], dtype=dtype, device=flag_gems.device)
    other = torch.full_like(inp, value + 1)
    if strided:
        storage = torch.empty(5, dtype=dtype, device=flag_gems.device)
        storage[1::2].copy_(inp)
        inp = storage[1::2]
    ref_inp, ref_other = inp.cpu(), other.cpu()
    utils.gems_assert_equal(flag_gems.eq(inp, other), torch.eq(ref_inp, ref_other))
    utils.gems_assert_equal(
        flag_gems.eq_scalar(inp, value + 1), torch.eq(ref_inp, value + 1)
    )


@pytest.mark.eq
@pytest.mark.eq_scalar
@pytest.mark.parametrize("value", [2**53, 2**63 - 2, -(2**63)])
def test_eq_int64_limits(value):
    inp = torch.tensor([value, value + 1], dtype=torch.int64, device=flag_gems.device)
    ref_inp = inp.cpu()
    other = torch.full_like(inp, value + 1)
    utils.gems_assert_equal(flag_gems.eq(inp, other), torch.eq(ref_inp, other.cpu()))
    utils.gems_assert_equal(
        flag_gems.eq_scalar(inp, value + 1), torch.eq(ref_inp, value + 1)
    )


@pytest.mark.eq
@pytest.mark.parametrize(
    "left_dtype,right_dtype",
    [
        (torch.int32, torch.int64),
        (torch.int64, torch.int32),
        (torch.int8, torch.uint8),
        (torch.uint8, torch.int8),
        (torch.bool, torch.int32),
        (torch.int32, torch.bool),
        (torch.int32, torch.float32),
        (torch.float32, torch.int32),
    ],
)
@pytest.mark.parametrize("scalar_tensor", [False, True])
def test_eq_mixed_dtype(left_dtype, right_dtype, scalar_tensor):
    left = torch.tensor([0, 1, -1], dtype=left_dtype, device=flag_gems.device)
    right = torch.tensor(
        -1 if scalar_tensor else [-1, 1, 0], dtype=right_dtype, device=flag_gems.device
    )
    expected = torch.eq(left.cpu(), right.cpu())
    utils.gems_assert_equal(flag_gems.eq(left, right), expected)


@pytest.mark.eq_scalar
@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.uint8, torch.int8, torch.int32, torch.int64]
)
@pytest.mark.parametrize("value", [True, 2, 255, 2**32 - 1, 2**63 - 1])
def test_eq_scalar_integer_promotion(dtype, value):
    inp = torch.tensor([0, 1, -1], dtype=dtype, device=flag_gems.device)
    utils.gems_assert_equal(flag_gems.eq_scalar(inp, value), torch.eq(inp.cpu(), value))


@pytest.mark.eq
@pytest.mark.eq_scalar
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
def test_eq_controls(dtype):
    inp = torch.tensor([0, 1, 0], dtype=dtype, device=flag_gems.device)
    other = torch.tensor([1, 1, 0], dtype=dtype, device=flag_gems.device)
    utils.gems_assert_equal(flag_gems.eq(inp, other), torch.eq(inp.cpu(), other.cpu()))
    utils.gems_assert_equal(flag_gems.eq_scalar(inp, 1), torch.eq(inp.cpu(), 1))


@pytest.mark.eq
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16])
def test_eq_small_integer_limits(dtype):
    limits = torch.iinfo(dtype)
    inp = torch.tensor(
        [limits.min, limits.max, 0, 1], dtype=dtype, device=flag_gems.device
    )
    other = inp.roll(1)
    utils.gems_assert_equal(flag_gems.eq(inp, other), torch.eq(inp.cpu(), other.cpu()))
    utils.gems_assert_equal(flag_gems.eq(inp, inp), torch.eq(inp.cpu(), inp.cpu()))
