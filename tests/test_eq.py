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


@pytest.mark.eq
@pytest.mark.eq_scalar
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("value", [2**24, -(2**24), 2**31 - 2])
def test_eq_integer_precision_original_matrix(dtype, strided, value):
    if dtype == torch.int64 and not utils.int64_is_supported:
        pytest.skip("INT64 is not supported on this backend")
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
@pytest.mark.skipif(
    not utils.int64_is_supported, reason="INT64 is not supported on this backend"
)
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
    if torch.int64 in (left_dtype, right_dtype) and not utils.int64_is_supported:
        pytest.skip("INT64 is not supported on this backend")
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
    if (dtype == torch.int64 or value > 2**31 - 1) and not utils.int64_is_supported:
        pytest.skip("INT64 is not supported on this backend")
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
    if dtype == torch.bfloat16 and not utils.bf16_is_supported:
        pytest.skip("BF16 is not supported on this backend")
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


@pytest.mark.eq
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "strong_dtype,strong_values,weak_dtype,weak_value",
    [
        (torch.int8, [-1, 0, 1], torch.uint8, 255),
        (torch.uint8, [0, 1, 255], torch.int8, -1),
        (torch.bool, [0, 1], torch.int64, 2),
        (torch.int8, [0, 1, -1], torch.bool, True),
    ],
)
def test_eq_zero_dim_integer_promotion(
    strong_dtype, strong_values, weak_dtype, weak_value, reverse
):
    if torch.int64 in (strong_dtype, weak_dtype) and not utils.int64_is_supported:
        pytest.skip("INT64 is not supported on this backend")
    strong = torch.tensor(strong_values, dtype=strong_dtype, device=flag_gems.device)
    weak = torch.tensor(weak_value, dtype=weak_dtype, device=flag_gems.device)
    left, right = (weak, strong) if reverse else (strong, weak)
    utils.gems_assert_equal(
        flag_gems.eq(left, right), torch.eq(left.cpu(), right.cpu())
    )


@pytest.mark.eq_scalar
def test_eq_scalar_bool_and_integer_launch_sequence():
    inp = torch.tensor([0, 1, -1], dtype=torch.int8, device=flag_gems.device)
    for value in (True, 2, False, 255):
        utils.gems_assert_equal(
            flag_gems.eq_scalar(inp, value), torch.eq(inp.cpu(), value)
        )


@pytest.mark.eq
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "left_dtype,left_values,right_dtype,right_values",
    [
        (torch.int8, [-1, 1], torch.uint8, [255, 1]),
        (torch.int32, [2**24 + 1, 2**24], torch.int64, [2**24 + 1, 2**24 + 1]),
    ],
)
def test_eq_mixed_integer_tensors(
    left_dtype, left_values, right_dtype, right_values, reverse
):
    if torch.int64 in (left_dtype, right_dtype) and not utils.int64_is_supported:
        pytest.skip("INT64 is not supported on this backend")
    left = torch.tensor(left_values, dtype=left_dtype, device=flag_gems.device)
    right = torch.tensor(right_values, dtype=right_dtype, device=flag_gems.device)
    if reverse:
        left, right = right, left
    utils.gems_assert_equal(
        flag_gems.eq(left, right), torch.eq(left.cpu(), right.cpu())
    )


@pytest.mark.eq
@pytest.mark.eq_scalar
@pytest.mark.skipif(
    not utils.fp64_is_supported, reason="FP64 is not supported on this backend"
)
def test_eq_float64_precision_is_preserved():
    inp = torch.tensor(
        [1.0, 1.0 + 2**-40], dtype=torch.float64, device=flag_gems.device
    )
    other = torch.ones_like(inp)
    utils.gems_assert_equal(flag_gems.eq(inp, other), torch.eq(inp.cpu(), other.cpu()))
    utils.gems_assert_equal(flag_gems.eq_scalar(inp, 1.0), torch.eq(inp.cpu(), 1.0))
