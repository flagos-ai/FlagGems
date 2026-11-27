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

# Inplace bitwise shift requires broadcast-compatible shapes where
# first dims are mapped according to broadcasting rules. Values are
# shifted in place, so self's dtype drives the output dtype.
INPLACE_BITWISE_SHAPES = [
    ((512, 1024), (512, 1024)),
    ((256, 512), (1, 512)),
    ((256, 512), (256, 1)),
    ((1024,), ()),
    ((256, 512, 128), (256, 512, 128)),
    ((16, 128, 64, 60), (16, 128, 64, 60)),
]

# Full coverage of the model-relevant shapes requested for this op:
# scalar (), 1D (1,), (256,), 2D (1024, 1024), 3D (20, 320, 15),
# 4D (16, 128, 64, 60), 5D (16, 7, 57, 32, 29) — 0 to 5 dims.
SHIFT_SHAPES = [
    (),
    (1,),
    (256,),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

# Shift counts: only the values that carry distinct semantics per dtype
# bit width — 0 (identity), bit_width-1 (sign boundary), bit_width (full
# shift to zero / sign fill), and adjacent out-of-range probes.
SHIFT_COUNTS = [0, 1, 2, 7, 8, 9, 15, 16, 31, 32, 63, 64, 65, 127, 128, 255]

# dtype set: bitwise ops are integer-only; float/complex raise at ATen level.
SHIFT_DTYPES = utils.SMALL_INT_DTYPES + utils.ALL_INT_DTYPES


@pytest.mark.bitwise_left_shift_
@pytest.mark.parametrize("shapes", INPLACE_BITWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.SMALL_INT_DTYPES + utils.ALL_INT_DTYPES)
def test_bitwise_left_shift_(shapes, dtype):
    shape_a, shape_b = shapes
    res_a = torch.randint(0, 100, shape_a, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    res_b = torch.randint(0, 8, shape_b, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_a = utils.to_reference(res_a.clone())
    ref_b = utils.to_reference(res_b)

    ref_a.bitwise_left_shift_(ref_b)
    flag_gems.bitwise_left_shift_(res_a, res_b)
    utils.gems_assert_close(res_a, ref_a, dtype)


@pytest.mark.bitwise_left_shift_
@pytest.mark.parametrize("shape", SHIFT_SHAPES)
@pytest.mark.parametrize("dtype", SHIFT_DTYPES)
def test_bitwise_left_shift__shapes(shape, dtype):
    res_a = utils.randint_in_range(shape, dtype).to(flag_gems.device)
    res_b = torch.randint(0, 8, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_a = utils.to_reference(res_a.clone())
    ref_b = utils.to_reference(res_b)

    ref_a.bitwise_left_shift_(ref_b)
    flag_gems.bitwise_left_shift_(res_a, res_b)
    utils.gems_assert_close(res_a, ref_a, dtype)


@pytest.mark.bitwise_left_shift_
@pytest.mark.parametrize("range_pair", [(-1, 1), (0, 1), (-1, 0), (0, None), (None, 0)])
@pytest.mark.parametrize("dtype", SHIFT_DTYPES)
def test_bitwise_left_shift__ranges(range_pair, dtype):
    low, high = range_pair
    shape = (256,)
    info = torch.iinfo(dtype)
    if low is None:
        low = info.min
    if high is None:
        high = info.max
    if dtype is torch.uint8:
        # randint rejects negative bounds for unsigned dtypes.
        low = max(low, 0)
    # randint's high is exclusive, so the upper bound itself must be pinned
    # explicitly: sampling [low, high) never produces `high`, and clamping to
    # info.max would silently drop dtype_max from the [0, dtype_max] range.
    if low < high:
        res_a = torch.randint(low, high, shape, dtype=dtype, device="cpu")
    else:
        # uint8 clamps (-1, 0) down to (0, 0); randint needs low < high.
        res_a = torch.full(shape, low, dtype=dtype, device="cpu")
    res_a.view(-1)[0] = low
    res_a.view(-1)[1] = high
    res_a = res_a.to(flag_gems.device)
    res_b = torch.randint(0, 8, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_a = utils.to_reference(res_a.clone())
    ref_b = utils.to_reference(res_b)

    ref_a.bitwise_left_shift_(ref_b)
    flag_gems.bitwise_left_shift_(res_a, res_b)
    utils.gems_assert_close(res_a, ref_a, dtype)


@pytest.mark.bitwise_left_shift_
@pytest.mark.parametrize("count", SHIFT_COUNTS)
@pytest.mark.parametrize("dtype", SHIFT_DTYPES)
def test_bitwise_left_shift__counts(count, dtype):
    shape = (256,)
    res_a = utils.randint_in_range(shape, dtype).to(flag_gems.device)
    info = torch.iinfo(dtype)
    # torch.full requires the fill value to fit the dtype; out-of-range counts
    # are skipped (ATen itself cannot represent them in this dtype either).
    if not (info.min <= count <= info.max):
        pytest.skip(f"shift count {count} out of range for {dtype}")
    res_b = torch.full(shape, count, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_a = utils.to_reference(res_a.clone())
    ref_b = utils.to_reference(res_b)

    ref_a.bitwise_left_shift_(ref_b)
    flag_gems.bitwise_left_shift_(res_a, res_b)
    utils.gems_assert_close(res_a, ref_a, dtype)


@pytest.mark.bitwise_left_shift_
@pytest.mark.parametrize(
    "broadcast_shapes", [((1024, 1024), (1024,)), ((20, 320, 15), (320, 15))]
)
@pytest.mark.parametrize("dtype", SHIFT_DTYPES)
def test_bitwise_left_shift__broadcast(broadcast_shapes, dtype):
    shape_a, shape_b = broadcast_shapes
    res_a = utils.randint_in_range(shape_a, dtype).to(flag_gems.device)
    res_b = torch.randint(0, 8, shape_b, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_a = utils.to_reference(res_a.clone())
    ref_b = utils.to_reference(res_b)

    ref_a.bitwise_left_shift_(ref_b)
    flag_gems.bitwise_left_shift_(res_a, res_b)
    utils.gems_assert_close(res_a, ref_a, dtype)


@pytest.mark.bitwise_left_shift_
@pytest.mark.parametrize("value_pair", [(True, True), (False, True), (True, False)])
@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
def test_bitwise_left_shift__bools(value_pair, dtype):
    a_val, b_val = value_pair
    shape = (256,)
    res_a = torch.full(shape, 1 if a_val else 0, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    res_b = torch.full(shape, 1 if b_val else 0, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_a = utils.to_reference(res_a.clone())
    ref_b = utils.to_reference(res_b)

    ref_a.bitwise_left_shift_(ref_b)
    flag_gems.bitwise_left_shift_(res_a, res_b)
    utils.gems_assert_close(res_a, ref_a, dtype)
