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
from .conftest import QUICK_MODE

# Bitwise AND is defined for integer and bool tensors only: Triton's `x & y`
# has no float lowering, and ATen's bitwise_and rejects float/bool tensors on
# both CPU and CUDA. fp8 is therefore excluded rather than skipped at runtime.
INT_DTYPES = (
    [torch.int32]
    if QUICK_MODE
    else [
        d
        for d in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
        if d is not torch.int64 or utils.int64_is_supported
    ]
)

SHAPES = (
    [(2, 19, 7)]
    if QUICK_MODE
    else [
        (),
        (1,),
        (256,),
        (1024, 1024),
        (20, 320, 15),
        (16, 128, 64, 60),
        (16, 7, 57, 32, 29),
    ]
)

# Scalar operands as a scalar-shaped tensor, grouped by the spec's categories.
# int group: zero, the 0x00FF mask the existing test used, -1 (all bits set)
# and the int16 boundaries.
INT_SCALARS = [0, 0x00FF, -1, 0x7FFF, -0x8000]
BOOL_SCALARS = [True, False]


def _bounds(dtype, value_range):
    """Clamp a spec value range to what ``torch.randint`` accepts for dtype."""
    if dtype is torch.uint8:
        lo, hi = 0, 0xFF
    elif dtype is torch.int8:
        lo, hi = -0x80, 0x7F
    else:
        info = torch.iinfo(dtype)
        lo, hi = int(info.min), int(info.max)

    if len(value_range) == 2:
        low, high = value_range
        return max(low, lo), min(high, hi)
    if value_range[0] == "dtype_max":
        return max(0, lo), hi
    return lo, max(0, lo) + 1  # "dtype_min": [dtype_min, 0)


def _randint(shape, dtype, value_range):
    low, high = _bounds(dtype, value_range)
    if low >= high:
        high = low + 1
    return torch.randint(low, high, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )


VALUE_RANGES = [
    (-1, 1),
    (0, 1),
    (-1, 0),
    ("dtype_max",),
    ("dtype_min",),
]

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason="enflame GCU runtime aborts (core dumped) when running this op",
)


@pytest.mark.and_scalar
@pytest.mark.parametrize("scalar", BOOL_SCALARS)
@pytest.mark.parametrize("shape", SHAPES)
def test_and_scalar_bool(shape, scalar, caplog):
    inp1 = torch.randint(0, 2, size=shape, dtype=torch.bool, device="cpu").to(
        flag_gems.device
    )
    ref_inp1 = utils.to_reference(inp1)

    ref_out = torch.ops.aten.__and__.Scalar(ref_inp1, scalar)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.and_scalar)):
        res_out = flag_gems.and_scalar(inp1, scalar)

    assert f"{utils.gems_log_prefix(flag_gems.and_scalar)} AND_SCALAR" in caplog.text
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.and_scalar
@pytest.mark.parametrize("scalar", INT_SCALARS)
@pytest.mark.parametrize("value_range", VALUE_RANGES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_and_scalar_int(shape, dtype, value_range, scalar, caplog):
    inp1 = _randint(shape, dtype, value_range)
    ref_inp1 = utils.to_reference(inp1)

    ref_out = torch.ops.aten.__and__.Scalar(ref_inp1, scalar)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.and_scalar)):
        res_out = flag_gems.and_scalar(inp1, scalar)

    assert f"{utils.gems_log_prefix(flag_gems.and_scalar)} AND_SCALAR" in caplog.text
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.and_scalar
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_and_scalar_dtype_boundary_scalars(shape, dtype, caplog):
    """Scalars sitting exactly on the tensor dtype's min/max boundaries."""
    info = torch.iinfo(dtype)
    inp1 = torch.randint(
        int(info.min), int(info.max), shape, dtype=dtype, device="cpu"
    ).to(flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)

    for scalar in (int(info.min), int(info.max), 0):
        ref_out = torch.ops.aten.__and__.Scalar(ref_inp1, scalar)
        with caplog.at_level(
            "DEBUG", logger=utils.gems_log_logger(flag_gems.and_scalar)
        ):
            res_out = flag_gems.and_scalar(inp1, scalar)

        assert (
            f"{utils.gems_log_prefix(flag_gems.and_scalar)} AND_SCALAR" in caplog.text
        )
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.and_scalar
@pytest.mark.parametrize("scalar", [0, 1, -1])
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_and_scalar_uniform_input(shape, dtype, scalar, caplog):
    """Degenerate all-equal inputs isolate the scalar path from data variety.

    0b0101_0101 fits every integer dtype in the matrix, so the same bit pattern
    is reused instead of a random value per dtype.
    """
    inp1 = torch.full(shape, 0b0101_0101, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp1 = utils.to_reference(inp1)

    ref_out = torch.ops.aten.__and__.Scalar(ref_inp1, scalar)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.and_scalar)):
        res_out = flag_gems.and_scalar(inp1, scalar)

    assert f"{utils.gems_log_prefix(flag_gems.and_scalar)} AND_SCALAR" in caplog.text
    utils.gems_assert_equal(res_out, ref_out)
