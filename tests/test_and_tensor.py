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
# has no float lowering, and ATen's bitwise_and rejects float tensors on both
# CPU and CUDA. fp8 is therefore excluded rather than skipped at runtime.
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

VALUE_RANGES = [
    (-1, 1),
    (0, 1),
    (-1, 0),
    ("dtype_max",),
    ("dtype_min",),
]

# Broadcast bases drawn from the spec's shape list. Each pair is a broadcastable
# operand next to a full-size operand; the leading and interior singleton axes
# cover both 1-D-style and multi-dimensional broadcast.
BROADCAST_SHAPES = (
    [((2, 1, 7), (2, 19, 7))]
    if QUICK_MODE
    else [
        ((1, 1024, 1024), (16, 1024, 1024)),
        ((20, 1, 15), (20, 320, 15)),
        ((16, 1, 64, 60), (16, 128, 64, 60)),
        ((16, 7, 1, 32, 29), (16, 7, 57, 32, 29)),
        ((), (1024, 1024)),
        ((1,), (16, 7, 57, 32, 29)),
    ]
)
# float dtypes are exercised through and_scalar only; the broadcast matrix stays
# on the integer dtypes the operator actually supports.
BROADCAST_DTYPES = (
    [torch.int32]
    if QUICK_MODE
    else [
        d
        for d in (torch.int8, torch.uint8, torch.int32, torch.int64)
        if d is not torch.int64 or utils.int64_is_supported
    ]
)


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


@pytest.mark.and_tensor
@pytest.mark.parametrize("value_range", VALUE_RANGES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_and_tensor(shape, dtype, value_range, caplog):
    inp1 = _randint(shape, dtype, value_range)
    inp2 = _randint(shape, dtype, value_range)

    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.ops.aten.__and__.Tensor(ref_inp1, ref_inp2)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.and_tensor)):
        res_out = flag_gems.and_tensor(inp1, inp2)

    assert f"{utils.gems_log_prefix(flag_gems.and_tensor)} AND" in caplog.text
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.and_tensor
@pytest.mark.parametrize("shape", SHAPES)
def test_and_tensor_bool(shape, caplog):
    inp1 = torch.randint(0, 2, size=shape, dtype=torch.bool, device="cpu").to(
        flag_gems.device
    )
    inp2 = torch.randint(0, 2, size=shape, dtype=torch.bool, device="cpu").to(
        flag_gems.device
    )

    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.ops.aten.__and__.Tensor(ref_inp1, ref_inp2)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.and_tensor)):
        res_out = flag_gems.and_tensor(inp1, inp2)

    assert f"{utils.gems_log_prefix(flag_gems.and_tensor)} AND" in caplog.text
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.and_tensor
@pytest.mark.parametrize("value_range", VALUE_RANGES)
@pytest.mark.parametrize("dtype", BROADCAST_DTYPES)
@pytest.mark.parametrize("shape_pair", BROADCAST_SHAPES)
def test_and_tensor_broadcast(shape_pair, dtype, value_range, caplog):
    lhs_shape, rhs_shape = shape_pair
    inp1 = _randint(lhs_shape, dtype, value_range)
    inp2 = _randint(rhs_shape, dtype, value_range)

    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.ops.aten.__and__.Tensor(ref_inp1, ref_inp2)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.and_tensor)):
        res_out = flag_gems.and_tensor(inp1, inp2)

    assert f"{utils.gems_log_prefix(flag_gems.and_tensor)} AND" in caplog.text
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.and_tensor
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_and_tensor_dtype_boundaries(dtype, caplog):
    """Both operands take the dtype's exact min/max so the AND is boundary-only."""
    info = torch.iinfo(dtype)
    values = [int(info.min), int(info.max), 0, -1, 1]
    inp1 = torch.tensor([values], dtype=dtype, device="cpu").to(flag_gems.device)
    inp2 = torch.tensor([values], dtype=dtype, device="cpu").to(flag_gems.device)

    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.ops.aten.__and__.Tensor(ref_inp1, ref_inp2)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.and_tensor)):
        res_out = flag_gems.and_tensor(inp1, inp2)

    assert f"{utils.gems_log_prefix(flag_gems.and_tensor)} AND" in caplog.text
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.and_tensor
@pytest.mark.parametrize("value_range", VALUE_RANGES)
@pytest.mark.parametrize("dtype", BROADCAST_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_and_tensor_non_contiguous(shape, dtype, value_range, caplog):
    """Transposed inputs force the generic strided kernel instead of the flat path."""
    if len(shape) < 2:
        pytest.skip("transpose needs at least two dims")

    inp1 = _randint(shape, dtype, value_range)
    if inp1.dim() > 2:
        inp1 = inp1.transpose(0, inp1.dim() - 1)
    else:
        inp1 = inp1.t()
    # inp2 always carries the exact same reversed shape as the now-transposed
    # inp1, so the two are elementwise (no broadcasting) but non-contiguous.
    inp2 = _randint(inp1.shape, dtype, value_range)

    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.ops.aten.__and__.Tensor(ref_inp1, ref_inp2)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.and_tensor)):
        res_out = flag_gems.and_tensor(inp1, inp2)

    assert f"{utils.gems_log_prefix(flag_gems.and_tensor)} AND" in caplog.text
    utils.gems_assert_equal(res_out, ref_out)
