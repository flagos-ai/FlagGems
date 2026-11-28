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

DIAGONAL_COPY_OFFSETS = [0, 1, -1, 2, -2]
# Cover common dimension pairs for 2D, 3D, and 4D tensors
DIAGONAL_COPY_DIMS = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 3),
]


# Various tensor ranks (3D/4D), square/non-square dims, and edge cases (1-sized dims)
DIAGONAL_COPY_SHAPES = [
    (3, 4, 5),
    (5, 4, 3),
    (2, 3, 4, 5),
    (1, 2, 3),
    (3, 3, 3),
]


@pytest.mark.diagonal_copy
@pytest.mark.parametrize("shape", DIAGONAL_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("offset", DIAGONAL_COPY_OFFSETS)
@pytest.mark.parametrize("dims", DIAGONAL_COPY_DIMS)
def test_diagonal_copy(shape, dtype, offset, dims):
    dim1, dim2 = dims
    # Validate dims are within bounds
    if dim1 >= len(shape) or dim2 >= len(shape):
        pytest.skip("dim out of bounds for shape")
    if dim1 == dim2:
        pytest.skip("dim1 and dim2 must be different")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.diagonal_copy(ref_inp, offset, dim1, dim2)
    res_out = flag_gems.diagonal_copy(inp, offset, dim1, dim2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diagonal_copy
@pytest.mark.parametrize("shape", DIAGONAL_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
@pytest.mark.parametrize("offset", [0, 1, -1])
@pytest.mark.parametrize("dims", [(0, 1), (1, 2), (0, 2)])
def test_diagonal_copy_int(shape, dtype, offset, dims):
    dim1, dim2 = dims
    # Validate dims are within bounds
    if dim1 >= len(shape) or dim2 >= len(shape):
        pytest.skip("dim out of bounds for shape")
    if dim1 == dim2:
        pytest.skip("dim1 and dim2 must be different")

    if flag_gems.vendor_name == "cambricon" and dtype == torch.int16:
        inp = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        inp = torch.randint(-1000, 1000, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.diagonal_copy(ref_inp, offset, dim1, dim2)
    res_out = flag_gems.diagonal_copy(inp, offset, dim1, dim2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.diagonal_copy
@pytest.mark.parametrize("shape", DIAGONAL_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES + utils.FP8_DTYPES)
@pytest.mark.parametrize("offset", [0, 1, -1])
def test_diagonal_copy_low_precision(shape, dtype, offset):
    if dtype in utils.FP8_DTYPES:
        # A pure bit copy, so every value must round-trip exactly.
        inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).to(dtype)
    else:
        # Stay inside each dtype's range (uint8 rejects negatives).
        lo, hi = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        inp = torch.randint(max(lo, -100), min(hi, 100) + 1, shape, dtype=dtype).to(
            flag_gems.device
        )

    ref_inp = utils.to_reference(inp)
    ref_out = torch.diagonal_copy(ref_inp, offset, 0, 1)
    res_out = flag_gems.diagonal_copy(inp, offset, 0, 1)

    assert res_out.dtype == ref_out.dtype
    if dtype in utils.FP8_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.diagonal_copy
def test_diagonal_copy_empty():
    # Test edge case: empty tensor
    shape = (0, 3, 4)
    dtype = torch.float32
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.diagonal_copy(ref_inp, 0, 1, 2)
    res_out = flag_gems.diagonal_copy(inp, 0, 1, 2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.diagonal_copy
def test_diagonal_copy_single_element():
    # Test edge case: single element
    shape = (1, 1, 1)
    dtype = torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.diagonal_copy(ref_inp, 0, 1, 2)
    res_out = flag_gems.diagonal_copy(inp, 0, 1, 2)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Extended coverage: value-range grid x shape x dtype, scalar params, broadcast.
# ---------------------------------------------------------------------------

# Five value ranges per the coverage spec, over [-1, 1] plus the dtype edges.
DIAGONAL_VALUE_RANGES = [
    ("[-1,1]", -1.0, 1.0),
    ("[0,1]", 0.0, 1.0),
    ("[-1,0]", -1.0, 0.0),
    ("[0,max]", 0.0, None),
    ("[min,0]", None, 0.0),
]


def _range_bounds_diag(name, dtype):
    """Resolve the symbolic (lo, hi) pair for `name` against `dtype`."""
    lo, hi = dict((r[0], (r[1], r[2])) for r in DIAGONAL_VALUE_RANGES)[name]
    if dtype.is_floating_point:
        info = torch.finfo(dtype)
    else:
        info = torch.iinfo(dtype)
    if lo is None:
        lo = info.min
    if hi is None:
        hi = info.max
    return lo, hi


def _make_diag_input(shape, dtype, lo, hi):
    """Build a tensor in [lo, hi] for `dtype`, on the active device."""
    if dtype.is_floating_point:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(lo, hi)
    else:
        lo_i = int(max(lo, torch.iinfo(dtype).min))
        hi_i = int(min(hi, torch.iinfo(dtype).max))
        inp = torch.randint(lo_i, hi_i + 1, shape, dtype=dtype, device=flag_gems.device)
    return inp


@pytest.mark.diagonal_copy
@pytest.mark.parametrize("rng", DIAGONAL_VALUE_RANGES, ids=lambda r: r[0])
@pytest.mark.parametrize("shape", DIAGONAL_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("offset", DIAGONAL_COPY_OFFSETS)
@pytest.mark.parametrize("dims", DIAGONAL_COPY_DIMS)
def test_diagonal_copy_value_ranges(shape, rng, dtype, offset, dims):
    dim1, dim2 = dims
    if dim1 >= len(shape) or dim2 >= len(shape):
        pytest.skip("dim out of bounds for shape")
    if dim1 == dim2:
        pytest.skip("dim1 and dim2 must be different")

    _, lo, hi = rng
    lo, hi = _range_bounds_diag(rng[0], dtype)
    inp = _make_diag_input(shape, dtype, lo, hi)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.diagonal_copy(ref_inp, offset, dim1, dim2)
    res_out = flag_gems.diagonal_copy(inp, offset, dim1, dim2)
    utils.gems_assert_close(res_out, ref_out, dtype)


# Scalar params sweep: bool, int boundaries, float boundaries incl. nan/inf.
DIAGONAL_PARAMS = [
    True,
    False,
    0,
    1,
    -1,
    255,
    -255,
    2**31 - 1,
    -(2**31),
    2**63 - 1,
    -(2**63),
    0.0,
    -0.0,
    1.0,
    -1.0,
    float("inf"),
    float("-inf"),
    float("nan"),
]


@pytest.mark.diagonal_copy
@pytest.mark.parametrize("val", DIAGONAL_PARAMS)
@pytest.mark.parametrize("shape", DIAGONAL_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("offset", [0, 1, -1])
@pytest.mark.parametrize("dims", [(0, 1), (1, 2)])
def test_diagonal_copy_params(shape, val, dtype, offset, dims):
    dim1, dim2 = dims
    if dim1 >= len(shape) or dim2 >= len(shape):
        pytest.skip("dim out of bounds for shape")
    if dim1 == dim2:
        pytest.skip("dim1 and dim2 must be different")
    # int64-range scalars overflow at::Half on GPU init, matching ATen's
    # integer-scalar conversion window; skip those combos.
    if dtype is torch.float16 and isinstance(val, int) and not isinstance(val, bool):
        if not -(2**15) <= val <= 2**15 - 1:
            pytest.skip("int64-range scalar overflows fp16 in ATen")

    inp = torch.full(shape, float(val), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.diagonal_copy(ref_inp, offset, dim1, dim2)
    res_out = flag_gems.diagonal_copy(inp, offset, dim1, dim2)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# Broadcast shapes: 1-D and multi-D broadcasts against the 4D base.
DIAGONAL_BROADCAST_CASES = [
    ((16, 128, 64, 60), (60,)),
    ((16, 128, 64, 60), (1, 60)),
    ((16, 128, 64, 60), (1, 1, 64, 60)),
    ((16, 128, 64, 60), (16, 1, 1, 1)),
]


@pytest.mark.diagonal_copy
@pytest.mark.parametrize("shapes", DIAGONAL_BROADCAST_CASES, ids=lambda c: str(c))
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("offset", [0, 1, -1])
def test_diagonal_copy_broadcast(shapes, dtype, offset):
    # diagonal_copy selects whole diagonal blocks, so broadcast means feeding
    # a stride-0 expanded view of the small tensor (duplicated data, same
    # diagonal values everywhere a block repeats).
    shape, other_shape = shapes
    inp = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)
    expanded = inp.expand(shape)
    ref_inp = utils.to_reference(expanded)

    ref_out = torch.diagonal_copy(ref_inp, offset, 0, 1)
    res_out = flag_gems.diagonal_copy(expanded, offset, 0, 1)
    utils.gems_assert_close(res_out, ref_out, dtype)
