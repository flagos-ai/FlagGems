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

import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

time.sleep(0.5)


# Shapes that exercise 1D / 2D / higher-rank paths together with dim choices,
# including negative dims.
SHAPE_DIM_CASES = [
    ((8,), 0),
    ((8,), -1),
    ((16, 4), 0),
    ((16, 4), 1),
    ((16, 4), -1),
    ((4, 16), 0),
    ((6, 5, 4), 0),
    ((6, 5, 4), 1),
    ((6, 5, 4), 2),
    ((6, 5, 4), -2),
    ((3, 4, 5, 6), 2),
    ((3, 4, 5, 6), -1),
]


def _make_input(shape, dim, dtype, pattern):
    """Build inputs stressing different consecutive-duplicate distributions.

    Unlike ``unique_dim``, only *adjacent* repeats matter, so the patterns
    build runs of consecutive equal slices of varying length.
    """
    size_dim = shape[dim]
    if pattern == "runs":
        # Alternate runs of 1-3 identical random slices.
        slice_shape = list(shape)
        slice_shape[dim] = 1
        if dtype in utils.INT_DTYPES:
            base = torch.randint(-3, 3, slice_shape, device=flag_gems.device).to(dtype)
        else:
            base = torch.randn(slice_shape, dtype=dtype, device=flag_gems.device)
        pieces = []
        while sum(p.size(dim) for p in pieces) < size_dim:
            if pieces and torch.rand((), device=flag_gems.device) < 0.6:
                pieces.append(pieces[-1])
            else:
                pieces.append(
                    torch.randint(-3, 3, slice_shape, device=flag_gems.device).to(dtype)
                    if dtype in utils.INT_DTYPES
                    else torch.randn(slice_shape, dtype=dtype, device=flag_gems.device)
                )
        return torch.cat(pieces, dim=dim)
    if pattern == "all_unique":
        if dtype in utils.INT_DTYPES:
            return torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if pattern == "all_duplicate":
        slice_shape = list(shape)
        slice_shape[dim] = 1
        if dtype in utils.INT_DTYPES:
            base = torch.randint(-3, 3, slice_shape, device=flag_gems.device).to(dtype)
        else:
            base = torch.randn(slice_shape, dtype=dtype, device=flag_gems.device)
        repeat = [1] * len(shape)
        repeat[dim] = size_dim
        return base.repeat(*repeat)
    raise ValueError(f"unknown pattern {pattern}")


@pytest.mark.unique_dim_consecutive
@pytest.mark.parametrize("shape, dim", SHAPE_DIM_CASES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
@pytest.mark.parametrize("pattern", ["runs", "all_unique", "all_duplicate"])
@pytest.mark.parametrize("return_inverse", [False, True])
@pytest.mark.parametrize("return_counts", [False, True])
def test_unique_dim_consecutive_int(
    shape, dim, dtype, pattern, return_inverse, return_counts
):
    inp = _make_input(shape, dim, dtype, pattern)
    ref_inp = utils.to_reference(inp, False)

    res_out, res_inv, res_counts = flag_gems.unique_dim_consecutive(
        inp,
        dim=dim,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )
    ref_out, ref_inv, ref_counts = torch.ops.aten.unique_dim_consecutive.default(
        ref_inp,
        dim,
        return_inverse,
        return_counts,
    )

    utils.gems_assert_equal(res_out, ref_out)

    if return_inverse:
        utils.gems_assert_equal(res_inv, ref_inv)
    else:
        assert res_inv.numel() == 0

    if return_counts:
        utils.gems_assert_equal(res_counts, ref_counts)
    else:
        assert res_counts.numel() == 0


@pytest.mark.unique_dim_consecutive
@pytest.mark.parametrize(
    "shape, dim",
    [
        ((16, 4), 0),
        ((16, 4), 1),
        ((4, 8, 6), 1),
        ((4, 8, 6), -1),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("pattern", ["runs", "all_duplicate"])
@pytest.mark.parametrize("return_inverse", [True])
@pytest.mark.parametrize("return_counts", [True])
def test_unique_dim_consecutive_float(
    shape, dim, dtype, pattern, return_inverse, return_counts
):
    # Floating point duplicate detection relies on exact equality, so build
    # inputs by repeating slices (no nan / no rounding gymnastics).
    inp = _make_input(shape, dim, dtype, pattern)
    ref_inp = utils.to_reference(inp, False)

    res_out, res_inv, res_counts = flag_gems.unique_dim_consecutive(
        inp,
        dim=dim,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )
    ref_out, ref_inv, ref_counts = torch.ops.aten.unique_dim_consecutive.default(
        ref_inp,
        dim,
        return_inverse,
        return_counts,
    )

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_inv, ref_inv)
    utils.gems_assert_equal(res_counts, ref_counts)


@pytest.mark.unique_dim_consecutive
@pytest.mark.parametrize("return_inverse", [False, True])
@pytest.mark.parametrize("return_counts", [False, True])
def test_unique_dim_consecutive_edge(return_inverse, return_counts):
    # 1D input, single element, all-same and empty inputs.
    inp = torch.tensor(
        [1, 1, 2, 2, 1, 1, 3], device=flag_gems.device, dtype=torch.int32
    )
    ref_inp = utils.to_reference(inp, False)

    res_out, res_inv, res_counts = flag_gems.unique_dim_consecutive(
        inp, 0, return_inverse, return_counts
    )
    ref_out, ref_inv, ref_counts = torch.ops.aten.unique_dim_consecutive.default(
        ref_inp, 0, return_inverse, return_counts
    )
    utils.gems_assert_equal(res_out, ref_out)
    if return_inverse:
        utils.gems_assert_equal(res_inv, ref_inv)
    if return_counts:
        utils.gems_assert_equal(res_counts, ref_counts)

    single = torch.tensor([[7, 7]], device=flag_gems.device, dtype=torch.int32)
    ref_single = utils.to_reference(single, False)
    res_out, res_inv, res_counts = flag_gems.unique_dim_consecutive(
        single, 0, return_inverse, return_counts
    )
    ref_out, ref_inv, ref_counts = torch.ops.aten.unique_dim_consecutive.default(
        ref_single, 0, return_inverse, return_counts
    )
    utils.gems_assert_equal(res_out, ref_out)
    if return_inverse:
        utils.gems_assert_equal(res_inv, ref_inv)
    if return_counts:
        utils.gems_assert_equal(res_counts, ref_counts)

    all_same = torch.ones(4, 3, device=flag_gems.device, dtype=torch.int32)
    ref_same = utils.to_reference(all_same, False)
    res_out, res_inv, res_counts = flag_gems.unique_dim_consecutive(
        all_same, 0, return_inverse, return_counts
    )
    ref_out, ref_inv, ref_counts = torch.ops.aten.unique_dim_consecutive.default(
        ref_same, 0, return_inverse, return_counts
    )
    utils.gems_assert_equal(res_out, ref_out)
    if return_inverse:
        utils.gems_assert_equal(res_inv, ref_inv)
    if return_counts:
        utils.gems_assert_equal(res_counts, ref_counts)

    empty = torch.empty(0, device=flag_gems.device, dtype=torch.int32)
    res_out, res_inv, res_counts = flag_gems.unique_dim_consecutive(
        empty, 0, return_inverse, return_counts
    )
    assert res_out.numel() == 0
    assert res_inv.numel() == 0
    assert res_counts.numel() == 0


@pytest.mark.unique_dim_consecutive
def test_unique_dim_consecutive_wide_rows():
    # Multi-chunk row comparison path (row_len > single tile).
    inp = torch.randint(0, 3, (20, 2500), device=flag_gems.device, dtype=torch.int64)
    ref_inp = utils.to_reference(inp, False)

    res_out, res_inv, res_counts = flag_gems.unique_dim_consecutive(inp, 0, True, True)
    ref_out, ref_inv, ref_counts = torch.ops.aten.unique_dim_consecutive.default(
        ref_inp, 0, True, True
    )
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_inv, ref_inv)
    utils.gems_assert_equal(res_counts, ref_counts)
