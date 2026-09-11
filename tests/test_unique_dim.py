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
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

random.seed(time.time() // 100)

device = flag_gems.device


# Shapes that exercise 2D / 3D / higher-rank paths together with dim choices,
# including negative dims and a "no other dim" 1D layout.
SHAPE_DIM_CASES = [
    ((8,), 0),
    ((8,), -1),
    ((16, 4), 0),
    ((16, 4), 1),
    ((16, 4), -1),
    ((4, 16), 0),
    ((32, 32), 0),
    ((32, 32), 1),
    ((6, 5, 4), 0),
    ((6, 5, 4), 1),
    ((6, 5, 4), 2),
    ((6, 5, 4), -2),
    ((3, 4, 5, 6), 2),
    ((3, 4, 5, 6), -1),
]


def _make_input(shape, dim, dtype, pattern):
    """Build inputs that stress different duplicate distributions along dim."""
    size_dim = shape[dim]
    if pattern == "few_unique":
        # Build the tensor by repeating ``k`` random slices along the target dim.
        k = max(1, size_dim // 4)
        slice_shape = list(shape)
        slice_shape[dim] = k
        if dtype in utils.INT_DTYPES:
            base = torch.randint(-3, 3, slice_shape, device=flag_gems.device).to(dtype)
        else:
            base = torch.randn(slice_shape, dtype=dtype, device=flag_gems.device)
        # Repeat the base slices to fill ``size_dim`` entries.
        repeat = [1] * len(shape)
        repeat[dim] = (size_dim + k - 1) // k
        rep = base.repeat(*repeat)
        idx = torch.arange(size_dim, device=flag_gems.device) % k
        # Permute repeats so duplicates are not consecutive.
        perm = torch.randperm(size_dim, device=flag_gems.device)
        idx = idx[perm]
        out = torch.index_select(rep, dim, idx[: rep.size(dim)])
        return out
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


@pytest.mark.unique_dim
@pytest.mark.parametrize("shape, dim", SHAPE_DIM_CASES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
@pytest.mark.parametrize("pattern", ["few_unique", "all_unique", "all_duplicate"])
@pytest.mark.parametrize("return_inverse", [False, True])
@pytest.mark.parametrize("return_counts", [False, True])
def test_unique_dim_int(shape, dim, dtype, pattern, return_inverse, return_counts):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp = _make_input(shape, dim, dtype, pattern)
    ref_inp = utils.to_reference(inp, False)

    with flag_gems.use_gems():
        res = torch.unique(
            inp,
            sorted=True,
            return_inverse=return_inverse,
            return_counts=return_counts,
            dim=dim,
        )
    ref = torch.unique(
        ref_inp,
        sorted=True,
        return_inverse=return_inverse,
        return_counts=return_counts,
        dim=dim,
    )

    if not (return_inverse or return_counts):
        res_out, ref_out = res, ref
        utils.gems_assert_equal(res_out, ref_out)
        return

    if return_inverse and return_counts:
        res_out, res_inv, res_counts = res
        ref_out, ref_inv, ref_counts = ref
    elif return_inverse:
        res_out, res_inv = res
        ref_out, ref_inv = ref
        res_counts = ref_counts = None
    else:
        res_out, res_counts = res
        ref_out, ref_counts = ref
        res_inv = ref_inv = None

    utils.gems_assert_equal(res_out, ref_out)
    if res_inv is not None:
        utils.gems_assert_equal(res_inv, ref_inv)
    if res_counts is not None:
        utils.gems_assert_equal(res_counts, ref_counts)


@pytest.mark.unique_dim
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
@pytest.mark.parametrize("return_inverse", [True])
@pytest.mark.parametrize("return_counts", [True])
def test_unique_dim_float(shape, dim, dtype, return_inverse, return_counts):
    # Floating point duplicate detection relies on exact equality, so build
    # inputs by repeating slices (no nan / no rounding gymnastics).
    inp = _make_input(shape, dim, dtype, "few_unique")
    ref_inp = utils.to_reference(inp, False)

    with flag_gems.use_gems():
        res_out, res_inv, res_counts = torch.unique(
            inp,
            sorted=True,
            return_inverse=return_inverse,
            return_counts=return_counts,
            dim=dim,
        )
    ref_out, ref_inv, ref_counts = torch.unique(
        ref_inp,
        sorted=True,
        return_inverse=return_inverse,
        return_counts=return_counts,
        dim=dim,
    )

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_inv, ref_inv)
    utils.gems_assert_equal(res_counts, ref_counts)


@pytest.mark.unique_dim
@pytest.mark.parametrize("return_inverse", [False, True])
@pytest.mark.parametrize("return_counts", [False, True])
def test_unique_dim_aten_returns_three_tensors(return_inverse, return_counts):
    inp = torch.tensor(
        [[2, 0], [1, 0], [2, 0], [0, 0]],
        dtype=torch.int32,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, False)

    with flag_gems.use_gems():
        res_out, res_inv, res_counts = torch.ops.aten.unique_dim.default(
            inp, 0, True, return_inverse, return_counts
        )
    ref_out, ref_inv, ref_counts = torch.ops.aten.unique_dim.default(
        ref_inp, 0, True, return_inverse, return_counts
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
