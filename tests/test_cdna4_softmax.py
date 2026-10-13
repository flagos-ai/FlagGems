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

"""Tests for the CDNA4-specific split softmax.

The generic softmax test file only uses shapes with either many rows or a short
reduction, so none of them reach the split path; these cover the regime the arch
override exists for. The arch loader imports the override under the top-level
name "cdna4.ops.softmax", so that is where the gate function has to be read from
to be sure the module under test is the one actually installed.
"""

import math
import sys

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

ARCH_SOFTMAX = sys.modules.get("cdna4.ops.softmax")
INSTALLED = ARCH_SOFTMAX is not None and ARCH_SOFTMAX.softmax is flag_gems.softmax

pytestmark = pytest.mark.skipif(
    not INSTALLED,
    reason="CDNA4 split softmax is not the installed softmax on this device",
)

# (shape, dim) pairs the gate is expected to take over. A 1-D input is the case
# that motivated the override: M collapses to 1, so the generic kernel runs the
# whole reduction on a single workgroup. Lengths clear _MIN_SPLIT_N and row counts
# stay low enough to reach _MIN_BLOCKS_PER_ROW chunks, so these engage whatever the
# CU count of the part.
SPLIT_SHAPES = (
    [([1048576], 0)]
    if cfg.QUICK_MODE
    else [
        ([262144], 0),
        ([1048576], 0),
        ([4, 262144], 1),
        ([2, 2, 131072], 2),
        ([1, 98305], 1),
    ]
)

# Shapes the gate must leave alone: reduction too short to pay for the second
# launch, too many rows to reach the chunk minimum, or a non-inner reduction.
GENERIC_SHAPES = (
    [([1, 256], 1)]
    if cfg.QUICK_MODE
    else [
        ([1, 256], 1),
        ([8192], 0),
        ([1, 98303], 1),
        ([4096, 4096], 1),
        ([64, 512, 512], 1),
    ]
)

FLOAT_DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.FLOAT_DTYPES


def mnk(shape, dim):
    n = shape[dim]
    m = math.prod(shape[:dim])
    return m, n, math.prod(shape) // m // n


def split_plan(shape, dim):
    m, n, k = mnk(shape, dim)
    if k != 1:
        return None
    return ARCH_SOFTMAX._split_plan(m, n, 0)


@pytest.mark.softmax
@pytest.mark.parametrize("shape, dim", SPLIT_SHAPES)
def test_cdna4_softmax_gate_engages(shape, dim):
    m, n, _ = mnk(shape, dim)
    plan = split_plan(shape, dim)
    assert plan is not None, f"expected the split path for M={m} N={n}"

    cu = ARCH_SOFTMAX._cu_count(0)
    assert plan >= ARCH_SOFTMAX._MIN_BLOCKS_PER_ROW
    assert plan & (plan - 1) == 0, "workgroups per row must be a power of two"
    # Every workgroup needs a whole tile, and the grid must not overshoot the
    # block count the sweep found best.
    assert plan * ARCH_SOFTMAX._TILE_N <= n
    assert m * plan <= ARCH_SOFTMAX._BLOCKS_PER_CU * cu


@pytest.mark.softmax
@pytest.mark.parametrize("shape, dim", GENERIC_SHAPES)
def test_cdna4_softmax_gate_defers(shape, dim):
    assert (
        split_plan(shape, dim) is None
    ), f"{shape} dim={dim} should fall through to the generic implementation"


@pytest.mark.softmax
def test_cdna4_softmax_gate_limits():
    """Pin both limits from both sides.

    Written against the constants and the CU count rather than literals, so the
    assertions keep describing the intended rule on a part with a different CU
    count.
    """
    cu = ARCH_SOFTMAX._cu_count(0)

    # Nothing below the minimum length, however starved the grid is.
    assert ARCH_SOFTMAX._split_plan(1, ARCH_SOFTMAX._MIN_SPLIT_N - 1, 0) is None
    assert ARCH_SOFTMAX._split_plan(1, ARCH_SOFTMAX._MIN_SPLIT_N, 0) is not None

    # The chunk minimum is what bounds the row count: num_blocks halves as M
    # doubles, so there is a largest M that still reaches _MIN_BLOCKS_PER_ROW.
    # Long enough that the length cap is not the one biting.
    n = 64 * ARCH_SOFTMAX._MIN_SPLIT_N
    row_cap = (
        ARCH_SOFTMAX._BLOCKS_PER_CU * cu // ARCH_SOFTMAX._MIN_BLOCKS_PER_ROW
    )
    assert ARCH_SOFTMAX._split_plan(row_cap, n, 0) == ARCH_SOFTMAX._MIN_BLOCKS_PER_ROW
    assert ARCH_SOFTMAX._split_plan(row_cap + 1, n, 0) is None

    # Every admitted plan is a power of two at or above the chunk minimum.
    for m in (1, 2, 8, 32, row_cap):
        plan = ARCH_SOFTMAX._split_plan(m, n, 0)
        assert plan >= ARCH_SOFTMAX._MIN_BLOCKS_PER_ROW
        assert plan & (plan - 1) == 0


@pytest.mark.softmax
@pytest.mark.parametrize("shape, dim", SPLIT_SHAPES + GENERIC_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cdna4_softmax(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.softmax(inp, dim=dim)

    utils.gems_assert_close(
        res_out, ref_out, dtype, equal_nan=True, reduce_dim=shape[dim]
    )


@pytest.mark.softmax
@pytest.mark.parametrize("shape, dim", SPLIT_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cdna4_softmax_out(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    # Deliberately the wrong size, so the resize path is covered too.
    out = torch.empty((1,), dtype=dtype, device=flag_gems.device)

    ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        torch.ops.aten._softmax.out(inp, dim, False, out=out)

    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True, reduce_dim=shape[dim])


@pytest.mark.softmax
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cdna4_softmax_half_to_float(dtype):
    if dtype is torch.float32:
        pytest.skip("half_to_float only applies to reduced-precision input")
    shape, dim = SPLIT_SHAPES[0]
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._softmax(inp, dim, True)

    assert res_out.dtype is torch.float32
    utils.gems_assert_close(
        res_out, ref_out, torch.float32, equal_nan=True, reduce_dim=shape[dim]
    )


@pytest.mark.softmax
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cdna4_softmax_neg_inf(dtype):
    """Masked attention rows are the reason the combine needs an -inf guard.

    A chunk that is entirely -inf must contribute a zero sum instead of exp(nan),
    and a row that is entirely -inf has to come out the same as the single
    workgroup version would produce.
    """
    shape, dim = SPLIT_SHAPES[0]
    n = shape[dim]
    tile = ARCH_SOFTMAX._TILE_N

    rows = {
        # Whole leading chunks masked out, so some workgroups see only -inf.
        "leading_chunks": lambda t: t[..., : min(8 * tile, n // 2)].fill_(
            float("-inf")
        ),
        # Only one finite element in the row, far from the first chunk.
        "single_finite": lambda t: (
            t.fill_(float("-inf")),
            t[..., n // 2].fill_(1.0),
        ),
        # Every element masked: the reference is nan and the arch path must
        # agree rather than producing a number.
        "all_masked": lambda t: t.fill_(float("-inf")),
    }

    for name, mutate in rows.items():
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        mutate(inp)
        ref_inp = utils.to_reference(inp, True)

        ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
        with flag_gems.use_gems():
            res_out = torch.nn.functional.softmax(inp, dim=dim)

        assert split_plan(shape, dim) is not None, name
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, reduce_dim=n)


@pytest.mark.softmax
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cdna4_softmax_at_row_cap(dtype):
    """Cover grid.y at the row cap: every row combines its own set of partials.

    The shapes in SPLIT_SHAPES only reach M=4, so without this the widest grid the
    split path ever runs with is a handful of rows.
    """
    cu = ARCH_SOFTMAX._cu_count(0)
    # The largest row count that still reaches the chunk minimum.
    m = ARCH_SOFTMAX._BLOCKS_PER_CU * cu // ARCH_SOFTMAX._MIN_BLOCKS_PER_ROW
    n = 4 * ARCH_SOFTMAX._MIN_SPLIT_N
    if ARCH_SOFTMAX._split_plan(m, n, 0) is None:
        pytest.skip(f"the gate does not take M={m} N={n} on a {cu} CU device")

    inp = torch.randn((m, n), dtype=dtype, device=flag_gems.device)
    # One fully masked row, so the all -inf guard is hit with live siblings.
    inp[0].fill_(float("-inf"))
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.softmax(ref_inp, dim=-1)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.softmax(inp, dim=-1)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, reduce_dim=n)


@pytest.mark.softmax
def test_cdna4_softmax_non_contiguous():
    shape, dim = SPLIT_SHAPES[0]
    n = shape[dim]
    base = torch.randn(
        shape[:dim] + [2 * n], dtype=torch.float32, device=flag_gems.device
    )
    inp = base[..., ::2]
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.softmax(inp, dim=dim)

    utils.gems_assert_close(
        res_out, ref_out, torch.float32, equal_nan=True, reduce_dim=n
    )
