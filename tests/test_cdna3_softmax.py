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

"""Tests for the CDNA3-specific split softmax.

The generic softmax test file only uses shapes with either many rows or a short
reduction, so none of them reach the split path; these cover the regime the arch
override exists for. The arch loader imports the override under the top-level
name "cdna3.ops.softmax", so that is where the gate function has to be read from
to be sure the module under test is the one actually installed.
"""

import math
import sys

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

ARCH_SOFTMAX = sys.modules.get("cdna3.ops.softmax")
INSTALLED = ARCH_SOFTMAX is not None and ARCH_SOFTMAX.softmax is flag_gems.softmax

pytestmark = pytest.mark.skipif(
    not INSTALLED,
    reason="CDNA3 split softmax is not the installed softmax on this device",
)

# (shape, dim) pairs the gate is expected to take over at every float width. A 1-D
# input is the case that motivated the override: M collapses to 1, so the generic
# kernel runs the whole reduction on a single workgroup. Row counts stay inside the
# row budget and lengths clear the minimum, so these engage regardless of dtype.
SPLIT_SHAPES = (
    [([262144], 0)]
    if cfg.QUICK_MODE
    else [
        ([262144], 0),
        ([1048576], 0),
        ([4, 262144], 1),
        ([2, 2, 65536], 2),
        # Not a multiple of the tile, so the masked tail of every chunk is live.
        ([1, 65537], 1),
    ]
)

# Shapes the gate must leave alone at every float width: reduction too short to pay
# for the second launch, too many rows to be starved, or a non-inner reduction.
GENERIC_SHAPES = (
    [([1, 256], 1)]
    if cfg.QUICK_MODE
    else [
        ([1, 256], 1),
        ([8192], 0),
        # One element short of the minimum, so the length limit is pinned live.
        ([1, 65535], 1),
        ([4096, 4096], 1),
        ([64, 512, 512], 1),
    ]
)

FLOAT_DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.FLOAT_DTYPES

# Element widths the row budget treats differently, and the widths the suite has to
# cover on both sides of that split.
NARROW = ARCH_SOFTMAX._NARROW_MAX_ITEMSIZE if INSTALLED else 2
WIDE = NARROW + 2
ITEMSIZES = [NARROW, WIDE]


def mnk(shape, dim):
    n = shape[dim]
    m = math.prod(shape[:dim])
    return m, n, math.prod(shape) // m // n


def itemsize_of(dtype):
    return torch.finfo(dtype).bits // 8


def split_plan(shape, dim, itemsize):
    m, n, k = mnk(shape, dim)
    if k != 1:
        return None
    return ARCH_SOFTMAX._split_plan(m, n, itemsize, 0)


def row_budget(itemsize):
    cus_per_row = (
        ARCH_SOFTMAX._NARROW_MIN_CUS_PER_ROW
        if itemsize <= NARROW
        else ARCH_SOFTMAX._WIDE_MIN_CUS_PER_ROW
    )
    return ARCH_SOFTMAX._cu_count(0) // cus_per_row


@pytest.mark.softmax
@pytest.mark.parametrize("itemsize", ITEMSIZES)
@pytest.mark.parametrize("shape, dim", SPLIT_SHAPES)
def test_cdna3_softmax_gate_engages(shape, dim, itemsize):
    m, n, _ = mnk(shape, dim)
    plan = split_plan(shape, dim, itemsize)
    assert plan is not None, f"expected the split path for M={m} N={n}"

    cu = ARCH_SOFTMAX._cu_count(0)
    assert plan >= ARCH_SOFTMAX._MIN_BLOCKS_PER_ROW
    assert plan & (plan - 1) == 0, "workgroups per row must be a power of two"
    # Every workgroup needs a whole tile, and the grid must not overshoot the
    # block count the sweep found best.
    assert plan * ARCH_SOFTMAX._TILE_N <= n
    assert m * plan <= ARCH_SOFTMAX._BLOCKS_PER_CU * cu


@pytest.mark.softmax
@pytest.mark.parametrize("itemsize", ITEMSIZES)
@pytest.mark.parametrize("shape, dim", GENERIC_SHAPES)
def test_cdna3_softmax_gate_defers(shape, dim, itemsize):
    assert (
        split_plan(shape, dim, itemsize) is None
    ), f"{shape} dim={dim} should fall through to the generic implementation"


@pytest.mark.softmax
@pytest.mark.parametrize("itemsize", ITEMSIZES)
def test_cdna3_softmax_gate_row_budget(itemsize):
    """Pin the row budget from both sides, for both element widths.

    Written against the constants and the CU count rather than literals, so the
    assertions keep describing the intended rule on a part with a different CU
    count.
    """
    cap = row_budget(itemsize)
    assert cap >= 1
    # Long enough that only the row budget can be the deciding limit.
    n = 8 * ARCH_SOFTMAX._MIN_SPLIT_N
    assert ARCH_SOFTMAX._split_plan(cap, n, itemsize, 0) is not None
    assert ARCH_SOFTMAX._split_plan(cap + 1, n, itemsize, 0) is None


@pytest.mark.softmax
def test_cdna3_softmax_row_budget_follows_element_width():
    """A wider element has to get the tighter budget.

    The row counts the narrow budget reaches and the wide one does not are the ones
    cut into only four chunks per row, which is the configuration a wide element
    measured poorly at. Pin the ordering and a row count only the narrow budget has
    room for.
    """
    narrow_cap = row_budget(NARROW)
    wide_cap = row_budget(WIDE)
    assert narrow_cap > wide_cap, "narrow elements are the ones with room for rows"

    n = 8 * ARCH_SOFTMAX._MIN_SPLIT_N
    assert ARCH_SOFTMAX._split_plan(narrow_cap, n, NARROW, 0) is not None
    assert ARCH_SOFTMAX._split_plan(narrow_cap, n, WIDE, 0) is None


@pytest.mark.softmax
@pytest.mark.parametrize("itemsize", ITEMSIZES)
def test_cdna3_softmax_length_limit_is_element_width_independent(itemsize):
    """The length limit is the one limit that does not depend on element width.

    It is set by the host cost of the second launch, which the element width does
    not change, and all three dtypes measured the same break-even. Pinned so that a
    retune which makes the length width-dependent has to say so here.
    """
    assert (
        ARCH_SOFTMAX._split_plan(1, ARCH_SOFTMAX._MIN_SPLIT_N, itemsize, 0) is not None
    )
    assert (
        ARCH_SOFTMAX._split_plan(1, ARCH_SOFTMAX._MIN_SPLIT_N - 1, itemsize, 0) is None
    )


@pytest.mark.softmax
def test_cdna3_softmax_gate_needs_four_chunks_per_row():
    """The chunk minimum is what the row sweep actually measured.

    Every shape that got four or more chunks per row beat the generic kernel and
    every shape that got two lost, with no exception across both lengths and all
    three dtypes, so a shape must never be admitted with fewer than four.
    """
    cu = ARCH_SOFTMAX._cu_count(0)
    assert ARCH_SOFTMAX._MIN_BLOCKS_PER_ROW >= 4
    n = 8 * ARCH_SOFTMAX._MIN_SPLIT_N

    # On the narrow budget the chunk minimum is what binds first, so the largest
    # admitted row count is the one landing exactly on it.
    last = ARCH_SOFTMAX._BLOCKS_PER_CU * cu // ARCH_SOFTMAX._MIN_BLOCKS_PER_ROW
    assert last <= row_budget(NARROW), "the narrow budget must not outrun the chunks"
    assert (
        ARCH_SOFTMAX._split_plan(last, n, NARROW, 0)
        == ARCH_SOFTMAX._MIN_BLOCKS_PER_ROW
    )
    assert ARCH_SOFTMAX._split_plan(last + 1, n, NARROW, 0) is None

    # A reduction long enough for the row budget but too short to be cut into the
    # minimum number of whole tiles must be refused by the tile cap, not admitted.
    short = (ARCH_SOFTMAX._MIN_BLOCKS_PER_ROW - 1) * ARCH_SOFTMAX._TILE_N
    assert ARCH_SOFTMAX._split_plan(1, short, NARROW, 0) is None


@pytest.mark.softmax
@pytest.mark.parametrize("n", [65536, 131072, 262144, 1048576])
def test_cdna3_softmax_wide_budget_avoids_four_chunk_rows(n):
    """A wide element must never be admitted with only four chunks per row.

    A wide element measured poorly once its rows were cut into exactly four chunks.
    Raising the wide budget is therefore not a free knob: it is what keeps wide rows
    off that band, so this pins the property over every row count the gate takes
    rather than just the constant behind it.
    """
    for m in range(1, row_budget(WIDE) + 1):
        plan = ARCH_SOFTMAX._split_plan(m, n, WIDE, 0)
        if plan is None:
            continue
        assert (
            plan > ARCH_SOFTMAX._MIN_BLOCKS_PER_ROW
        ), f"M={m} N={n} wide is admitted with only {plan} chunks per row"

    # And that the budget is placed exactly at that edge rather than short of it:
    # the first row count it refuses is the first one that would get four chunks.
    beyond = row_budget(WIDE) + 1
    chunks_beyond = ARCH_SOFTMAX._prev_power_of_2(
        max(1, ARCH_SOFTMAX._BLOCKS_PER_CU * ARCH_SOFTMAX._cu_count(0) // beyond)
    )
    assert chunks_beyond == ARCH_SOFTMAX._MIN_BLOCKS_PER_ROW


@pytest.mark.softmax
@pytest.mark.parametrize("itemsize", ITEMSIZES)
@pytest.mark.parametrize("n", [32768, 65536, 262144, 1048576])
def test_cdna3_softmax_never_exceeds_two_blocks_per_cu(n, itemsize):
    """No admitted shape may reach four workgroups per CU.

    Grids at or below two workgroups per CU are the ones that stayed well behaved
    throughout the sweep. Raising _BLOCKS_PER_CU is therefore not a free knob, so
    this asserts the bound over every row count the gate takes rather than just the
    constant.
    """
    cu = ARCH_SOFTMAX._cu_count(0)
    assert ARCH_SOFTMAX._BLOCKS_PER_CU <= 2
    for m in range(1, row_budget(itemsize) + 1):
        plan = ARCH_SOFTMAX._split_plan(m, n, itemsize, 0)
        if plan is None:
            continue
        assert m * plan <= 2 * cu, f"M={m} N={n} asks for {m * plan} workgroups"


@pytest.mark.softmax
@pytest.mark.parametrize("shape, dim", SPLIT_SHAPES + GENERIC_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cdna3_softmax(shape, dim, dtype):
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
def test_cdna3_softmax_out(shape, dim, dtype):
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
def test_cdna3_softmax_half_to_float(dtype):
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
def test_cdna3_softmax_neg_inf(dtype):
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

        assert split_plan(shape, dim, itemsize_of(dtype)) is not None, name
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, reduce_dim=n)


@pytest.mark.softmax
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cdna3_softmax_at_row_cap(dtype):
    """Cover grid.y at the row cap: every row combines its own set of partials.

    The shapes in SPLIT_SHAPES only reach M=4, so without this the widest grid the
    split path ever runs with is a handful of rows.
    """
    m = row_budget(itemsize_of(dtype))
    n = 2 * ARCH_SOFTMAX._MIN_SPLIT_N
    if ARCH_SOFTMAX._split_plan(m, n, itemsize_of(dtype), 0) is None:
        cu = ARCH_SOFTMAX._cu_count(0)
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
def test_cdna3_softmax_non_contiguous():
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
