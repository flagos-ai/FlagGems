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

"""Correctness tests for ``aten::_histogramdd_bin_edges``.

PyTorch registers a CPU kernel for this operator only (there is no device
kernel), so the reference and the candidate both receive CPU tensors -- the
operator's real contract. Inputs still come from the shared value-range helper
in ``tests/test_utils.py`` and are only relocated to CPU afterwards.

The operator returns one 1-D tensor of bin edges per histogram dimension
(``len(bins)`` entries, each ``bins[d] + 1`` long); every component of that
list is compared against the native result.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

# Probed native support: float32/float64 only. Every other dtype raises
# RuntimeError('histogramdd' not implemented for ...), so the required dtype
# list collapses to these two and the remaining ones become negative rows.
SUPPORTED_DTYPES = [torch.float32, torch.float64]
UNSUPPORTED_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.bool,
]

# The operator requires rank >= 2, so the spec's 0-D/1-D shapes are invalid
# inputs (negative rows below). The grid keeps the remaining spec shapes and
# adds rank-2 boundaries: empty dataset, zero-width innermost dimension
# (bins == []), a single histogram dimension and small odd sizes.
_SPEC_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 2]
_EXTRA_SHAPES = [
    (2, 19, 7),
    (0, 2),
    (4, 0),
    (1, 4),
    (8, 2),
    (5, 1),
    (256, 3),
    (3, 7),
    (64, 5),
]
GRID_SHAPES = _SPEC_SHAPES + _EXTRA_SHAPES

# Parameter-coverage workload: a spec shape whose innermost dimension (D = 15)
# keeps len(bins) == D and len(range) == 2 * D expressible.
_PARAM_SHAPE = (20, 320, 15)
_PARAM_D = _PARAM_SHAPE[-1]
_PARAM_BINS = [4] * _PARAM_D

_EDGE_LEN = 5
_OUT_SENTINEL = 1234.5  # outside any edge produced from data in [-1, 1]


def _cpu_input(dtype, shape, value_range):
    """Value-range input handed to both paths; the native kernel is CPU-only."""
    return tu.make_input(dtype, shape, value_range).cpu()


def _bins(shape):
    return [4] * shape[-1]


def _assert_edges(res, ref):
    """Compare the returned edge list dimension by dimension."""
    assert isinstance(res, list)
    assert len(res) == len(ref)
    for res_edge, ref_edge in zip(res, ref):
        tu.assert_result_close(res_edge, ref_edge)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", GRID_SHAPES)
def test__histogramdd_bin_edges(shape, value_range, dtype):
    bins = _bins(shape)
    inp = _cpu_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._histogramdd_bin_edges(ref_inp, bins)
    res_out = flag_gems._histogramdd_bin_edges(inp, bins)

    _assert_edges(res_out, ref_out)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("per_dim", [0, 1, 4, 9])
def test__histogramdd_bin_edges_bin_counts(dtype, per_dim):
    # `bins` is an int[]: zero, the smallest positive count and a larger count.
    # Negative counts are invalid and covered by the negative suite below.
    bins = [per_dim] * _PARAM_D
    inp = _cpu_input(dtype, _PARAM_SHAPE, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._histogramdd_bin_edges(ref_inp, bins)
    res_out = flag_gems._histogramdd_bin_edges(inp, bins)

    _assert_edges(res_out, ref_out)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__histogramdd_bin_edges_bins_per_dimension(dtype):
    # Per-dimension counts in the tuple form of the int[] argument.
    bins = tuple(range(1, _PARAM_D + 1))
    inp = _cpu_input(dtype, _PARAM_SHAPE, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._histogramdd_bin_edges(ref_inp, bins)
    res_out = flag_gems._histogramdd_bin_edges(inp, bins)

    _assert_edges(res_out, ref_out)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__histogramdd_bin_edges_range_default(dtype):
    # Schema default: omitting `range` derives the extent from the data.
    inp = _cpu_input(dtype, _PARAM_SHAPE, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._histogramdd_bin_edges(ref_inp, _PARAM_BINS)
    res_out = flag_gems._histogramdd_bin_edges(inp, _PARAM_BINS)

    _assert_edges(res_out, ref_out)


# `range` is a flat float[] of 2 * D bounds; the rows cover negative, zero and
# positive bounds. The boundary row uses half-bounds because a full
# [dtype_min, dtype_max] span overflows the dtype and natively yields NaN edges.
_RANGE_ROWS = [
    ("-1", "1"),
    ("0", "1"),
    ("-1", "0"),
    ("0", "0"),
    ("min/2", "max/2"),
]


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("low,high", _RANGE_ROWS)
def test__histogramdd_bin_edges_range_values(dtype, low, high):
    op_range = [tu.resolve_bound(low, dtype), tu.resolve_bound(high, dtype)] * _PARAM_D
    inp = _cpu_input(dtype, _PARAM_SHAPE, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._histogramdd_bin_edges(
        ref_inp, _PARAM_BINS, range=op_range
    )
    res_out = flag_gems._histogramdd_bin_edges(inp, _PARAM_BINS, range=op_range)

    _assert_edges(res_out, ref_out)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("density", [False, True])
@pytest.mark.parametrize("with_weight", [False, True])
def test__histogramdd_bin_edges_weight_density(with_weight, density):
    # Edges do not depend on `weight`/`density` (probed); the rows check that
    # the candidate accepts both optional arguments.
    inp = _cpu_input(torch.float32, _PARAM_SHAPE, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    weight = torch.ones(_PARAM_SHAPE[0], dtype=torch.float32) if with_weight else None
    ref_weight = weight.clone() if with_weight else None

    ref_out = torch.ops.aten._histogramdd_bin_edges(
        ref_inp, _PARAM_BINS, weight=ref_weight, density=density
    )
    res_out = flag_gems._histogramdd_bin_edges(
        inp, _PARAM_BINS, weight=weight, density=density
    )

    _assert_edges(res_out, ref_out)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__histogramdd_bin_edges_out(container, dtype):
    # The `.out` overload is callable natively: it returns None and fills the
    # caller's Tensor[] buffers in place.
    inp = _cpu_input(dtype, _PARAM_SHAPE, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    res_buffers = [
        torch.full((_EDGE_LEN,), _OUT_SENTINEL, dtype=dtype) for _ in range(_PARAM_D)
    ]
    ref_buffers = [
        torch.full((_EDGE_LEN,), _OUT_SENTINEL, dtype=dtype) for _ in range(_PARAM_D)
    ]

    torch.ops.aten._histogramdd_bin_edges.out(
        ref_inp, _PARAM_BINS, out=container(ref_buffers)
    )
    res_out = flag_gems._histogramdd_bin_edges(
        inp, _PARAM_BINS, out=container(res_buffers)
    )

    assert res_out is None
    assert [buf.dtype for buf in res_buffers] == [dtype] * _PARAM_D
    _assert_edges(res_buffers, ref_buffers)
    assert torch.equal(inp, ref_inp)  # only the out buffers are written


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__histogramdd_bin_edges_strided_input(dtype):
    # Columns selected with stride 2 from a wider row plus a non-zero storage
    # offset: both paths read the same (non-contiguous) view.
    base = _cpu_input(dtype, (16, 8), ["-1", "1"])
    inp = base[:, 1:7:2]
    ref_inp = tu.to_reference(inp)
    bins = [4, 4, 4]

    ref_out = torch.ops.aten._histogramdd_bin_edges(ref_inp, bins)
    res_out = flag_gems._histogramdd_bin_edges(inp, bins)

    _assert_edges(res_out, ref_out)


_SPECIAL_ROWS = tu.selected_cases(tu.special_value_cases(SUPPORTED_DTYPES), quick=[])


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_ROWS)
def test__histogramdd_bin_edges_special_values(dtype, scenario):
    # Non-finite samples need an explicit finite `range`: the data-derived
    # extent of a NaN/Inf input is rejected as non-finite. Edges come from the
    # range, so the expected result is finite for every scenario.
    inp = tu.make_special_input(dtype, scenario).cpu().reshape(-1, 1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._histogramdd_bin_edges(ref_inp, [3], range=[-1.0, 1.0])
    res_out = flag_gems._histogramdd_bin_edges(inp, [3], range=[-1.0, 1.0])

    _assert_edges(res_out, ref_out)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES)
def test__histogramdd_bin_edges_unsupported_dtype(dtype):
    inp = _cpu_input(torch.float32, (8, 3), ["-1", "1"]).to(dtype)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._histogramdd_bin_edges(inp, [4, 4, 4])


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("shape", [(), (1,), (256,)])
def test__histogramdd_bin_edges_invalid_rank(shape):
    # rank < 2 is rejected: 'input tensor should have at least 2 dimensions'.
    inp = _cpu_input(torch.float32, shape, ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._histogramdd_bin_edges(inp, [4] * len(shape))


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("bins", [[1, 1], [], [-3, -3, -3]])
def test__histogramdd_bin_edges_invalid_bins(bins):
    # len(bins) must equal the innermost dimension (D = 3 here); negative
    # counts are rejected as well.
    inp = _cpu_input(torch.float32, (32, 3), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._histogramdd_bin_edges(inp, bins)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize(
    "op_range",
    [
        [-1.0, 1.0],
        [-1.0, 1.0, -1.0, 1.0, -1.0, float("inf")],
        [-1.0, 1.0, -1.0, 1.0, -1.0, float("nan")],
    ],
)
def test__histogramdd_bin_edges_invalid_range(op_range):
    # A flat range of 2 * D finite bounds is required: wrong length and
    # non-finite bounds are both rejected natively.
    inp = _cpu_input(torch.float32, (32, 3), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._histogramdd_bin_edges(inp, [4, 4, 4], range=op_range)


@pytest.mark.histogramdd_bin_edges
@pytest.mark.parametrize("scenario", ["nan", "inf"])
def test__histogramdd_bin_edges_non_finite_data_without_range(scenario):
    # Without an explicit range the derived data extent is not finite.
    inp = tu.make_special_input(torch.float32, scenario).cpu().reshape(-1, 1)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._histogramdd_bin_edges(inp, [3])
