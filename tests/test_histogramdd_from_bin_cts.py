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

# torch._histogramdd_from_bin_cts is only implemented for floating types on the
# ATen reference (CPU). float16/bfloat16/int raise NotImplementedError, so the
# dtype sweep is restricted to float32 (and float64 where the device supports it).
if utils.QUICK_MODE:
    HIST_DTYPES = [torch.float32]
else:
    # CPU ATen reference only supports float types; fp64 included where the device supports it.
    HIST_DTYPES = [torch.float32, torch.float64]

# (points, dims) configurations to exercise: 1D, 2D, 3D and >2 leading dims.
HIST_SHAPES = [
    (64, 1),
    (1000, 1),
    (64, 2),
    (1024, 2),
    (100, 3),
    (8, 5, 2),  # ndim > 2 -> leading dims are flattened
]

# Per-shape bin configurations. ``bins`` is a list with one int per dimension.
HIST_BINS = [
    [4],
    [10, 10],
    [5, 5, 5],
]


def _make_input(shape, dtype, device):
    """Generate deterministic points in a known range so both CPU and CUDA paths
    see the same data and bin boundaries are exercised."""
    torch.manual_seed(0)
    # Values roughly in [-3, 3) for each coordinate.
    inp = torch.randn(shape, dtype=dtype, device=device) * 3
    return inp


def _bins_for(shape):
    """Pick a bin configuration whose length matches the last dim of ``shape``."""
    ndim = shape[-1]
    for bins in HIST_BINS:
        if len(bins) == ndim:
            return bins
    # Fallback: one bin per dimension.
    return [5] * ndim


def _range_for(ndim):
    """A per-dimension explicit range covering the data with room to spare."""
    return [-3.0, 3.0] * ndim


def _to_cpu_ref(inp):
    """Move ``inp`` to CPU for the reference call.

    ``torch._histogramdd_from_bin_cts`` is only implemented for the CPU backend
    in PyTorch, so the reference must always run on CPU even when TO_CPU is
    False (mirroring ``tests/test_histogramdd_bin_edges.py``).
    """
    return (
        utils.to_reference(inp).to("cpu")
        if not utils.TO_CPU
        else utils.to_reference(inp)
    )


def _assert_close(res, ref, dtype, equal_nan=False):
    """Compare a CUDA result against a CPU reference.

    The reference lives on CPU (the op is CPU-only in PyTorch); move the result
    onto the reference device so ``gems_assert_close`` sees matching devices.
    """
    if res.device.type != ref.device.type:
        res = res.to(ref.device)
    utils.gems_assert_close(res, ref, dtype, equal_nan=equal_nan)


# ---------------------------------------------------------------------------
# Base variant: aten::_histogramdd_from_bin_cts
# ---------------------------------------------------------------------------


@pytest.mark.histogramdd_from_bin_cts
@pytest.mark.parametrize("shape", HIST_SHAPES)
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_from_bin_cts_basic(shape, dtype):
    bins = _bins_for(shape)
    inp = _make_input(shape, dtype, flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    ref_out = torch._histogramdd_from_bin_cts(ref_inp, bins)
    res_out = flag_gems._histogramdd_from_bin_cts(inp, bins)
    _assert_close(res_out, ref_out, dtype)


@pytest.mark.histogramdd_from_bin_cts
@pytest.mark.parametrize("shape", HIST_SHAPES)
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_from_bin_cts_with_range(shape, dtype):
    bins = _bins_for(shape)
    ndim = shape[-1]
    inp = _make_input(shape, dtype, flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    ref_out = torch._histogramdd_from_bin_cts(ref_inp, bins, range=_range_for(ndim))
    res_out = flag_gems._histogramdd_from_bin_cts(inp, bins, range=_range_for(ndim))
    _assert_close(res_out, ref_out, dtype)


@pytest.mark.histogramdd_from_bin_cts
@pytest.mark.parametrize("shape", HIST_SHAPES)
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_from_bin_cts_with_weight(shape, dtype):
    bins = _bins_for(shape)
    ndim = shape[-1]
    # weight shape matches input shape excluding the innermost dimension.
    weight_shape = tuple(shape[:-1])
    inp = _make_input(shape, dtype, flag_gems.device)
    weight = torch.rand(weight_shape, dtype=dtype, device=flag_gems.device) * 5
    ref_inp = _to_cpu_ref(inp)
    ref_weight = _to_cpu_ref(weight)

    ref_out = torch._histogramdd_from_bin_cts(
        ref_inp, bins, range=_range_for(ndim), weight=ref_weight
    )
    res_out = flag_gems._histogramdd_from_bin_cts(
        inp, bins, range=_range_for(ndim), weight=weight
    )
    _assert_close(res_out, ref_out, dtype)


@pytest.mark.histogramdd_from_bin_cts
@pytest.mark.parametrize("shape", HIST_SHAPES)
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_from_bin_cts_density(shape, dtype):
    bins = _bins_for(shape)
    ndim = shape[-1]
    inp = _make_input(shape, dtype, flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    ref_out = torch._histogramdd_from_bin_cts(
        ref_inp, bins, range=_range_for(ndim), density=True
    )
    res_out = flag_gems._histogramdd_from_bin_cts(
        inp, bins, range=_range_for(ndim), density=True
    )
    _assert_close(res_out, ref_out, dtype)


@pytest.mark.histogramdd_from_bin_cts
def test_histogramdd_from_bin_cts_boundary_and_outliers():
    """Exact bin-boundary placement and out-of-range / NaN / inf exclusion."""
    inp = torch.tensor(
        [
            [0.0],
            [0.5],
            [1.0],
            [1.5],
            [2.0],
            [-1.0],
            [3.0],
            [float("nan")],
            [float("inf")],
            [float("-inf")],
        ],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref_inp = _to_cpu_ref(inp)

    # 4 equal-width bins over [0, 2]; rightmost bin includes the right edge.
    # NaN/inf inputs are excluded from every bin, so the counts are finite;
    # equal_nan=True is harmless here (no NaN lands in the count tensor).
    ref_out = torch._histogramdd_from_bin_cts(ref_inp, [4], range=[0.0, 2.0])
    res_out = flag_gems._histogramdd_from_bin_cts(inp, [4], range=[0.0, 2.0])
    _assert_close(res_out, ref_out, torch.float32, equal_nan=True)


@pytest.mark.histogramdd_from_bin_cts
def test_histogramdd_from_bin_cts_auto_range_constant_dim():
    """A dimension whose min == max is expanded to (min - 0.5, max + 0.5)."""
    inp = torch.tensor(
        [[0.0, 5.0], [1.0, 5.0], [2.0, 5.0]],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref_inp = _to_cpu_ref(inp)

    ref_out = torch._histogramdd_from_bin_cts(ref_inp, [3, 3])
    res_out = flag_gems._histogramdd_from_bin_cts(inp, [3, 3])
    _assert_close(res_out, ref_out, torch.float32)


@pytest.mark.histogramdd_from_bin_cts
def test_histogramdd_from_bin_cts_empty_input():
    inp = torch.empty(0, 2, dtype=torch.float32, device=flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    ref_out = torch._histogramdd_from_bin_cts(
        ref_inp, [3, 3], range=[0.0, 2.0, 0.0, 2.0]
    )
    res_out = flag_gems._histogramdd_from_bin_cts(
        inp, [3, 3], range=[0.0, 2.0, 0.0, 2.0]
    )
    _assert_close(res_out, ref_out, torch.float32)


@pytest.mark.histogramdd_from_bin_cts
@pytest.mark.parametrize(
    "range_",
    [
        [1.5, -0.5, 0.0, 1.0],  # dimension 0 inverted
        [0.0, 1.0, 1.5, -0.5],  # dimension 1 inverted
        [0.0, 1.0, 1.5, -0.5, 0.0, 1.0],  # inverted in a 3-D range
    ],
)
def test_histogramdd_from_bin_cts_rejects_inverted_range(range_):
    """An inverted explicit range must raise, not return an all-zero histogram.

    ATen validates ``left <= right`` per dimension and reports
    'min should not exceed max, but got min <lo> max <hi> for dimension <d>'.
    """
    ndim = len(range_) // 2
    inp = torch.randn(8, ndim, dtype=torch.float32, device=flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    with pytest.raises(RuntimeError) as ref_err:
        torch._histogramdd_from_bin_cts(ref_inp, [3] * ndim, range=range_)

    with pytest.raises(RuntimeError) as res_err:
        flag_gems._histogramdd_from_bin_cts(inp, [3] * ndim, range=range_)

    assert str(res_err.value) == str(ref_err.value)
    assert "min should not exceed max" in str(res_err.value)


@pytest.mark.histogramdd_from_bin_cts
def test_histogramdd_from_bin_cts_range_bound_formatting():
    """Bounds are printed with ATen's ``%g`` formatting (six significant digits)."""
    inp = torch.randn(8, 2, dtype=torch.float32, device=flag_gems.device)
    ref_inp = _to_cpu_ref(inp)
    range_ = [0.123456789, -0.5, 0.0, 1.0]

    with pytest.raises(RuntimeError) as ref_err:
        torch._histogramdd_from_bin_cts(ref_inp, [3, 3], range=range_)
    with pytest.raises(RuntimeError) as res_err:
        flag_gems._histogramdd_from_bin_cts(inp, [3, 3], range=range_)

    assert "min 0.123457 max -0.5" in str(res_err.value)
    assert str(res_err.value) == str(ref_err.value)


@pytest.mark.histogramdd_from_bin_cts
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("row", [0, 150, 299])
def test_histogramdd_from_bin_cts_auto_range_must_be_finite(bad, row):
    """A non-finite auto-range must raise like ATen.

    With ``range=None`` the range comes from the data's per-dimension min/max,
    so a NaN or +-Inf anywhere makes it non-finite. ATen raises
    "dimension <d>'s range [<lo>, <hi>] is not finite" rather than building a
    histogram from a garbage range. Reaching the failing row also exercises the
    cross-tile NaN propagation in the range kernel (BLOCK_M is 128 rows).
    """
    inp = torch.randn(300, 3, dtype=torch.float32, device=flag_gems.device)
    inp[row, 1] = bad
    ref_inp = _to_cpu_ref(inp)

    with pytest.raises(RuntimeError) as ref_err:
        torch._histogramdd_from_bin_cts(ref_inp, [3, 3, 3])

    with pytest.raises(RuntimeError) as res_err:
        flag_gems._histogramdd_from_bin_cts(inp, [3, 3, 3])

    assert str(res_err.value) == str(ref_err.value)
    assert "is not finite" in str(res_err.value)


@pytest.mark.histogramdd_from_bin_cts
def test_histogramdd_from_bin_cts_finite_auto_range_unchanged():
    """The finiteness check must not reject ordinary finite data."""
    inp = torch.randn(300, 3, dtype=torch.float32, device=flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    ref_out = torch._histogramdd_from_bin_cts(ref_inp, [3, 3, 3])
    res_out = flag_gems._histogramdd_from_bin_cts(inp, [3, 3, 3])
    _assert_close(res_out, ref_out, torch.float32)


@pytest.mark.histogramdd_from_bin_cts
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("col", [0, 1, 2])
def test_histogramdd_from_bin_cts_auto_range_reports_lowest_dimension(bad, col):
    """The reported dimension is the lowest offending one, not just any of them.

    The non-finite flag is written by the range kernel (a per-column reduction
    plus an atomic min), so a regression that reported the wrong column -- the
    last one, or a tile-local one -- would still raise, but with the wrong
    dimension in the message. Poisoning two columns at once pins the ordering.
    """
    inp = torch.randn(64, 3, dtype=torch.float32, device=flag_gems.device)
    for c in range(col, 3):
        inp[0, c] = bad
    ref_inp = _to_cpu_ref(inp)

    with pytest.raises(RuntimeError) as ref_err:
        torch._histogramdd_from_bin_cts(ref_inp, [3, 3, 3])
    with pytest.raises(RuntimeError) as res_err:
        flag_gems._histogramdd_from_bin_cts(inp, [3, 3, 3])

    assert f"dimension {col}'s range" in str(res_err.value)
    assert str(res_err.value) == str(ref_err.value)


@pytest.mark.histogramdd_from_bin_cts
@pytest.mark.parametrize("nbins", [1, 2, 3, 4, 5, 7, 10])
def test_histogramdd_from_bin_cts_zero_width_explicit_range(nbins):
    """An explicit zero-width range is widened, not folded into the last bin.

    ATen expands a degenerate range before building the edges ("Expand empty
    range to match numpy behavior and avoid division by 0 in normalization"), so
    ``range=[1.0, 1.0]`` behaves like the degenerate auto-range case: the edges
    become ``linspace(0.5, 1.5, nbins + 1)`` and a point at exactly 1.0 lands in
    the middle bin (``nbins // 2``) rather than the last one. The neighbouring
    point at 5.0 is outside the widened range and must stay uncounted.
    """
    pts = torch.tensor(
        [[1.0], [1.0], [5.0]], dtype=torch.float32, device=flag_gems.device
    )
    ref_inp = _to_cpu_ref(pts)

    ref_out = torch._histogramdd_from_bin_cts(ref_inp, [nbins], range=[1.0, 1.0])
    res_out = flag_gems._histogramdd_from_bin_cts(pts, [nbins], range=[1.0, 1.0])

    # Counts are integers, so this must agree exactly, not approximately.
    utils.gems_assert_equal(res_out.to(ref_out.device), ref_out)
    assert res_out.sum().item() == 2.0, "the out-of-range point must stay uncounted"
    assert res_out[nbins // 2].item() == 2.0, "the points at 1.0 share the middle bin"


@pytest.mark.histogramdd_from_bin_cts
@pytest.mark.parametrize(
    "range_,dim",
    [
        ([float("inf"), float("inf")], 0),
        ([float("-inf"), float("inf")], 0),
        ([float("nan"), 1.0], 0),
        ([0.0, 1.0, float("inf"), 1.0], 1),
        ([0.0, 1.0, 1.0, float("-inf")], 1),
    ],
)
def test_histogramdd_from_bin_cts_rejects_non_finite_explicit_range(range_, dim):
    """An explicit non-finite bound must raise, like the auto-range case.

    ATen applies the same finiteness check to a caller-supplied ``range``, and it
    runs *before* the ``min <= max`` check: ``[inf, -inf]`` reports "is not
    finite" rather than "min should not exceed max".
    """
    ndim = len(range_) // 2
    inp = torch.randn(8, ndim, dtype=torch.float32, device=flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    with pytest.raises(RuntimeError) as ref_err:
        torch._histogramdd_from_bin_cts(ref_inp, [3] * ndim, range=range_)
    with pytest.raises(RuntimeError) as res_err:
        flag_gems._histogramdd_from_bin_cts(inp, [3] * ndim, range=range_)

    assert str(res_err.value) == str(ref_err.value)
    assert f"dimension {dim}'s range" in str(res_err.value)
    assert "is not finite" in str(res_err.value)


@pytest.mark.histogramdd_from_bin_cts
def test_histogramdd_from_bin_cts_finite_check_precedes_min_max():
    """The finiteness error wins over the inverted-range error, as in ATen."""
    inp = torch.randn(8, dtype=torch.float32, device=flag_gems.device).reshape(-1, 1)
    ref_inp = _to_cpu_ref(inp)
    range_ = [float("inf"), float("-inf")]

    with pytest.raises(RuntimeError) as ref_err:
        torch._histogramdd_from_bin_cts(ref_inp, [3], range=range_)
    with pytest.raises(RuntimeError) as res_err:
        flag_gems._histogramdd_from_bin_cts(inp, [3], range=range_)

    assert "is not finite" in str(res_err.value)
    assert "min should not exceed max" not in str(res_err.value)
    assert str(res_err.value) == str(ref_err.value)


@pytest.mark.histogramdd_from_bin_cts
@pytest.mark.parametrize("nbins", [4, 5, 7])
def test_histogramdd_from_bin_cts_zero_width_explicit_range_2d(nbins):
    """One degenerate dimension must not disturb the other dimension's binning."""
    pts = torch.tensor(
        [[1.0, 0.0], [1.0, 0.5], [1.0, 0.5], [1.0, 1.0]],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref_inp = _to_cpu_ref(pts)
    range_ = [1.0, 1.0, 0.0, 1.0]

    ref_out = torch._histogramdd_from_bin_cts(ref_inp, [nbins, 4], range=range_)
    res_out = flag_gems._histogramdd_from_bin_cts(pts, [nbins, 4], range=range_)
    utils.gems_assert_equal(res_out.to(ref_out.device), ref_out)


@pytest.mark.histogramdd_from_bin_cts
def test_histogramdd_from_bin_cts_zero_width_edges_match_linspace():
    """The widened edges themselves must match ATen's ``linspace(lo-.5, hi+.5)``.

    Placement is by exact comparison against the edges, so a point on the widened
    boundaries is the sensitive probe: 0.5 and 1.5 are *inside* ATen's range for
    ``range=[1.0, 1.0]`` even though they are outside the caller's nominal range.
    """
    nbins = 8
    edges = torch.linspace(0.5, 1.5, nbins + 1, dtype=torch.float32)
    probes = edges.tolist() + [edges[0].item() - 0.5, edges[-1].item() + 0.5]
    pts = torch.tensor(probes, dtype=torch.float32, device=flag_gems.device).reshape(
        -1, 1
    )
    ref_inp = _to_cpu_ref(pts)

    ref_out = torch._histogramdd_from_bin_cts(ref_inp, [nbins], range=[1.0, 1.0])
    res_out = flag_gems._histogramdd_from_bin_cts(pts, [nbins], range=[1.0, 1.0])
    utils.gems_assert_equal(res_out.to(ref_out.device), ref_out)


# ---------------------------------------------------------------------------
# Out variant: aten::_histogramdd_from_bin_cts.out
# ---------------------------------------------------------------------------


@pytest.mark.histogramdd_from_bin_cts_out
@pytest.mark.parametrize("shape", HIST_SHAPES)
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_from_bin_cts_out(shape, dtype):
    bins = _bins_for(shape)
    ndim = shape[-1]
    rng = _range_for(ndim)

    inp = _make_input(shape, dtype, flag_gems.device)
    ref_inp = _to_cpu_ref(inp)

    # Reference output (on CPU).
    ref_out = torch._histogramdd_from_bin_cts(ref_inp, bins, range=rng)

    out_shape = tuple(bins)
    out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems._histogramdd_from_bin_cts_out(inp, bins, range=rng, out=out)

    # The .out variant returns the out tensor and writes into it in-place.
    _assert_close(res_out, ref_out, dtype)
    _assert_close(out, ref_out, dtype)


@pytest.mark.histogramdd_from_bin_cts
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_from_bin_cts_out_weighted_density(dtype):
    # Exercise the .out path with weight + density: the returned tensor and the
    # in-place ``out`` buffer must both match the CPU reference. Grouped under
    # the base mark since the strict per-function mark check keys the expected
    # mark off the trailing test-name token.
    inp = _make_input((100, 2), dtype, flag_gems.device)
    weight = torch.rand(100, dtype=dtype, device=flag_gems.device) * 4
    rng = _range_for(2)
    ref_inp = _to_cpu_ref(inp)
    ref_weight = _to_cpu_ref(weight)

    ref_out = torch._histogramdd_from_bin_cts(
        ref_inp, [10, 10], range=rng, weight=ref_weight, density=True
    )

    out = torch.empty((10, 10), dtype=dtype, device=flag_gems.device)
    res_out = flag_gems._histogramdd_from_bin_cts_out(
        inp, [10, 10], range=rng, weight=weight, density=True, out=out
    )
    _assert_close(res_out, ref_out, dtype)
    _assert_close(out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Precision, density edge cases and the full out= contract
# ---------------------------------------------------------------------------


@pytest.mark.histogramdd_from_bin_cts
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_from_bin_cts_bin_boundary_precision(dtype):
    """Points sitting exactly on a bin edge, and one ulp either side of it.

    Bin placement uses an exact comparison against the edges, so a 1-ulp error in
    an edge moves the point a whole bin. FP64 is the sensitive case: narrowing the
    coordinates or the edges to fp32 puts e.g. 0.2999999999999999 over [0, 1] with
    10 bins into bin 3 instead of bin 2.
    """
    nbins = 10
    rng = [0.0, 1.0]
    edges = torch.linspace(rng[0], rng[1], nbins + 1, dtype=dtype)

    # Every edge, plus its two neighbours in the floating point ordering.
    probes = []
    for e in edges.tolist():
        t = torch.tensor(e, dtype=dtype)
        probes += [
            e,
            torch.nextafter(t, torch.tensor(-1.0, dtype=dtype)).item(),
            torch.nextafter(t, torch.tensor(2.0, dtype=dtype)).item(),
        ]
    inp = torch.tensor(probes, dtype=dtype, device=flag_gems.device).reshape(-1, 1)
    ref_inp = _to_cpu_ref(inp)

    ref_out = torch._histogramdd_from_bin_cts(ref_inp, [nbins], range=rng)
    res_out = flag_gems._histogramdd_from_bin_cts(inp, [nbins], range=rng)
    # Counts are integers, so this must agree exactly, not approximately.
    utils.gems_assert_equal(res_out.to(ref_out.device), ref_out)


@pytest.mark.histogramdd_from_bin_cts
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_from_bin_cts_density_zero_total(dtype):
    """density with a zero total count must not be forced to zero.

    ATen divides by the total count directly, so an empty or fully out-of-range
    input yields NaN rather than a zero-filled histogram.
    """
    rng = [0.0, 1.0, 0.0, 1.0]

    # All points outside the explicit range.
    inp = torch.tensor([[5.0, 5.0], [6.0, 6.0]], dtype=dtype, device=flag_gems.device)
    ref_out = torch._histogramdd_from_bin_cts(
        _to_cpu_ref(inp), [2, 2], range=rng, density=True
    )
    res_out = flag_gems._histogramdd_from_bin_cts(inp, [2, 2], range=rng, density=True)
    assert torch.isnan(ref_out).all(), "reference should produce NaN here"
    _assert_close(res_out, ref_out, dtype, equal_nan=True)

    # Empty input: same reasoning, zero points and therefore a zero total.
    empty = torch.zeros(0, 2, dtype=dtype, device=flag_gems.device)
    ref_empty = torch._histogramdd_from_bin_cts(
        _to_cpu_ref(empty), [2, 2], range=rng, density=True
    )
    res_empty = flag_gems._histogramdd_from_bin_cts(
        empty, [2, 2], range=rng, density=True
    )
    _assert_close(res_empty, ref_empty, dtype, equal_nan=True)


@pytest.mark.histogramdd_from_bin_cts
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_from_bin_cts_density_cancelling_weights(dtype):
    """Weights that cancel to zero give infinities, with the signs preserved."""
    rng = [0.0, 1.0, 0.0, 1.0]
    inp = torch.tensor(
        [[0.25, 0.25], [0.75, 0.75]], dtype=dtype, device=flag_gems.device
    )
    weight = torch.tensor([1.0, -1.0], dtype=dtype, device=flag_gems.device)

    ref_out = torch._histogramdd_from_bin_cts(
        _to_cpu_ref(inp),
        [2, 2],
        range=rng,
        weight=_to_cpu_ref(weight),
        density=True,
    )
    res_out = flag_gems._histogramdd_from_bin_cts(
        inp, [2, 2], range=rng, weight=weight, density=True
    )
    assert torch.isinf(ref_out).any(), "reference should produce infinities here"
    _assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.histogramdd_from_bin_cts_out
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_from_bin_cts_out_resize(dtype):
    """A mis-shaped out is resized in place and the caller's object is returned."""
    inp = _make_input((256, 2), dtype, flag_gems.device)
    bins = [3, 3]
    rng = _range_for(2)
    ref_out = torch._histogramdd_from_bin_cts(_to_cpu_ref(inp), bins, range=rng)

    for start_shape in [(0,), (5, 7)]:
        out = torch.empty(start_shape, dtype=dtype, device=flag_gems.device)
        res = flag_gems._histogramdd_from_bin_cts_out(inp, bins, range=rng, out=out)
        assert res is out, "the out tensor itself must be returned"
        assert tuple(out.shape) == tuple(bins)
        _assert_close(out, ref_out, dtype)


@pytest.mark.histogramdd_from_bin_cts_out
def test_histogramdd_from_bin_cts_out_dtype_mismatch():
    """A dtype mismatch raises rather than being silently cast or re-bound."""
    inp = _make_input((64, 2), torch.float32, flag_gems.device)
    out = torch.empty((3, 3), dtype=torch.float64, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="dtype"):
        flag_gems._histogramdd_from_bin_cts_out(
            inp, [3, 3], range=_range_for(2), out=out
        )


@pytest.mark.histogramdd_from_bin_cts_out
@pytest.mark.parametrize("dtype", HIST_DTYPES)
def test_histogramdd_from_bin_cts_out_non_contiguous(dtype):
    """A non-contiguous out of the right shape is accepted, as in ATen."""
    inp = _make_input((256, 2), dtype, flag_gems.device)
    bins = [3, 3]
    rng = _range_for(2)
    ref_out = torch._histogramdd_from_bin_cts(_to_cpu_ref(inp), bins, range=rng)

    base = torch.empty((3, 6), dtype=dtype, device=flag_gems.device)
    out = base[:, ::2]
    assert not out.is_contiguous()
    res = flag_gems._histogramdd_from_bin_cts_out(inp, bins, range=rng, out=out)
    assert res is out
    _assert_close(out, ref_out, dtype)
