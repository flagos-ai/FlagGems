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
