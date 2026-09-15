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

# The aten reference for ``_histogramdd_from_bin_tensors`` has no native CUDA
# implementation in this PyTorch build, so the reference histogram is always
# computed on CPU (the inputs are moved to CPU, the reference is run, and
# ``gems_assert_close`` moves the GEMS result back to CPU for comparison).
HISTDD_SHAPES = (
    [(100, 1), (1000, 2), (2000, 2), (2000, 3)]
    if not utils.QUICK_MODE
    else [(100, 2), (1000, 2)]
)
HISTDD_NUM_BINS = [4, 7, 11] if not utils.QUICK_MODE else [7]
# Weighted histograms only validate float32: low-precision weight accumulation
# in the aten reference diverges from FlagGems' wider accumulator.
HISTDD_WEIGHT_DTYPES = [torch.float32] if utils.QUICK_MODE else utils.FLOAT_DTYPES


def _make_input(shape, dtype, device):
    """Random N-dimensional points (shape[-1] is the dimensionality)."""
    return torch.randn(shape, dtype=dtype, device=device)


def _ref_histogramdd(inp, bins, *, weight=None, density=False):
    """Run the aten reference on CPU (no native CUDA impl is available)."""
    inp_cpu = inp.detach().to("cpu")
    bins_cpu = tuple(b.detach().to("cpu") for b in bins)
    weight_cpu = weight.detach().to("cpu") if weight is not None else None
    return torch._histogramdd_from_bin_tensors(
        inp_cpu, bins_cpu, weight=weight_cpu, density=density
    )


def _assert_weighted_close(res, ref, dtype, num_points):
    """Compare a weighted histogram against the aten reference.

    Each bin is a reduction over the points that fall into it.  The aten
    reference accumulates low-precision (float16/bfloat16) weights directly in
    that dtype, so its per-bin sums carry an accumulation bias of up to
    roughly ``num_points * eps(dtype)`` that a wider (float32) accumulator --
    which FlagGems uses for fidelity -- does not reproduce.  Scale the
    absolute tolerance by the number of points to absorb this irreducible
    difference while keeping the relative tolerance tight for large bins.
    """
    reduce_dim = num_points
    atol = 1e-4 if dtype == torch.float32 else 1e-3
    utils.gems_assert_close(res.to("cpu"), ref, dtype, reduce_dim=reduce_dim, atol=atol)


@pytest.mark.histogramdd_from_bin_tensors
@pytest.mark.parametrize("shape", HISTDD_SHAPES)
@pytest.mark.parametrize("num_bins", HISTDD_NUM_BINS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_histogramdd_from_bin_tensors(shape, num_bins, dtype):
    """Basic multi-dimensional histogram (no weights)."""
    D = shape[-1]
    inp = _make_input(shape, dtype, flag_gems.device)
    bins = tuple(
        torch.linspace(-3.0, 3.0, num_bins, dtype=dtype, device=flag_gems.device)
        for _ in range(D)
    )

    ref_out = _ref_histogramdd(inp, bins)
    res_out = flag_gems._histogramdd_from_bin_tensors(inp, bins)

    utils.gems_assert_close(res_out.to("cpu"), ref_out, dtype)


@pytest.mark.histogramdd_from_bin_tensors
@pytest.mark.parametrize("shape", HISTDD_SHAPES)
@pytest.mark.parametrize("num_bins", HISTDD_NUM_BINS)
@pytest.mark.parametrize("dtype", HISTDD_WEIGHT_DTYPES)
def test_histogramdd_from_bin_tensors_weighted(shape, num_bins, dtype):
    """Histogram with per-point weights (weight dtype matches input dtype)."""
    D = shape[-1]
    inp = _make_input(shape, dtype, flag_gems.device)
    bins = tuple(
        torch.linspace(-3.0, 3.0, num_bins, dtype=dtype, device=flag_gems.device)
        for _ in range(D)
    )
    weight = torch.rand(shape[:-1], dtype=dtype, device=flag_gems.device)

    ref_out = _ref_histogramdd(inp, bins, weight=weight)
    res_out = flag_gems._histogramdd_from_bin_tensors(inp, bins, weight=weight)

    _assert_weighted_close(res_out, ref_out, dtype, inp.numel() // D)


@pytest.mark.histogramdd_from_bin_tensors
@pytest.mark.parametrize("shape", HISTDD_SHAPES)
@pytest.mark.parametrize("num_bins", HISTDD_NUM_BINS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_histogramdd_from_bin_tensors_density(shape, num_bins, dtype):
    """Density-normalised histogram (counts / total / bin volume)."""
    D = shape[-1]
    inp = _make_input(shape, dtype, flag_gems.device)
    bins = tuple(
        torch.linspace(-3.0, 3.0, num_bins, dtype=dtype, device=flag_gems.device)
        for _ in range(D)
    )

    ref_out = _ref_histogramdd(inp, bins, density=True)
    res_out = flag_gems._histogramdd_from_bin_tensors(inp, bins, density=True)

    utils.gems_assert_close(res_out.to("cpu"), ref_out, dtype)


@pytest.mark.histogramdd_from_bin_tensors
@pytest.mark.parametrize("shape", HISTDD_SHAPES)
@pytest.mark.parametrize("num_bins", HISTDD_NUM_BINS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_histogramdd_from_bin_tensors_density_weighted(shape, num_bins, dtype):
    """Density-normalised histogram with weights."""
    D = shape[-1]
    inp = _make_input(shape, dtype, flag_gems.device)
    bins = tuple(
        torch.linspace(-3.0, 3.0, num_bins, dtype=dtype, device=flag_gems.device)
        for _ in range(D)
    )
    weight = torch.rand(shape[:-1], dtype=dtype, device=flag_gems.device)

    ref_out = _ref_histogramdd(inp, bins, weight=weight, density=True)
    res_out = flag_gems._histogramdd_from_bin_tensors(
        inp, bins, weight=weight, density=True
    )

    _assert_weighted_close(res_out, ref_out, dtype, inp.numel() // D)


@pytest.mark.histogramdd_from_bin_tensors
def test_histogramdd_from_bin_tensors_non_uniform_bins():
    """Non-uniform, per-dimension bin edges (different counts per dim)."""
    inp = torch.randn(2000, 2, dtype=torch.float32, device=flag_gems.device)
    bins = (
        torch.tensor([-3.0, -1.0, 0.0, 2.0, 3.0], device=flag_gems.device),
        torch.tensor([-2.0, 0.0, 1.0, 2.0], device=flag_gems.device),
    )

    ref_out = _ref_histogramdd(inp, bins)
    res_out = flag_gems._histogramdd_from_bin_tensors(inp, bins)

    utils.gems_assert_close(res_out.to("cpu"), ref_out, torch.float32)


@pytest.mark.histogramdd_from_bin_tensors
def test_histogramdd_from_bin_tensors_higher_dim_input():
    """Input with more than 2 dims is flattened to (M, D) points."""
    inp = torch.randn(4, 5, 2, dtype=torch.float32, device=flag_gems.device)
    bins = (
        torch.linspace(-3.0, 3.0, 7, device=flag_gems.device),
        torch.linspace(-3.0, 3.0, 7, device=flag_gems.device),
    )
    weight = torch.rand(4, 5, dtype=torch.float32, device=flag_gems.device)

    ref_out = _ref_histogramdd(inp, bins, weight=weight)
    res_out = flag_gems._histogramdd_from_bin_tensors(inp, bins, weight=weight)

    utils.gems_assert_close(res_out.to("cpu"), ref_out, torch.float32)


@pytest.mark.histogramdd_from_bin_tensors
def test_histogramdd_from_bin_tensors_outliers():
    """Points outside the bin range are dropped from the histogram."""
    inp = torch.tensor(
        [[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0], [0.0, -10.0], [0.0, 10.0]],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    bins = (
        torch.linspace(-3.0, 3.0, 7, device=flag_gems.device),
        torch.linspace(-3.0, 3.0, 7, device=flag_gems.device),
    )

    ref_out = _ref_histogramdd(inp, bins)
    res_out = flag_gems._histogramdd_from_bin_tensors(inp, bins)

    utils.gems_assert_close(res_out.to("cpu"), ref_out, torch.float32)


@pytest.mark.histogramdd_from_bin_tensors
def test_histogramdd_from_bin_tensors_boundary_edges():
    """Points exactly on internal/external edges follow left-inclusive rule."""
    edges = torch.tensor([0.0, 1.0, 2.0, 3.0], device=flag_gems.device)
    inp = torch.tensor(
        [[0.0], [0.999], [1.0], [1.001], [2.0], [2.999], [3.0], [3.001], [-0.5]],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    bins = (edges,)

    ref_out = _ref_histogramdd(inp, bins)
    res_out = flag_gems._histogramdd_from_bin_tensors(inp, bins)

    # Counts are exact integers, so require bit-equality.
    utils.gems_assert_equal(res_out.to("cpu"), ref_out)


@pytest.mark.histogramdd_from_bin_tensors
@pytest.mark.parametrize("D", [1, 2, 3, 4, 5])
def test_histogramdd_from_bin_tensors_dims(D):
    """Smoke test across the supported dimensionalities (1..5)."""
    inp = torch.randn(500, D, dtype=torch.float32, device=flag_gems.device)
    bins = tuple(
        torch.linspace(-3.0, 3.0, 5, device=flag_gems.device) for _ in range(D)
    )

    ref_out = _ref_histogramdd(inp, bins)
    res_out = flag_gems._histogramdd_from_bin_tensors(inp, bins)

    utils.gems_assert_close(res_out.to("cpu"), ref_out, torch.float32)


# ---------------------------------------------------------------------------
# Out variant: aten::_histogramdd_from_bin_tensors.out
# ---------------------------------------------------------------------------


@pytest.mark.histogramdd_from_bin_tensors_out
@pytest.mark.parametrize("shape", HISTDD_SHAPES)
@pytest.mark.parametrize("num_bins", HISTDD_NUM_BINS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_histogramdd_from_bin_tensors_out(shape, num_bins, dtype):
    """Out variant: result is written into the provided ``out`` tensor."""
    D = shape[-1]
    inp = _make_input(shape, dtype, flag_gems.device)
    bins = tuple(
        torch.linspace(-3.0, 3.0, num_bins, dtype=dtype, device=flag_gems.device)
        for _ in range(D)
    )
    num_bins_per_dim = num_bins - 1
    out_shape = (num_bins_per_dim,) * D
    out = torch.zeros(out_shape, dtype=dtype, device=flag_gems.device)

    ref_out = _ref_histogramdd(inp, bins)
    res_out = flag_gems._histogramdd_from_bin_tensors_out(inp, bins, out=out)

    # The .out variant writes into and returns the provided out buffer; the
    # returned tensor shares storage with ``out`` (reshape may return a view).
    assert res_out.data_ptr() == out.data_ptr()
    utils.gems_assert_close(res_out.to("cpu"), ref_out, dtype)


@pytest.mark.histogramdd_from_bin_tensors
@pytest.mark.parametrize("shape", HISTDD_SHAPES)
@pytest.mark.parametrize("num_bins", HISTDD_NUM_BINS)
@pytest.mark.parametrize("dtype", HISTDD_WEIGHT_DTYPES)
def test_histogramdd_from_bin_tensors_out_weighted(shape, num_bins, dtype):
    """Out variant with weights.

    Grouped under the base mark since the strict per-function mark check keys
    the expected mark off the trailing test-name token (``_weighted``), which is
    not a registered variant id.
    """
    D = shape[-1]
    inp = _make_input(shape, dtype, flag_gems.device)
    bins = tuple(
        torch.linspace(-3.0, 3.0, num_bins, dtype=dtype, device=flag_gems.device)
        for _ in range(D)
    )
    weight = torch.rand(shape[:-1], dtype=dtype, device=flag_gems.device)
    num_bins_per_dim = num_bins - 1
    out_shape = (num_bins_per_dim,) * D
    out = torch.zeros(out_shape, dtype=dtype, device=flag_gems.device)

    ref_out = _ref_histogramdd(inp, bins, weight=weight)
    res_out = flag_gems._histogramdd_from_bin_tensors_out(
        inp, bins, weight=weight, out=out
    )

    # The .out variant writes into and returns the provided out buffer; the
    # returned tensor shares storage with ``out`` (reshape may return a view).
    assert res_out.data_ptr() == out.data_ptr()
    _assert_weighted_close(res_out, ref_out, dtype, inp.numel() // D)
