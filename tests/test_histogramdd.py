# Copyright 2026, The FlagOS Contributors.
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


@pytest.mark.histogramdd
@pytest.mark.parametrize("shape", [(50, 2), (100, 3), (80, 4)])
@pytest.mark.parametrize("bins", [[4, 5], [3, 4, 5], [2, 3, 4, 5]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_histogramdd_basic(shape, bins, dtype):
    if len(bins) != shape[1]:
        pytest.skip("bins length must match D dimension")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # Native histogramdd is CPU-only; compute reference on CPU
    ref_hist, ref_edges = torch.histogramdd(inp.cpu(), bins=bins)
    res_hist, res_edges = flag_gems.histogramdd(inp, bins=bins)

    # Compare histogram (move result to CPU for comparison)
    utils.gems_assert_equal(res_hist.cpu(), ref_hist)

    # Compare bin edges
    assert len(res_edges) == len(ref_edges), "edge list length mismatch"
    for i, (res_edge, ref_edge) in enumerate(zip(res_edges, ref_edges)):
        utils.gems_assert_close(
            res_edge.cpu(),
            ref_edge,
            dtype,
            atol=1e-4,
        )


@pytest.mark.histogramdd
@pytest.mark.parametrize("shape", [(60, 2), (40, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_histogramdd_with_range(shape, dtype):
    D = shape[1]
    bins = [5] * D
    # Define explicit range per dimension
    range_vals = []
    for d in range(D):
        range_vals.extend([-2.0, 2.0])

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_hist, ref_edges = torch.histogramdd(inp.cpu(), bins=bins, range=range_vals)
    res_hist, res_edges = flag_gems.histogramdd(inp, bins=bins, range=range_vals)

    utils.gems_assert_equal(res_hist.cpu(), ref_hist)

    for i, (res_edge, ref_edge) in enumerate(zip(res_edges, ref_edges)):
        utils.gems_assert_close(
            res_edge.cpu(),
            ref_edge,
            dtype,
            atol=1e-4,
        )


@pytest.mark.histogramdd
@pytest.mark.parametrize("shape", [(30, 2), (50, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_histogramdd_density(shape, dtype):
    D = shape[1]
    bins = [4] * D

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_hist, ref_edges = torch.histogramdd(inp.cpu(), bins=bins, density=True)
    res_hist, res_edges = flag_gems.histogramdd(inp, bins=bins, density=True)

    # Density normalization may have larger relative errors
    utils.gems_assert_close(res_hist.cpu(), ref_hist, dtype, atol=1e-3)

    for i, (res_edge, ref_edge) in enumerate(zip(res_edges, ref_edges)):
        utils.gems_assert_close(
            res_edge.cpu(),
            ref_edge,
            dtype,
            atol=1e-4,
        )


@pytest.mark.histogramdd
def test_histogramdd_empty():
    """Test with empty input (N=0)."""
    inp = torch.zeros(0, 2, dtype=torch.float32, device=flag_gems.device)
    bins = [3, 3]

    hist, edges = flag_gems.histogramdd(inp, bins=bins)

    assert hist.shape == (3, 3)
    assert hist.sum().item() == 0.0
    assert len(edges) == 2


@pytest.mark.histogramdd
def test_histogramdd_degenerate_range():
    """Test with all values equal (degenerate range)."""
    # All points at the same location
    inp = torch.ones(10, 2, dtype=torch.float32, device=flag_gems.device) * 5.0
    bins = [3, 3]

    ref_hist, ref_edges = torch.histogramdd(inp.cpu(), bins=bins)
    res_hist, res_edges = flag_gems.histogramdd(inp, bins=bins)

    utils.gems_assert_equal(res_hist.cpu(), ref_hist)

    # Check that edges span [val-0.5, val+0.5] for degenerate case
    for i, (res_edge, ref_edge) in enumerate(zip(res_edges, ref_edges)):
        utils.gems_assert_close(
            res_edge.cpu(),
            ref_edge,
            torch.float32,
            atol=1e-4,
        )
