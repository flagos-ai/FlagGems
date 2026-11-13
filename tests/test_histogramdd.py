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


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


_POINT_CASES = [
    pytest.param((50, 2), [4, 5], id="2d"),
    pytest.param((100, 3), [3, 4, 5], id="3d"),
    pytest.param((80, 4), [2, 3, 4, 5], id="4d"),
]


def _random_input(factory, shape, dtype):
    device = flag_gems.device
    if flag_gems.vendor_name == "iluvatar" and dtype == torch.float64:
        # Iluvatar's device FP64 RNG produces zeros; same-dtype copies work.
        return factory(shape, dtype=dtype, device="cpu").to(device)
    return factory(shape, dtype=dtype, device=device)


def _cpu_input(inp):
    if flag_gems.vendor_name == "iluvatar" and inp.dtype == torch.float64:
        return inp.cpu()
    return utils.to_reference(inp, True).cpu()


def _reference(inp, bins, *, range=None, weight=None, density=False):
    ref_inp = _cpu_input(inp)
    if isinstance(bins, (tuple, list)) and bins and isinstance(bins[0], torch.Tensor):
        ref_bins = tuple(_cpu_input(item) for item in bins)
    else:
        ref_bins = bins
    ref_weight = _cpu_input(weight) if weight is not None else None
    hist, edges = torch.histogramdd(
        ref_inp, bins=ref_bins, range=range, weight=ref_weight, density=density
    )
    return hist.to(inp.dtype), tuple(edge.to(inp.dtype) for edge in edges)


def _assert_result(result, reference, dtype, device, *, hist_atol=1e-4, exact_hist=False):
    hist, edges = result
    ref_hist, ref_edges = reference
    assert hist.device == device
    assert all(edge.device == device for edge in edges)
    assert hist.shape == ref_hist.shape
    assert hist.dtype == ref_hist.dtype
    assert len(edges) == len(ref_edges)
    if not exact_hist and dtype in utils.ALL_FLOAT_DTYPES:
        utils.gems_assert_close(hist.cpu(), ref_hist, dtype, atol=hist_atol)
    else:
        utils.gems_assert_equal(hist.cpu(), ref_hist)
    for edge, ref_edge in zip(edges, ref_edges):
        assert edge.shape == ref_edge.shape
        assert edge.dtype == ref_edge.dtype
        utils.gems_assert_close(edge.cpu(), ref_edge, dtype, atol=1e-4)


@pytest.mark.histogramdd
@pytest.mark.parametrize("shape,bins", _POINT_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_histogramdd_bin_counts(shape, bins, dtype):
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("float64 is not supported by this device")
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    before = inp.clone()
    reference = _reference(inp, bins)

    result = flag_gems.histogramdd(inp, bins=bins)

    _assert_result(result, reference, dtype, inp.device, exact_hist=True)
    torch.testing.assert_close(inp, before, atol=0, rtol=0)


@pytest.mark.histogramdd
@pytest.mark.parametrize("shape", [(60, 2), (40, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_histogramdd_with_range(shape, dtype):
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("float64 is not supported by this device")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    bins = [5] * shape[-1]
    bounds = [-2.0, 2.0] * shape[-1]
    reference = _reference(inp, bins, range=bounds)
    result = flag_gems.histogramdd(inp, bins=bins, range=bounds)
    _assert_result(result, reference, dtype, inp.device)
    utils.gems_assert_equal(result[0].cpu(), reference[0])


@pytest.mark.histogramdd
@pytest.mark.parametrize("shape", [(30, 2), (50, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_histogramdd_density(shape, dtype):
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("float64 is not supported by this device")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    bins = [4] * shape[-1]
    reference = _reference(inp, bins, density=True)
    result = flag_gems.histogramdd(inp, bins=bins, density=True)
    _assert_result(result, reference, dtype, inp.device, hist_atol=1e-3)


@pytest.mark.histogramdd
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("density", [False, True])
def test_histogramdd_range_weight_density(dtype, weighted, density):
    torch.manual_seed(0)
    inp = _random_input(torch.randn, (60, 2), dtype)
    weight = (
        _random_input(torch.rand, (60,), dtype) if weighted else None
    )
    range_ = [-2.0, 2.0, -3.0, 3.0]
    reference = _reference(inp, [5, 4], range=range_, weight=weight, density=density)

    result = flag_gems.histogramdd(
        inp, bins=[5, 4], range=range_, weight=weight, density=density
    )

    _assert_result(result, reference, dtype, inp.device)


@pytest.mark.histogramdd
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_histogramdd_tensor_list_bins(dtype):
    inp = torch.tensor(
        [[-1.0, 0.0], [0.25, 0.5], [1.0, 1.5], [2.0, -0.5]],
        dtype=dtype,
        device=flag_gems.device,
    )
    bins = (
        torch.tensor([-2.0, 0.0, 1.0, 3.0], dtype=dtype, device=flag_gems.device),
        torch.tensor([-1.0, 0.25, 1.0, 2.0], dtype=dtype, device=flag_gems.device),
    )
    reference = _reference(inp, bins)

    result = flag_gems.histogramdd(inp, bins=bins)

    _assert_result(result, reference, dtype, inp.device)


@pytest.mark.histogramdd
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_histogramdd_integer_bins_overload(dtype):
    torch.manual_seed(0)
    inp = _random_input(torch.randn, (40, 3), dtype)
    reference = _reference(inp, 4)

    result = flag_gems.histogramdd(inp, bins=4)

    _assert_result(result, reference, dtype, inp.device)


@pytest.mark.histogramdd
def test_histogramdd_empty_and_degenerate_range():
    empty = torch.empty((0, 2), dtype=torch.float32, device=flag_gems.device)
    empty_result = flag_gems.histogramdd(empty, bins=[3, 3])
    _assert_result(
        empty_result,
        _reference(empty, [3, 3]),
        torch.float32,
        empty.device,
        exact_hist=True,
    )

    inp = torch.full((10, 2), 5.0, device=flag_gems.device)
    result = flag_gems.histogramdd(inp, bins=[3, 3])
    _assert_result(
        result, _reference(inp, [3, 3]), torch.float32, inp.device, exact_hist=True
    )


@pytest.mark.histogramdd
def test_histogramdd_noncontiguous_input():
    torch.manual_seed(0)
    inp = torch.randn((2, 32), device=flag_gems.device).transpose(0, 1)
    assert not inp.is_contiguous()
    reference = _reference(inp, [4, 5])

    result = flag_gems.histogramdd(inp, bins=[4, 5])

    _assert_result(result, reference, torch.float32, inp.device)


@pytest.mark.histogramdd
def test_histogramdd_rejects_mismatched_bins():
    inp = torch.randn((20, 2), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.histogramdd(inp, bins=[4, 5, 6])
