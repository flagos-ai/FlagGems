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
from flag_gems.ops.histogram import histogram_bin_ct, histogram_bins_tensor

from . import conftest as cfg
from .accuracy_utils import gems_assert_close

if cfg.QUICK_MODE:
    HISTOGRAM_SHAPES = [(128,), (64, 64)]
    HISTOGRAM_BINS = [10]
else:
    HISTOGRAM_SHAPES = [(128,), (1024,), (4096,), (64, 64), (32, 64, 16)]
    HISTOGRAM_BINS = [10, 50, 100]
HISTOGRAM_DTYPES = [torch.float32, torch.float64]


@pytest.mark.histogram
@pytest.mark.histogram_bin_ct
@pytest.mark.parametrize("shape", HISTOGRAM_SHAPES)
@pytest.mark.parametrize("bins", HISTOGRAM_BINS)
@pytest.mark.parametrize("dtype", HISTOGRAM_DTYPES)
def test_accuracy_histogram_bin_ct(shape, bins, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # histogram is CPU-only in torch, must use CPU reference
    ref_inp = inp.cpu()
    ref_hist, ref_edges = torch.histogram(ref_inp, bins=bins)
    # Direct dispatch, NO use_gems()
    res_hist, res_edges = histogram_bin_ct(inp, bins=bins)
    gems_assert_close(res_hist.cpu(), ref_hist, dtype)
    gems_assert_close(res_edges.cpu(), ref_edges, dtype)


@pytest.mark.histogram
@pytest.mark.histogram_bin_ct
@pytest.mark.parametrize("shape", HISTOGRAM_SHAPES)
@pytest.mark.parametrize("bins", HISTOGRAM_BINS)
@pytest.mark.parametrize("dtype", HISTOGRAM_DTYPES)
def test_accuracy_histogram_bin_ct_with_range(shape, bins, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.cpu()
    ref_hist, ref_edges = torch.histogram(ref_inp, bins=bins, range=(0.0, 10.0))
    res_hist, res_edges = histogram_bin_ct(inp, bins=bins, range=(0.0, 10.0))
    gems_assert_close(res_hist.cpu(), ref_hist, dtype)
    gems_assert_close(res_edges.cpu(), ref_edges, dtype)


@pytest.mark.histogram
@pytest.mark.histogram_bin_ct
@pytest.mark.parametrize("shape", [(256,), (100, 50)])
@pytest.mark.parametrize("dtype", HISTOGRAM_DTYPES)
def test_accuracy_histogram_bin_ct_with_weight(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.cpu()
    ref_weight = weight.cpu()
    ref_hist, ref_edges = torch.histogram(ref_inp, bins=15, weight=ref_weight)
    res_hist, res_edges = histogram_bin_ct(inp, bins=15, weight=weight)
    gems_assert_close(res_hist.cpu(), ref_hist, dtype, reduce_dim=15)
    gems_assert_close(res_edges.cpu(), ref_edges, dtype)


@pytest.mark.histogram
@pytest.mark.histogram_bin_ct
@pytest.mark.parametrize("shape", [(512,), (64, 64)])
@pytest.mark.parametrize("dtype", HISTOGRAM_DTYPES)
def test_accuracy_histogram_bin_ct_with_density(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.cpu()
    ref_hist, ref_edges = torch.histogram(ref_inp, bins=25, density=True)
    res_hist, res_edges = histogram_bin_ct(inp, bins=25, density=True)
    gems_assert_close(res_hist.cpu(), ref_hist, dtype, reduce_dim=25)
    gems_assert_close(res_edges.cpu(), ref_edges, dtype)


@pytest.mark.histogram
@pytest.mark.histogram_bins_tensor
@pytest.mark.parametrize("shape", [(128,), (512,), (64, 64)])
@pytest.mark.parametrize("dtype", HISTOGRAM_DTYPES)
def test_accuracy_histogram_bins_tensor(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    edges = torch.linspace(-3.0, 3.0, 11, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.cpu()
    ref_edges = edges.cpu()
    ref_hist, ref_bin_edges = torch.histogram(ref_inp, ref_edges)
    res_hist, res_bin_edges = histogram_bins_tensor(inp, edges)
    gems_assert_close(res_hist.cpu(), ref_hist, dtype, reduce_dim=10)
    gems_assert_close(res_bin_edges.cpu(), ref_bin_edges, dtype)


@pytest.mark.histogram
@pytest.mark.histogram_bins_tensor
@pytest.mark.parametrize("shape", [(256,)])
@pytest.mark.parametrize("dtype", HISTOGRAM_DTYPES)
def test_accuracy_histogram_bins_tensor_nonuniform(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Non-uniform bins
    edges = torch.tensor([0.0, 1.0, 2.0, 10.0], dtype=dtype, device=flag_gems.device)
    ref_inp = inp.cpu()
    ref_edges = edges.cpu()
    ref_hist, ref_bin_edges = torch.histogram(ref_inp, ref_edges)
    res_hist, res_bin_edges = histogram_bins_tensor(inp, edges)
    gems_assert_close(res_hist.cpu(), ref_hist, dtype, reduce_dim=3)
    gems_assert_close(res_bin_edges.cpu(), ref_bin_edges, dtype)
