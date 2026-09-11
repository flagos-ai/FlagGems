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

from . import base


def _bin_ct_input_fn(shape, dtype, device):
    # histogram.bin_ct operates on any-shaped input, flattened internally.
    inp = torch.randn(shape, dtype=dtype, device=device)
    # 100 bins is a common default for histogram benchmarks.
    yield inp, {"bins": 100}


def _bins_tensor_input_fn(shape, dtype, device):
    inp = torch.rand(shape, dtype=dtype, device=device)
    # 101 edges -> 100 uniform bins over [0, 1] to match rand() range.
    bins = torch.linspace(0.0, 1.0, 101, dtype=dtype, device=device)
    yield inp, {"bins": bins}


def _bin_ct_torch_cpu(inp, bins):
    """Run torch.histogram on CPU; it has no CUDA implementation."""
    device = inp.device
    hist, edges = torch.histogram(inp.cpu(), bins=bins)
    return hist.to(device), edges.to(device)


def _bins_tensor_torch_cpu(inp, bins):
    """Run torch.histogram on CPU; it has no CUDA implementation."""
    device = inp.device
    hist, edges = torch.histogram(inp.cpu(), bins=bins.cpu())
    return hist.to(device), edges.to(device)


@pytest.mark.histogram
@pytest.mark.histogram_bin_ct
def test_histogram_bin_ct():
    """
    Benchmark histogram.bin_ct operator.

    Note: Native torch.histogram has no CUDA implementation and only runs on
    CPU. This benchmark compares the GPU gems kernel against the CPU native
    reference, which is not a fair device-to-device comparison but shows
    performance capability.

    torch.histogram does not support float16, so only float32 and float64
    are tested.
    """
    bench = base.GenericBenchmark(
        input_fn=_bin_ct_input_fn,
        op_name="histogram.bin_ct",
        torch_op=_bin_ct_torch_cpu,
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()


@pytest.mark.histogram
@pytest.mark.histogram_bins_tensor
def test_histogram_bins_tensor():
    """
    Benchmark histogram.bins_tensor operator.

    Note: Native torch.histogram has no CUDA implementation and only runs on
    CPU. This benchmark compares the GPU gems kernel against the CPU native
    reference, which is not a fair device-to-device comparison but shows
    performance capability.

    torch.histogram does not support float16, so only float32 and float64
    are tested.
    """
    bench = base.GenericBenchmark(
        input_fn=_bins_tensor_input_fn,
        op_name="histogram.bins_tensor",
        torch_op=_bins_tensor_torch_cpu,
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
