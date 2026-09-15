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

from . import base, consts

# ``aten::_histogramdd_from_bin_tensors`` has no native CUDA implementation in
# this PyTorch build, so the torch baseline is computed on CPU (the inputs are
# moved to CPU, the reference histogram is run there, and the result is moved
# back to the original device).  The FlagGems kernel runs on the GPU, so the
# reported speedup compares the CPU reference against the Triton kernel.
HISTDD_BENCH_SHAPES = [
    (1024, 2),
    (4096, 2),
    (16384, 2),
    (4096, 3),
    (8192, 4),
]


def _histogramdd_reference(inp, bins, *, weight=None, density=False):
    """CPU reference baseline (no native CUDA impl is available)."""
    inp_cpu = inp.detach().to("cpu")
    bins_cpu = tuple(b.detach().to("cpu") for b in bins)
    weight_cpu = weight.detach().to("cpu") if weight is not None else None
    out = torch._histogramdd_from_bin_tensors(
        inp_cpu, bins_cpu, weight=weight_cpu, density=density
    )
    return out.to(inp.device)


def _histogramdd_input_fn(shape, dtype, device):
    """Yield (input, bins, kwargs) tuples for the benchmark variants."""
    D = shape[-1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    bins = tuple(
        torch.linspace(-3.0, 3.0, 11, dtype=dtype, device=device) for _ in range(D)
    )
    # Plain histogram (no weights).
    yield (inp, bins, {})
    # Weighted histogram.
    weight = torch.rand(shape[:-1], dtype=dtype, device=device)
    yield (inp, bins, {"weight": weight})
    # Density histogram.
    yield (inp, bins, {"density": True})


class HistogramddBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Override the default (gigabyte-scale) shapes with point-cloud sizes
        # that are meaningful for a multi-dimensional histogram.
        self.shapes = HISTDD_BENCH_SHAPES


@pytest.mark.histogramdd_from_bin_tensors
def test_histogramdd_from_bin_tensors():
    bench = HistogramddBenchmark(
        input_fn=_histogramdd_input_fn,
        op_name="histogramdd_from_bin_tensors",
        torch_op=_histogramdd_reference,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems._histogramdd_from_bin_tensors)
    bench.run()


@pytest.mark.histogramdd_from_bin_tensors_out
def test_histogramdd_from_bin_tensors_out():
    def _out_input_fn(shape, dtype, device):
        D = shape[-1]
        inp = torch.randn(shape, dtype=dtype, device=device)
        bins = tuple(
            torch.linspace(-3.0, 3.0, 11, dtype=dtype, device=device) for _ in range(D)
        )
        out = torch.zeros((10,) * D, dtype=dtype, device=device)
        # ``out`` is passed as a keyword argument to the aten .out overload.
        yield (inp, bins, {"out": out})
        weight = torch.rand(shape[:-1], dtype=dtype, device=device)
        out_w = torch.zeros((10,) * D, dtype=dtype, device=device)
        yield (inp, bins, {"weight": weight, "out": out_w})

    def _out_reference(inp, bins, *, weight=None, density=False, out=None):
        ref = _histogramdd_reference(inp, bins, weight=weight, density=density)
        if out is not None:
            out.copy_(ref)
            return out
        return ref

    bench = HistogramddBenchmark(
        input_fn=_out_input_fn,
        op_name="histogramdd_from_bin_tensors_out",
        torch_op=_out_reference,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems._histogramdd_from_bin_tensors_out)
    bench.run()
