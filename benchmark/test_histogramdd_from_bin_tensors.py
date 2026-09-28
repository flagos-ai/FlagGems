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

# ``aten::_histogramdd_from_bin_tensors`` has no native CUDA implementation, so the
# baseline necessarily runs on CPU. Two consequences, both handled below:
#
#  * The transfers must not be timed. Moving the inputs to CPU and the result back
#    inside the timed call charges the native op for copies the Triton kernel never
#    performs, which inflates the ratio by transfer cost rather than measuring the
#    kernel. CPU copies of each generated input are staged up front instead.
#  * ``speedup`` is not reported at all: ``latency_base`` is a CPU measurement and
#    ``latency`` a CUDA one, so their ratio is not operator parity. Both latencies
#    are still shown individually, which is the honest form of this comparison.
HISTDD_BENCH_SHAPES = [
    (1024, 2),
    (4096, 2),
    (16384, 2),
    (4096, 3),
    (8192, 4),
]


# Pre-staged CPU copies of the generated inputs, keyed by storage identity, so the
# timed baseline call does no host-device traffic.
_CPU_CACHE = {}


def _stage(t):
    """Remember a CPU copy of ``t`` and return the key used to look it up."""
    key = (t.data_ptr(), tuple(t.shape), t.dtype)
    _CPU_CACHE[key] = t.detach().to("cpu")
    return key


def _cpu(t):
    """Return the pre-staged CPU copy of ``t``, falling back to a fresh copy."""
    staged = _CPU_CACHE.get((t.data_ptr(), tuple(t.shape), t.dtype))
    return t.detach().to("cpu") if staged is None else staged


def _histogramdd_reference(inp, bins, *, weight=None, density=False):
    """Native CPU baseline; reads pre-staged CPU tensors so no copies are timed."""
    return torch._histogramdd_from_bin_tensors(
        _cpu(inp),
        tuple(_cpu(b) for b in bins),
        weight=None if weight is None else _cpu(weight),
        density=density,
    )


def _histogramdd_input_fn(shape, dtype, device):
    """Yield (input, bins, kwargs) tuples for the benchmark variants."""
    D = shape[-1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    bins = tuple(
        torch.linspace(-3.0, 3.0, 11, dtype=dtype, device=device) for _ in range(D)
    )
    _stage(inp)
    for b in bins:
        _stage(b)
    # Plain histogram (no weights).
    yield (inp, bins, {})
    # Weighted histogram.
    weight = torch.rand(shape[:-1], dtype=dtype, device=device)
    _stage(weight)
    yield (inp, bins, {"weight": weight})
    # Density histogram.
    yield (inp, bins, {"density": True})


class HistogramddBenchmark(base.GenericBenchmark):
    # No speedup column: the baseline is CPU-only, so the ratio would compare two
    # different devices. See the module docstring above.
    DEFAULT_METRICS = ["latency_base", "latency"]

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
        _stage(inp)
        for b in bins:
            _stage(b)
        out = torch.zeros((10,) * D, dtype=dtype, device=device)
        # ``out`` is passed as a keyword argument to the aten .out overload.
        yield (inp, bins, {"out": out})
        weight = torch.rand(shape[:-1], dtype=dtype, device=device)
        _stage(weight)
        out_w = torch.zeros((10,) * D, dtype=dtype, device=device)
        yield (inp, bins, {"weight": weight, "out": out_w})

    def _out_reference(inp, bins, *, weight=None, density=False, out=None):
        # Deliberately does not copy back into ``out``: that would be a D2H/H2D
        # round trip the gems path never performs. The baseline measures the
        # native kernel only.
        return _histogramdd_reference(inp, bins, weight=weight, density=density)

    bench = HistogramddBenchmark(
        input_fn=_out_input_fn,
        op_name="histogramdd_from_bin_tensors_out",
        torch_op=_out_reference,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems._histogramdd_from_bin_tensors_out)
    bench.run()
