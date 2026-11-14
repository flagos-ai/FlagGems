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

from . import base

# torch._histogramdd_from_bin_cts is only implemented for float32/float64.
BENCH_DTYPES = [torch.float32]

# Point clouds (M, N): vary the number of points M while keeping the dimension
# N small (1, 2, 3). The last dim is the histogram dimension; large N would
# explode the number of output bins, so it stays bounded here.
POINT_SHAPES = [
    (1024, 2),
    (10000, 2),
    (100000, 2),
    (1000, 3),
    (50000, 3),
    (100000, 3),
    (10000, 1),
    (1000000, 1),
]


class HistogramddBenchmark(base.GenericBenchmark):
    """GenericBenchmark with a fixed, controlled shape set.

    Unlike the default ``set_more_shapes`` (which adds huge last-dim tensors
    suitable for pointwise ops), histogramdd treats the *last* dim as the number
    of histogram dimensions, so it must stay small while the leading dims (the
    number of points) grow. We therefore pin ``self.shapes`` to POINT_SHAPES
    instead of the default large-2D/3D pointwise shapes.
    """

    def init_user_config(self):
        super().init_user_config()
        # Override the default/loaded shapes with our point-cloud shapes.
        self.shapes = POINT_SHAPES

    def set_more_shapes(self):
        return POINT_SHAPES


def _make_cpu_baseline(torch_op):
    """Wrap a CUDA-only-input op to run its (CPU-only) reference on CPU.

    torch._histogramdd_from_bin_cts has no native CUDA kernel, so the "native"
    baseline (latency_base) must run on CPU. We move the inputs/weights to CPU,
    run the reference there, and move the result back to CUDA so the device
    matches the GEMS path. This measures CPU-vs-GPU speedup, which is the
    meaningful comparison for an op PyTorch only ships on CPU.
    """

    def wrapper(inp, *, bins, range=None, weight=None, density=False, **kw):
        inp_cpu = inp.detach().to("cpu")
        weight_cpu = weight.detach().to("cpu") if weight is not None else None
        out = torch_op(inp_cpu, bins, range=range, weight=weight_cpu, density=density)
        return out.to(inp.device)

    return wrapper


def _input_fn(shape, dtype, device):
    """Yield (input, kwargs): a point cloud and the bins/range kwargs."""
    inp = torch.randn(shape, dtype=dtype, device=device) * 10
    ndim = shape[-1]
    bins = [20] * ndim
    yield inp, {"bins": bins, "range": [-30.0, 30.0] * ndim}


def _input_fn_weighted(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device) * 10
    ndim = shape[-1]
    bins = [20] * ndim
    weight = torch.rand(shape[:-1], dtype=dtype, device=device) * 5
    yield inp, {"bins": bins, "range": [-30.0, 30.0] * ndim, "weight": weight}


def _input_fn_density(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device) * 10
    ndim = shape[-1]
    bins = [20] * ndim
    yield inp, {"bins": bins, "range": [-30.0, 30.0] * ndim, "density": True}


_BASELINE = _make_cpu_baseline(torch._histogramdd_from_bin_cts)


@pytest.mark.histogramdd_from_bin_cts
def test_histogramdd_from_bin_cts():
    bench = HistogramddBenchmark(
        input_fn=_input_fn,
        op_name="histogramdd_from_bin_cts",
        torch_op=_BASELINE,
        dtypes=BENCH_DTYPES,
    )
    bench.set_gems(flag_gems._histogramdd_from_bin_cts)
    bench.run()


@pytest.mark.histogramdd_from_bin_cts
def test_histogramdd_from_bin_cts_weighted():
    bench = HistogramddBenchmark(
        input_fn=_input_fn_weighted,
        op_name="histogramdd_from_bin_cts",
        torch_op=_BASELINE,
        dtypes=BENCH_DTYPES,
    )
    bench.set_gems(flag_gems._histogramdd_from_bin_cts)
    bench.run()


@pytest.mark.histogramdd_from_bin_cts
def test_histogramdd_from_bin_cts_density():
    bench = HistogramddBenchmark(
        input_fn=_input_fn_density,
        op_name="histogramdd_from_bin_cts",
        torch_op=_BASELINE,
        dtypes=BENCH_DTYPES,
    )
    bench.set_gems(flag_gems._histogramdd_from_bin_cts)
    bench.run()


@pytest.mark.histogramdd_from_bin_cts_out
def test_histogramdd_from_bin_cts_out():
    """Benchmark the .out variant via the aten namespace.

    The baseline wraps the CPU reference; the GEMS path calls the .out overload
    with a freshly-allocated output tensor.
    """

    def baseline(inp, *, bins, range=None, weight=None, density=False, **kw):
        inp_cpu = inp.detach().to("cpu")
        weight_cpu = weight.detach().to("cpu") if weight is not None else None
        out_shape = tuple(bins)
        out = torch.empty(out_shape, dtype=inp.dtype, device="cpu")
        torch.ops.aten._histogramdd_from_bin_cts.out(
            inp_cpu, bins, range=range, weight=weight_cpu, density=density, out=out
        )
        return out.to(inp.device)

    def gems(inp, *, bins, range=None, weight=None, density=False, **kw):
        out_shape = tuple(bins)
        out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
        return flag_gems._histogramdd_from_bin_cts_out(
            inp, bins, range=range, weight=weight, density=density, out=out
        )

    bench = HistogramddBenchmark(
        input_fn=_input_fn,
        op_name="histogramdd_from_bin_cts_out",
        torch_op=baseline,
        dtypes=BENCH_DTYPES,
    )
    bench.set_gems(gems)
    bench.run()
