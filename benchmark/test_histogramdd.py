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
from .generated_operator_utils import OperatorBenchmark

_POINT_SHAPES = [(1024, 2), (10000, 2), (4096, 3), (8192, 4)]
_MASTER_SHAPES = [((4096, dimension), "master") for dimension in (2, 3, 4)]


class HistogramddBenchmark(OperatorBenchmark):
    DEFAULT_METRICS = ["latency_base", "latency"]

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(
            shape_file_path, default_shapes=_POINT_SHAPES + _MASTER_SHAPES
        )


def _case_fn(shape, dtype):
    if len(shape) == 2 and shape[1] == "master":
        dims = shape[0]
        bins = [4] * dims[-1]
        densities = (
            (False, True)
            if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
            else (False,)
        )
        for density in densities:
            yield base.BenchmarkCasePlan(
                shape={"input": list(dims)},
                params={"bins_kind": "counts", "bins": bins, "density": density},
                builder_args=(dims, "master", density),
            )
        return
    if dtype != torch.float32:
        return
    dims = tuple(shape)
    dimension = dims[-1]
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={"bins_kind": "counts", "bins": [20] * dimension},
        builder_args=(dims, "counts", False),
    )
    # The core Definition is the bin-count overload. Keep the existing tensor
    # edge workload available at comprehensive level.
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield base.BenchmarkCasePlan(
            shape={"input": list(dims)},
            params={"bins_kind": "tensor_list", "edges": 21},
            builder_args=(dims, "tensor_list", False),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, bins_kind, density = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    dimension = shape[-1]
    if bins_kind == "master":
        kwargs = {"bins": [4] * dimension}
        if density:
            kwargs["density"] = True
        return inp, kwargs
    if bins_kind == "counts":
        bins = [20] * dimension
        return inp, {"bins": bins, "range": [-3.0, 3.0] * dimension}
    bins = tuple(
        torch.linspace(-3.0, 3.0, 21, dtype=dtype, device=device)
        for _ in range(dimension)
    )
    return inp, {"bins": bins}


def _cpu_reference(inp, *, bins, range=None, weight=None, density=False):
    cpu_bins = (
        tuple(item.detach().to("cpu") for item in bins)
        if isinstance(bins, (tuple, list))
        and bins
        and isinstance(bins[0], torch.Tensor)
        else bins
    )
    cpu_weight = weight.detach().to("cpu") if weight is not None else None
    return torch.histogramdd(
        inp.detach().to("cpu"),
        bins=cpu_bins,
        range=range,
        weight=cpu_weight,
        density=density,
    )


@pytest.mark.histogramdd
def test_histogramdd():
    bench = HistogramddBenchmark(
        op_name="histogramdd",
        torch_op=_cpu_reference,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32, torch.float64],
    )
    bench.set_gems(lambda *args, **kwargs: flag_gems.histogramdd(*args, **kwargs))
    bench.run()
