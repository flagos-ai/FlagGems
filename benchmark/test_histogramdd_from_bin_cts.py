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
from .generated_operator_utils import OperatorBenchmark


_POINT_SHAPES = [
    (1024, 2),
    (10000, 2),
    (100000, 2),
    (4096, 3),
    (100000, 3),
    (1000000, 1),
]


class HistogramddFromBinCtsBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_POINT_SHAPES)


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    dimension = dims[-1]
    common = (dims, [20] * dimension, [-30.0, 30.0] * dimension)
    for mode in ("plain", "weighted", "density"):
        yield base.BenchmarkCasePlan(
            shape={"input": list(dims)},
            params={"mode": mode, "bins": [20] * dimension},
            builder_args=(*common, mode),
        )


def _out_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    dimension = dims[-1]
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims), "out": [20] * dimension},
        params={"mode": "out", "bins": [20] * dimension},
        builder_args=(dims, [20] * dimension, [-30.0, 30.0] * dimension),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, bins, range_, mode = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device) * 10
    kwargs = {"bins": bins, "range": range_}
    if mode == "weighted":
        kwargs["weight"] = torch.rand(shape[:-1], dtype=dtype, device=device)
    elif mode == "density":
        kwargs["density"] = True
    return inp, kwargs


def _out_build_inputs_fn(plan, dtype, device):
    shape, bins, range_ = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device) * 10
    out = torch.empty(tuple(bins), dtype=dtype, device=device)
    return inp, {"bins": bins, "range": range_, "out": out}


def _cpu_reference(inp, *, bins, range=None, weight=None, density=False):
    cpu_weight = weight.detach().to("cpu") if weight is not None else None
    result = torch._histogramdd_from_bin_cts(
        inp.detach().to("cpu"),
        bins,
        range=range,
        weight=cpu_weight,
        density=density,
    )
    return result.to(inp.device)


def _cpu_out_reference(
    inp, *, bins, range=None, weight=None, density=False, out=None
):
    del out
    cpu_weight = weight.detach().to("cpu") if weight is not None else None
    cpu_out = torch.empty(tuple(bins), dtype=inp.dtype, device="cpu")
    torch.ops.aten._histogramdd_from_bin_cts.out(
        inp.detach().to("cpu"),
        bins,
        range=range,
        weight=cpu_weight,
        density=density,
        out=cpu_out,
    )
    return cpu_out.to(inp.device)


@pytest.mark.histogramdd_from_bin_cts
@pytest.mark.underscore_histogramdd_from_bin_cts
def test_histogramdd_from_bin_cts():
    bench = HistogramddFromBinCtsBenchmark(
        op_name="_histogramdd_from_bin_cts",
        torch_op=_cpu_reference,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.histogramdd_from_bin_cts_out
@pytest.mark.underscore_histogramdd_from_bin_cts_out
def test_histogramdd_from_bin_cts_out():
    bench = HistogramddFromBinCtsBenchmark(
        op_name="_histogramdd_from_bin_cts_out",
        torch_op=_cpu_out_reference,
        case_fn=_out_case_fn,
        build_inputs_fn=_out_build_inputs_fn,
        dtypes=[torch.float32],
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
