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
from flag_gems.utils import shape_utils

from . import base, consts

pytestmark = [
    pytest.mark.skipif(
        flag_gems.vendor_name == "ascend",
        reason="CANN scatter_reduce falls back to CPU, so NPU events are unavailable",
    ),
    pytest.mark.skipif(
        flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
    ),
]

REDUCE_MODES = ("sum", "prod", "mean", "amax", "amin")
# None exercises the schema default by omitting the include_self keyword.
INCLUDE_SELF_CASES = (None, False)


class TensorSelectBenchmark(base.GenericBenchmark2DOnly):
    """Benchmark scatter-style ATen operators with tensor index inputs."""

    def set_more_metrics(self):
        """Add effective memory bandwidth to comprehensive benchmark runs."""
        return ["gbps"]

    def set_more_shapes(self):
        """Keep comprehensive aten scatter_reduce shapes bounded and two-dimensional."""
        if flag_gems.vendor_name == "kunlunxin":
            return []

        shapes = super().set_more_shapes()
        return [
            shape
            for shape in shapes
            if len(shape) == 2 and shape[0] > 16 and shape[1] > 16
        ]


def scatter_reduce_input_fn_factory(
    reduce, include_self, *, use_out=False, is_inplace=False
):
    """Build benchmark inputs for the three aten scatter_reduce overloads."""

    def inner(shape, dtype, device):
        """Yield one valid scatter_reduce argument set for a benchmark shape."""
        inp = torch.randn(shape, dtype=dtype, device=device)
        dim = -1
        index = torch.randint(0, shape[dim], shape, dtype=torch.long, device=device)
        if is_inplace and reduce == "sum":
            src = torch.zeros(shape, dtype=dtype, device=device)
        elif is_inplace and reduce == "prod":
            src = torch.ones(shape, dtype=dtype, device=device)
        else:
            src = torch.randn(shape, dtype=dtype, device=device)

        kwargs = {"reduce": reduce}
        if include_self is not None:
            kwargs["include_self"] = include_self
        if use_out:
            kwargs["out"] = torch.empty_like(inp)
        yield inp, dim, index, src, kwargs

    return inner


def gather_scatter_gbps(bench_fn_args, latency):
    """Estimate tensor traffic for aten scatter_reduce benchmark arguments."""
    inp, _, index, src = bench_fn_args[:4]
    io_amount = sum(shape_utils.size_in_bytes(item) for item in (inp, index, src, inp))
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("include_self", INCLUDE_SELF_CASES)
def test_scatter_reduce(reduce, include_self):
    """Benchmark aten::scatter_reduce.two for every supported reduction mode."""
    bench = TensorSelectBenchmark(
        op_name="scatter_reduce.two",
        torch_op=torch.ops.aten.scatter_reduce.two,
        input_fn=scatter_reduce_input_fn_factory(reduce, include_self),
        get_gbps=gather_scatter_gbps,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.scatter_reduce_two_
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("include_self", INCLUDE_SELF_CASES)
def test_scatter_reduce_(reduce, include_self):
    """Benchmark aten::scatter_reduce_.two for every supported reduction mode."""
    bench = TensorSelectBenchmark(
        op_name="scatter_reduce_.two",
        torch_op=torch.ops.aten.scatter_reduce_.two,
        input_fn=scatter_reduce_input_fn_factory(reduce, include_self, is_inplace=True),
        get_gbps=gather_scatter_gbps,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_two_out
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("include_self", INCLUDE_SELF_CASES)
def test_scatter_reduce_out(reduce, include_self):
    """Benchmark aten::scatter_reduce.two_out for every supported reduction mode."""
    bench = TensorSelectBenchmark(
        op_name="scatter_reduce.two_out",
        torch_op=torch.ops.aten.scatter_reduce.two_out,
        input_fn=scatter_reduce_input_fn_factory(reduce, include_self, use_out=True),
        get_gbps=gather_scatter_gbps,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
