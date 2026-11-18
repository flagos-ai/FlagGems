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


class TensorSelectBenchmark(base.GenericBenchmark2DOnly):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # Speed Up Benchmark Test, Big Shape Will Cause Timeout
        if flag_gems.vendor_name == "kunlunxin":
            return []

        shapes = super().set_more_shapes()
        shapes = [
            # this filter is for scatter
            shape
            for shape in shapes
            if len(shape) == 2 and shape[0] > 16 and shape[1] > 16
        ]
        return shapes


def gather_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)

    dim = -1
    size_dim = shape[dim]
    index_shape = list(shape)
    index_shape[dim] = 2 * shape[dim]
    index = torch.randint(0, size_dim, index_shape, dtype=torch.long, device=device)
    yield inp, dim, index


def _get_gbps(bench_fn_args, latency):
    inp, dim, index = bench_fn_args[:3]
    data_shape = list(inp.shape)
    data_shape[dim] = index.shape[dim]
    data = torch.empty(data_shape, dtype=inp.dtype, device=inp.device)
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [index, data, data]])
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.gather
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_gather():
    bench = TensorSelectBenchmark(
        op_name="gather",
        torch_op=torch.gather,
        input_fn=gather_input_fn,
        get_gbps=_get_gbps,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()



def gather_backward_case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    index_shape = (*shape[:-1], 2 * shape[-1])
    yield base.BenchmarkCasePlan(
        shape={"grad": list(index_shape), "self": list(shape), "index": list(index_shape)},
        params={"dim": -1, "sparse_grad": False},
        builder_args=(shape,),
    )


def build_gather_backward_inputs(plan, dtype, device):
    (shape,) = plan.builder_args
    inp, dim, index = next(gather_input_fn(shape, dtype, device))
    # Match the random upstream gradient previously built by the autograd timer.
    grad = torch.randn(index.shape, dtype=dtype, device=device)
    return grad, inp, dim, index, False


def _get_gather_backward_gbps(bench_fn_args, latency):
    _, inp, dim, index, _ = bench_fn_args
    return _get_gbps((inp, dim, index), latency)


@pytest.mark.gather_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_gather_backward():
    bench = TensorSelectBenchmark(
        op_name="gather_backward",
        torch_op=torch.ops.aten.gather_backward.default,
        case_fn=gather_backward_case_fn,
        build_inputs_fn=build_gather_backward_inputs,
        get_gbps=_get_gather_backward_gbps,
        dtypes=[torch.float32, torch.float16],
    )

    bench.run()
