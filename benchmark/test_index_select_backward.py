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

SHAPES = [(16, 64), (32, 128), (64, 256), (16, 16, 64), (32, 32, 128)]


class IndexSelectBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES

    def set_more_metrics(self):
        return ["gbps"]


def _case_fn(shape, dtype):
    del dtype
    for dim in range(len(shape)):
        self_sizes = list(shape)
        self_sizes[dim] += 8
        yield base.BenchmarkCasePlan(
            shape={"grad": list(shape), "output": self_sizes, "index": [shape[dim]]},
            params={"dim": dim},
            builder_args=(tuple(shape), tuple(self_sizes), dim),
        )


def _build_inputs_fn(plan, dtype, device):
    if flag_gems.vendor_name == "enflame":
        pytest.skip("Enflame native index_select_backward requires Int32 indices")
    torch.manual_seed(0)
    shape, self_sizes, dim = plan.builder_args
    grad = torch.randn(shape, dtype=dtype, device=device)
    index = torch.randint(0, self_sizes[dim], (shape[dim],), device=device)
    return grad, self_sizes, dim, index


def _get_gbps(args, latency):
    grad, self_sizes, _, index = args
    out = torch.empty(self_sizes, dtype=grad.dtype, device=grad.device)
    amount = sum(shape_utils.size_in_bytes(x) for x in (grad, index, out))
    return amount * 1e-9 / (latency * 1e-3)


@pytest.mark.index_select_backward
def test_index_select_backward():
    bench = IndexSelectBackwardBenchmark(
        op_name="index_select_backward",
        torch_op=torch.ops.aten.index_select_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )
    bench.run()


def _case_fn_int32(shape, dtype):
    for plan in _case_fn(shape, dtype):
        yield base.BenchmarkCasePlan(
            shape=plan.shape,
            params={**plan.params, "index_dtype": "torch.int32"},
            builder_args=plan.builder_args,
        )


def _build_inputs_int32_fn(plan, dtype, device):
    torch.manual_seed(0)
    shape, self_sizes, dim = plan.builder_args
    grad = torch.randn(shape, dtype=dtype, device=device)
    index_device = "cpu" if flag_gems.vendor_name == "enflame" else device
    index = torch.randint(
        0, self_sizes[dim], (shape[dim],), dtype=torch.int32, device=index_device
    ).to(device)
    assert index.dtype == torch.int32
    assert index.device == grad.device
    return grad, self_sizes, dim, index


@pytest.mark.index_select_backward
def test_index_select_backward_int32():
    bench = IndexSelectBackwardBenchmark(
        op_name="index_select_backward",
        torch_op=torch.ops.aten.index_select_backward,
        case_fn=_case_fn_int32,
        build_inputs_fn=_build_inputs_int32_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )
    bench.run()
