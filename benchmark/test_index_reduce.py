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
from .generated_operator_utils import OperatorBenchmark


REDUCTIONS = ("prod", "mean", "amax", "amin")
DEFAULT_SHAPES = [(1024, 1024), (4096, 256), (64, 512, 256)]

_ASCEND_NATIVE_BASELINE_SKIP = pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason=(
        "missing direct native kernel: schema=aten::index_reduce.out, "
        "vendor=ascend, device=npu, dispatch_key=PrivateUse1; "
        "torch-npu falls back to CPU"
    ),
)
_TSINGMICRO_SKIP = pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)


class IndexReduceBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=DEFAULT_SHAPES)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        return [(2048, 2048), (128, 1024, 512)]

    def get_gbps(self, args, latency):
        inp, _, index, source = args[:4]
        io_amount = sum(
            shape_utils.size_in_bytes(item) for item in (inp, index, source, inp)
        )
        return io_amount * 1e-9 / (latency * 1e-3)


def _case_fn_factory(reduce, inplace=False):
    def _case_fn(shape, dtype):
        del dtype
        dims = tuple(shape)
        dim = 0 if len(dims) == 1 else 1
        source_shape = list(dims)
        source_shape[dim] = max(1, source_shape[dim] // 2)
        yield base.BenchmarkCasePlan(
            shape={"self": list(dims), "index": [source_shape[dim]], "source": source_shape},
            params={"dim": dim, "reduce": reduce, "include_self": True, "inplace": inplace},
            builder_args=(dims, dim, tuple(source_shape), reduce),
        )

    return _case_fn


def _build_inputs_fn(plan, dtype, device):
    shape, dim, source_shape, reduce = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    index_len = source_shape[dim]
    index = torch.randperm(shape[dim], device=device)[:index_len]
    source = (
        torch.ones(source_shape, dtype=dtype, device=device)
        if reduce == "prod"
        else torch.randn(source_shape, dtype=dtype, device=device)
    )
    return inp, dim, index, source, {"reduce": reduce}


def _build_functional_inputs_fn(plan, dtype, device):
    shape, dim, source_shape, reduce = plan.builder_args
    inp = torch.randn(shape, dtype=dtype, device=device)
    index_len = source_shape[dim]
    index = torch.randperm(shape[dim], device=device)[:index_len]
    source = (
        torch.ones(source_shape, dtype=dtype, device=device)
        if reduce == "prod"
        else torch.randn(source_shape, dtype=dtype, device=device)
    )
    return inp, dim, index, source, {"reduce": reduce}


@pytest.mark.index_reduce
@_ASCEND_NATIVE_BASELINE_SKIP
@_TSINGMICRO_SKIP
@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_index_reduce(reduce):
    bench = IndexReduceBenchmark(
        op_name="index_reduce",
        torch_op=torch.index_reduce,
        case_fn=_case_fn_factory(reduce),
        build_inputs_fn=_build_functional_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


# Keep the shared catalog family benchmark after splitting the spreadsheet
# functional and out definitions into exact files.
@pytest.mark.index_reduce_
@_ASCEND_NATIVE_BASELINE_SKIP
@_TSINGMICRO_SKIP
@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_index_reduce_inplace_shared_family(reduce):
    bench = IndexReduceBenchmark(
        op_name="index_reduce_",
        torch_op=torch.Tensor.index_reduce_,
        case_fn=_case_fn_factory(reduce, inplace=True),
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
