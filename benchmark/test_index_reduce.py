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

from . import base, consts, utils


class IndexReduceBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [(1024, 1024), (4096, 256), (64, 512, 256)]
    DEFAULT_SHAPE_DESC = "(B), M, N"

    def __init__(self, *args, reduce, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduce = reduce

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        return [(2048, 2048), (128, 1024, 512)]

    def get_gbps(self, args, latency):
        inp = args[0]
        index = args[2]
        source = args[3]
        io_amount = sum(utils.size_in_bytes(item) for item in [inp, index, source, inp])
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=dtype, device=self.device)
            dim = 0 if len(shape) == 1 else 1
            source_shape = list(shape)
            index_len = max(1, source_shape[dim] // 2)
            source_shape[dim] = index_len
            index = torch.randperm(shape[dim], device=self.device)[:index_len]

            if self.reduce == "prod":
                source = torch.ones(source_shape, dtype=dtype, device=self.device)
            else:
                source = torch.randn(source_shape, dtype=dtype, device=self.device)

            yield inp, dim, index, source, {"reduce": self.reduce}


def _run_index_reduce_benchmark(reduce):
    bench = IndexReduceBenchmark(
        op_name=f"index_reduce_.{reduce}",
        torch_op=torch.Tensor.index_reduce_,
        dtypes=consts.FLOAT_DTYPES,
        reduce=reduce,
    )
    bench.run()


@pytest.mark.index_reduce_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_reduce_prod():
    _run_index_reduce_benchmark("prod")


@pytest.mark.index_reduce_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_reduce_mean():
    _run_index_reduce_benchmark("mean")


@pytest.mark.index_reduce_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_reduce_amax():
    _run_index_reduce_benchmark("amax")


@pytest.mark.index_reduce_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_reduce_amin():
    _run_index_reduce_benchmark("amin")
