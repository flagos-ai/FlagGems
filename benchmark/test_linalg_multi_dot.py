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
from .conftest import Config

CORE_CHAINS = [
    (64, 256, 128),
    (256, 64, 512),
    (128, 512, 64, 256),
    (512, 64, 256, 128),
    (128, 256, 64, 512, 128),
    (256, 128, 512, 64, 256),
]

COMPREHENSIVE_CHAINS = [
    (1024, 64, 1024),
    (2048, 32, 2048, 32),
    (512, 2048, 64, 1024),
    (1024, 128, 2048, 64, 512),
    (256, 1024, 64, 2048, 128, 512),
    (512, 128, 1024, 64, 2048, 32, 512),
]


class LinalgMultiDotBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_CHAINS)
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(COMPREHENSIVE_CHAINS)
        self.shape_desc = "matrix-chain dimensions (d0, d1, ..., dn)"

    def get_input_iter(self, cur_dtype):
        for dimensions in self.shapes:
            tensors = [
                torch.randn(
                    dimensions[index],
                    dimensions[index + 1],
                    dtype=cur_dtype,
                    device=self.device,
                )
                for index in range(len(dimensions) - 1)
            ]
            yield (tensors,)


class LinalgMultiDotOutBenchmark(LinalgMultiDotBenchmark):
    def get_input_iter(self, cur_dtype):
        for (tensors,) in super().get_input_iter(cur_dtype):
            out = torch.empty(
                tensors[0].shape[0],
                tensors[-1].shape[1],
                dtype=cur_dtype,
                device=self.device,
            )
            yield tensors, {"out": out}


@pytest.mark.linalg_multi_dot
def test_linalg_multi_dot_benchmark():
    bench = LinalgMultiDotBenchmark(
        op_name="linalg_multi_dot",
        torch_op=torch.ops.aten.linalg_multi_dot.default,
        gems_op=flag_gems.linalg_multi_dot,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.linalg_multi_dot_out
def test_linalg_multi_dot_out_benchmark():
    bench = LinalgMultiDotOutBenchmark(
        op_name="linalg_multi_dot_out",
        torch_op=torch.ops.aten.linalg_multi_dot.out,
        gems_op=flag_gems.linalg_multi_dot_out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
