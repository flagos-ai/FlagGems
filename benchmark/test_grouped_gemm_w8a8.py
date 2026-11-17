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

from pathlib import Path

import pytest
import torch
import yaml

import flag_gems
from tests.test_grouped_gemm_w8a8 import (
    HAS_GROUPED_FP8,
    deepgemm_grouped_gemm,
    grouped_gemm_mark,
    make_grouped_fp8_case,
    make_grouped_gemm_case,
    mate_grouped_fp8,
    requires_grouped_gemm,
)

from .base import Benchmark


class GroupedGemmBenchmark(Benchmark):
    DEFAULT_SHAPES = [(45, 512, 6144, 257), (45, 6144, 256, 257)]
    DEFAULT_SHAPE_DESC = "M,N,K,E"
    input_fn = staticmethod(make_grouped_gemm_case)

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            m, n, k, e = shape[:4]
            if len(shape) == 5:
                lengths = shape[4]
            else:
                lengths = [m // e + (expert < m % e) for expert in range(e)]
            if (
                len(lengths) != e
                or sum(lengths) != m
                or any(length < 0 for length in lengths)
            ):
                raise ValueError("expert row lengths must match M and E")
            yield self.input_fn(lengths, n, k, self.device)


class GroupedGemmInt8Benchmark(GroupedGemmBenchmark):
    pass


class GroupedGemmFp8Benchmark(GroupedGemmBenchmark):
    DEFAULT_SHAPE_FILES = Path(__file__).resolve().with_name("core_shapes.yaml")
    input_fn = staticmethod(make_grouped_fp8_case)

    def set_shapes(self, shape_file_path):
        if Path(shape_file_path).resolve() != self.DEFAULT_SHAPE_FILES:
            return super().set_shapes(shape_file_path)

        with self.DEFAULT_SHAPE_FILES.open() as source:
            entry = yaml.safe_load(source)["grouped_gemm_w8a8"]
        self.shapes = [tuple(shape) for shape in entry["shapes"]]
        self.shape_desc = entry["shape_desc"]


@requires_grouped_gemm
@grouped_gemm_mark
def test_grouped_gemm_w8a8():
    if HAS_GROUPED_FP8:
        pytest.importorskip("mate.gemm")
        benchmark = GroupedGemmFp8Benchmark
        native, op = mate_grouped_fp8, flag_gems.grouped_gemm_w8a8_fp8
    else:
        pytest.importorskip("deep_gemm")
        benchmark = GroupedGemmInt8Benchmark
        native, op = deepgemm_grouped_gemm, flag_gems.grouped_gemm_w8a8
    bench = benchmark(
        op_name=grouped_gemm_mark.name,
        torch_op=native,
        gems_op=op,
        dtypes=[torch.bfloat16],
    )
    bench.run()
