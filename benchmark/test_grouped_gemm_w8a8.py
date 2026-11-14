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
import yaml

import flag_gems
from tests.test_grouped_gemm_w8a8 import (
    deepgemm_grouped_gemm,
    make_grouped_fp8_case,
    make_grouped_gemm_case,
    mate_grouped_fp8,
)

from .base import Benchmark
from .test_blas_perf_parallel import ParallelBenchmarkMixin


class GroupedGemmBenchmark(ParallelBenchmarkMixin, Benchmark):
    RECORD_PARALLEL_RESULTS = True
    DEFAULT_SHAPES = [(45, 512, 6144, 257), (45, 6144, 256, 257)]
    DEFAULT_SHAPE_DESC = "M,N,K,E"

    def __init__(self, *args, input_fn=make_grouped_gemm_case, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def _get_shape_config_keys(self):
        return list(dict.fromkeys((self.op_name, "grouped_gemm_w8a8")))

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Both precisions share the core catalog; explicit precision-specific exports take priority.
        with open(shape_file_path) as source:
            config = yaml.safe_load(source)
        if self.op_name not in config and "grouped_gemm_w8a8" in config:
            entry = config["grouped_gemm_w8a8"]
            self.shapes = [tuple(shape) for shape in entry["shapes"]]
            self.shape_desc = entry.get("shape_desc", self.DEFAULT_SHAPE_DESC)

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


@pytest.mark.grouped_gemm_w8a8
@pytest.mark.skipif(
    "grouped_gemm_w8a8" not in vars(flag_gems),
    reason="requires a registered grouped INT8 GEMM implementation",
)
def test_grouped_gemm_w8a8():
    pytest.importorskip("deep_gemm")
    bench = GroupedGemmBenchmark(
        op_name="grouped_gemm_w8a8",
        torch_op=deepgemm_grouped_gemm,
        gems_op=flag_gems.grouped_gemm_w8a8,
        dtypes=[torch.bfloat16],
    )
    bench.run()


@pytest.mark.grouped_gemm_w8a8_fp8
@pytest.mark.skipif(
    "grouped_gemm_w8a8_fp8" not in vars(flag_gems),
    reason="requires a registered unpadded grouped FP8 GEMM backend",
)
def test_grouped_gemm_w8a8_fp8():
    pytest.importorskip("mate.gemm")
    bench = GroupedGemmBenchmark(
        op_name="grouped_gemm_w8a8_fp8",
        torch_op=mate_grouped_fp8,
        gems_op=flag_gems.grouped_gemm_w8a8_fp8,
        input_fn=make_grouped_fp8_case,
        dtypes=[torch.bfloat16],
    )
    bench.run()
