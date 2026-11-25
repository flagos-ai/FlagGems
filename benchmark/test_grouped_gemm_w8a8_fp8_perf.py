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

"""Compare compact FP8 grouped GEMM with DeepGEMM's native SM90 FP8 path.

Use --shape_file with a JSON containing cases=[{shape:[M,N,K,E], count:...}]
for a captured shape workload. Counts describe invocation frequency; tensors and
near-uniform expert counts are regenerated. INT8 captures are shape reuse only.
The reference times only prepacked native FP8 GEMM. Device routing, padding and
FP32 scale conversion happen before timing; unpadding is excluded. The candidate
uses compact inputs and includes its task preparation. Inputs are frozen per case.
Validated with upstream DeepGEMM v2.1.1.post3 (CUDA 12.8, SM90).
"""

import json
from pathlib import Path

import pytest
import torch

import flag_gems
from tests.test_grouped_gemm_w8a8_fp8 import make_case

from . import base, consts
from .grouped_gemm_fp8_reference import PrepackedDeepGemmBaseline


class GroupedFp8Benchmark(base.Benchmark):
    DEFAULT_SHAPES = [
        (m, n, k, 256)
        for m in (8, 512, 30728, 131072)
        for n, k in ((256, 6144), (6144, 128))
    ]
    DEFAULT_SHAPE_DESC = "M,N,K,E; prepacked native FP8, adaptation excluded"

    def set_shapes(self, shape_file_path=None):
        path = Path(shape_file_path or base.Config.shape_file)
        if path.suffix == ".json":
            self.rows = json.loads(path.read_text())["cases"]
            for row in self.rows:
                m, n, k, e = row["shape"]
                if (
                    m <= 0
                    or n <= 0
                    or n % 64
                    or k <= 0
                    or k % 128
                    or e <= 0
                    or row["count"] <= 0
                ):
                    raise ValueError("Invalid grouped FP8 workload shape or count")
            self.shapes = [tuple(row["shape"]) for row in self.rows]
            self.shape_desc = self.DEFAULT_SHAPE_DESC
        else:
            super().set_shapes(shape_file_path)
            self.rows = [dict(shape=list(shape), count=1) for shape in self.shapes]
        self.native = None

    def get_case_iter(self, dtype):
        for index, row in enumerate(self.rows):
            yield self._case_from_plan(
                dtype,
                index,
                consts.BenchmarkCasePlan(
                    shape=row["shape"],
                    params=dict(
                        source_count=row["count"],
                        input_dtype="float8_e4m3fn",
                        routing="regenerated near-uniform counts",
                        scale_layout="FP32 A[M,K/128], B[E,ceil(N/128),K/128]",
                        baseline="Prepacked DeepGEMM native FP8; packing/unpacking excluded",
                    ),
                ),
            )

    def build_inputs(self, case):
        m, n, k, e = case.shape
        args = make_case([m // e + (i < m % e) for i in range(e)], n, k)[0]
        self.native = PrepackedDeepGemmBaseline(args)
        return args

    def get_input_iter(self, dtype):
        for case in self.get_case_iter(dtype):
            yield self.build_inputs(case)

    def native_reference(self, *args):
        if self.native is None or any(
            x is not y for x, y in zip(args, self.native.args)
        ):
            self.native = PrepackedDeepGemmBaseline(args)
        return self.native()

    def get_tflops(self, op, *args, **kwargs):
        return 2 * args[0].shape[0] * args[0].shape[1] * args[2].shape[1]


@pytest.mark.grouped_gemm_w8a8_fp8
def test_grouped_gemm_w8a8_fp8_perf():
    if flag_gems.vendor_name != "nvidia" or not hasattr(
        flag_gems, "grouped_gemm_w8a8_fp8"
    ):
        pytest.skip("Requires the NVIDIA Hopper FP8 grouped GEMM backend")
    pytest.importorskip("deep_gemm")
    bench = GroupedFp8Benchmark(
        op_name="grouped_gemm_w8a8_fp8", torch_op=None, dtypes=[torch.float8_e4m3fn]
    )
    bench.torch_op = bench.native_reference
    bench.set_gems(flag_gems.grouped_gemm_w8a8_fp8)
    bench.run()
