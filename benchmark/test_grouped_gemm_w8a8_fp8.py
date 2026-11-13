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

from collections.abc import Generator

import pytest
import torch

import flag_gems
from tests.test_grouped_gemm_w8a8_fp8 import make_grouped_fp8_case, mate_grouped_fp8

from .base import Benchmark
from .test_blas_perf_parallel import ParallelBenchmarkMixin


class GroupedFP8Benchmark(ParallelBenchmarkMixin, Benchmark):
    """Accept (M, N, K, E[, expert_lengths]); M is the sum of expert_lengths."""

    RECORD_PARALLEL_RESULTS = True
    DEFAULT_SHAPES = [(45, 256, 6144, 256), (45, 6144, 128, 256)]
    DEFAULT_SHAPE_DESC = "M,N,K,E (unpadded rows)"

    def get_input_iter(self, cur_dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            if len(shape) == 5:
                m, n, k, experts, lengths = shape
                if (
                    len(lengths) != experts
                    or sum(lengths) != m
                    or any(length < 0 for length in lengths)
                ):
                    raise ValueError(
                        "expert row lengths must be nonnegative and match M and E"
                    )
            else:
                m, n, k, experts = shape
                lengths = [
                    m // experts + (expert < m % experts) for expert in range(experts)
                ]
            yield make_grouped_fp8_case(lengths, n, k, self.device)


@pytest.mark.grouped_gemm_w8a8_fp8
@pytest.mark.skipif(
    "grouped_gemm_w8a8_fp8" not in vars(flag_gems),
    reason="requires a registered unpadded grouped FP8 GEMM backend",
)
def test_grouped_gemm_w8a8_fp8():
    pytest.importorskip("mate.gemm")
    bench = GroupedFP8Benchmark(
        op_name="grouped_gemm_w8a8_fp8",
        torch_op=mate_grouped_fp8,
        gems_op=flag_gems.grouped_gemm_w8a8_fp8,
        dtypes=[torch.bfloat16],
    )
    bench.run()
