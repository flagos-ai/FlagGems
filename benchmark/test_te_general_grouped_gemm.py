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

"""
Benchmark for te_general_grouped_gemm operator.
"""

import pytest
import torch

from . import base, consts


def _torch_grouped_gemm(
    A_list,
    transa,
    B_list,
    transb,
    out_list,
):
    """Baseline implementation using PyTorch matmul."""
    for i in range(len(A_list)):
        A_mat = A_list[i].T if transa else A_list[i]
        B_mat = B_list[i].T if transb else B_list[i]
        torch.matmul(A_mat, B_mat, out=out_list[i])
    return out_list


def _flaggems_grouped_gemm(
    A_list,
    transa,
    B_list,
    transb,
    out_list,
):
    """FlagGems implementation."""
    from flag_gems.ops.te_general_grouped_gemm import te_general_grouped_gemm

    return te_general_grouped_gemm(A_list, transa, B_list, transb, out_list)


class TeGeneralGroupedGemmBenchmark(base.Benchmark):
    """
    Benchmark for te_general_grouped_gemm operation.

    This operator performs multiple independent GEMMs in parallel,
    commonly used in Mixture-of-Experts (MoE) models.
    """

    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]
    DEFAULT_DTYPES = [torch.float16, torch.bfloat16]

    def __init__(self):
        super().__init__(
            op_name="te_general_grouped_gemm",
            torch_op=_torch_grouped_gemm,
            dtypes=self.DEFAULT_DTYPES,
        )
        self.set_gems(_flaggems_grouped_gemm)

    def set_shapes(self, shape_file_path=None):
        """Set benchmark shapes: (num_gemms, M, N, K)."""
        self.shapes = [
            # Small shapes
            (2, 32, 64, 128),
            (4, 64, 128, 256),
            # Medium shapes
            (4, 256, 512, 256),
            (8, 512, 512, 512),
            # Large shapes (MoE-like)
            (8, 1024, 1024, 1024),
        ]

        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes = list(dict.fromkeys(self.shapes + self.set_more_shapes()))

        self.shape_desc = "num_gemms, M, N, K"

    def set_more_shapes(self):
        """Additional shapes for comprehensive benchmark."""
        return [
            # Larger MoE shapes
            (16, 2048, 2048, 4096),
            # Expert-parallel shapes (typical MoE configurations)
            (8, 4096, 14336, 4096),  # Llama-like FFN hidden
            (8, 4096, 4096, 14336),
        ]

    def get_input_iter(self, cur_dtype):
        """Generate inputs for benchmark.

        Uses TN layout: A is (K, M), B is (K, N), C is (M, N)
        """
        for num_gemms, m, n, k in self.shapes:
            A_list = [
                torch.randn(k, m, dtype=cur_dtype, device=self.device)
                for _ in range(num_gemms)
            ]
            B_list = [
                torch.randn(k, n, dtype=cur_dtype, device=self.device)
                for _ in range(num_gemms)
            ]
            out_list = [
                torch.zeros(m, n, dtype=cur_dtype, device=self.device)
                for _ in range(num_gemms)
            ]

            yield A_list, True, B_list, False, out_list

    def record_shapes(self, *args, **kwargs):
        """Record shapes in a concise format: (num_gemms, M, N, K)."""
        A_list = args[0]
        out_list = args[4]

        num_gemms = len(A_list)
        K = A_list[0].shape[0]
        M = A_list[0].shape[1]
        N = out_list[0].shape[1]

        return (num_gemms, M, N, K)

    def get_tflops(self, op, *args, **kwargs):
        """Calculate total TFLOPS for the grouped GEMM.

        For grouped GEMM: total_flops = num_gemms * M * N * 2 * K
        """
        A_list = args[0]
        out_list = args[4]

        num_gemms = len(A_list)
        # A is (K, M) for TN layout
        K = A_list[0].shape[0]
        M = A_list[0].shape[1]
        # out is (M, N)
        N = out_list[0].shape[1]

        total_flops = num_gemms * M * N * 2 * K
        return total_flops


@pytest.mark.te_general_grouped_gemm
def test_te_general_grouped_gemm():
    """Benchmark te_general_grouped_gemm performance."""
    bench = TeGeneralGroupedGemmBenchmark()
    bench.run()
