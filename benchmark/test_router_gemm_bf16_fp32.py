"""Benchmark for router_gemm_bf16_fp32 (DeepSeek V4 MoE routing GEMM).

Shapes match DeepSeek V4 production config:
    - N=384: number of experts in DeepSeek V4 MoE layer
    - K=7168: hidden dimension size
    - M=1-4096: batch size (tokens)
      - M=1-8: single/few token decode (latency-critical)
      - M=16-32: typical decode batch
      - M=64-256: small prefill
      - M=512-4096: large prefill (throughput-critical)

The torch reference uses torch.matmul (cuBLAS), representing the
optimized vendor BLAS baseline.
"""

import pytest
import torch

import flag_gems
from flag_gems.fused import router_gemm_bf16_fp32

from . import base

device = flag_gems.device

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)


class RouterGemmBF16FP32Benchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        # DeepSeek V4 production shapes:
        # - M=1: single token decode (latency-critical, e.g. chat/completion)
        # - M=2-8: speculative decoding batch
        # - M=16-32: typical decode batch size
        # - M=64-128: small prefill (short prompt)
        # - M=256-512: medium prefill
        # - M=1024-4096: large prefill (long context)
        # N=384: number of experts, K=7168: hidden dimension
        self.shapes = [
            (1, 384, 7168),
            (2, 384, 7168),
            (4, 384, 7168),
            (8, 384, 7168),
            (16, 384, 7168),
            (32, 384, 7168),
            (64, 384, 7168),
            (128, 384, 7168),
            (256, 384, 7168),
            (512, 384, 7168),
            (1024, 384, 7168),
            (2048, 384, 7168),
            (4096, 384, 7168),
        ]

    def get_input_iter(self, dtype):
        for M, N, K in self.shapes:
            input = torch.randn(M, K, device=device, dtype=torch.bfloat16)
            weight = torch.randn(N, K, device=device, dtype=torch.bfloat16)

            yield {
                "input": input,
                "weight": weight,
            }

    def get_gems_op(self):
        return router_gemm_bf16_fp32

    def get_torch_op(self):
        def torch_matmul_ref(input, weight):
            # cuBLAS GEMM baseline: input @ weight.T
            return torch.matmul(input.float(), weight.T.float())

        return torch_matmul_ref


@pytest.mark.router_gemm_bf16_fp32
def test_perf_router_gemm_bf16_fp32():
    bench = RouterGemmBF16FP32Benchmark(
        op_name="router_gemm_bf16_fp32",
        torch_op=None,
        dtypes=[torch.bfloat16],
    )
    bench.run()
