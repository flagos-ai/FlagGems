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


# -----------------------------
# Quant/Dequant helpers (naive)
# -----------------------------
def _quantize_per_tensor_symmetric(x: torch.Tensor, q_scale: float):
    """
    Naive symmetric per-tensor quantization to int8, then return (q_int8, q_scale).
    - x: fp tensor
    - q_scale: scalar float (must be > 0)
    """
    q = torch.clamp(torch.round(x / q_scale), -128, 127).to(torch.int8)
    return q, q_scale


def _dequantize_per_tensor(q: torch.Tensor, q_scale: float):
    """
    Naive per-tensor dequantization from int8 to fp32.
    """
    return q.float() * q_scale


# -----------------------------
# Baselines
# -----------------------------
def int8_gemm_naive_baseline_A(
    a_int8: torch.Tensor,
    w_int8: torch.Tensor,
    a_scale,
    w_scale,
    bias=None,
    out_dtype=torch.float16,
):
    """
    Baseline A:
      dequant(a,w) -> fp32 matmul -> +bias -> cast(out_dtype)
    """
    a = a_int8.float() * float(a_scale)
    w = w_int8.float() * (w_scale if torch.is_tensor(w_scale) else float(w_scale))
    out = a @ w
    if bias is not None:
        out = out + bias
    return out.to(out_dtype)


def int8_gemm_naive_baseline_B_quant_dequant(
    a_int8: torch.Tensor,
    w_int8: torch.Tensor,
    a_scale,
    w_scale,
    bias=None,
    out_dtype=torch.float16,
):
    """
    Baseline B:
      dequant(a,w) -> fp32 matmul -> +bias -> quant(int8) -> dequant(fp32) -> cast(out_dtype)
    """
    out = int8_gemm_naive_baseline_A(
        a_int8, w_int8, a_scale, w_scale, bias=bias, out_dtype=out_dtype
    )
    q, s = _quantize_per_tensor_symmetric(out.float(), 0.02)
    return _dequantize_per_tensor(q, s).to(out_dtype)


# -----------------------------
# Input generator
# -----------------------------
def int8_gemm_input_fn(m, n, k, out_dtype, device):
    # int8 inputs
    a = torch.randint(-128, 127, (m, k), dtype=torch.int8, device=device)
    w = torch.randint(-128, 127, (k, n), dtype=torch.int8, device=device)

    # scales
    a_scale = 0.02

    if Config.bench_level == consts.BenchLevel.COMPREHENSIVE and not Config.query:
        # per-channel
        w_scale = torch.rand((n,), device=device, dtype=torch.float32) * 0.05 + 0.001
    else:
        # scalar
        w_scale = 0.03

    # Optional bias
    if Config.bench_level == consts.BenchLevel.COMPREHENSIVE and not Config.query:
        bias = torch.randn((n,), device=device, dtype=torch.float32)
    else:
        bias = None

    # IMPORTANT: yield exactly the args that BOTH baseline and flag_gems.int8_gemm accept
    # (a, w, a_scale, w_scale, bias, out_dtype) -> 6 positional args
    yield a, w, a_scale, w_scale, bias, out_dtype


# -----------------------------
# Benchmark class
# -----------------------------
class Int8GemmBenchmark(base.Benchmark):
    """
    Benchmark for custom int8_gemm API:
      int8_gemm(a_int8, w_int8, a_scale, w_scale, bias=None, out_dtype=fp16/fp32)
    """

    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]
    DEFAULT_DTYPES = [torch.float16, torch.float32]
    DEFAULT_SHAPE_DESC = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 4096, 11008),
            (16, 4096, 11008),
            (64, 4096, 11008),
            (256, 4096, 11008),
            (2048, 4096, 11008),
            (64, 11008, 4096),
        ]
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def set_more_shapes(self):
        return [
            (128, 128, 128),
            (256, 512, 1024),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 1024),
        ]

    def get_input_iter(self, cur_dtype):
        for m, n, k in self.shapes:
            yield from int8_gemm_input_fn(m, n, k, cur_dtype, self.device)

    def get_tflops(self, op, *args, **kwargs):
        # matmul flops: 2*M*N*K
        a_int8 = args[0]
        w_int8 = args[1]
        M = a_int8.shape[0]
        K = a_int8.shape[1]
        N = w_int8.shape[1]
        return M * N * K * 2


# -----------------------------
# Benchmarks
# -----------------------------
@pytest.mark.int8_gemm
def test_int8_gemm_benchmark_vs_baseline_A():
    """
    Compare flag_gems.int8_gemm vs Baseline A:
      dequant -> matmul -> (+bias) -> cast
    """
    bench = Int8GemmBenchmark(
        op_name="int8_gemm",
        torch_op=int8_gemm_naive_baseline_A,
        gems_op=flag_gems.ops.int8_gemm,
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()


@pytest.mark.int8_gemm
def test_int8_gemm_benchmark_vs_baseline_B_quant_dequant():
    """
    Compare flag_gems.int8_gemm vs Baseline B:
      dequant -> matmul -> (+bias) -> quant(int8) -> dequant -> cast
    """
    bench = Int8GemmBenchmark(
        op_name="int8_gemm",
        torch_op=int8_gemm_naive_baseline_B_quant_dequant,
        gems_op=flag_gems.ops.int8_gemm,
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()
