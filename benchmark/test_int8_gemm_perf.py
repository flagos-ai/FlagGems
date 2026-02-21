# benchmark/test_int8_gemm_perf.py

import pytest
import torch

# import flag_gems
import flag_gems.ops
from flag_gems.ops import int8_gemm as gems_int8_gemm
from benchmark.attri_util import DEFAULT_METRICS, BenchLevel
from benchmark.conftest import Config
from benchmark.performance_utils import Benchmark


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
    out_qscale: float = 0.02,
):
    """
    Baseline B:
      dequant(a,w) -> fp32 matmul -> +bias -> quant(int8) -> dequant(fp32) -> cast(out_dtype)
    """
    a = a_int8.float() * float(a_scale)
    w = w_int8.float() * (w_scale if torch.is_tensor(w_scale) else float(w_scale))
    out = a @ w
    if bias is not None:
        out = out + bias

    q, s = _quantize_per_tensor_symmetric(out, float(out_qscale))
    out = _dequantize_per_tensor(q, float(s))
    return out.to(out_dtype)


# -----------------------------
# Benchmark class
# -----------------------------
class Int8GemmBenchmark(Benchmark):
    """
    Benchmark for custom int8_gemm API:
      int8_gemm(a_int8, w_int8, a_scale, w_scale, bias=None, out_dtype=fp16/fp32)

    IMPORTANT:
    - Base Benchmark.init_user_config() loads shapes from YAML (often M,N).
      That breaks MNK unpacking.
    - So we override init_user_config() to KEEP MNK shapes and NOT read YAML.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]
    DEFAULT_DTYPES = [torch.float16, torch.float32]

    DEFAULT_SHAPES = [(256, 256, 256)]
    DEFAULT_SHAPE_DESC = "M, N, K"

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def init_user_config(self):
        # keep base behaviors for mode/dtypes/metrics, but DO NOT load shapes from YAML
        self.mode = Config.mode
        self.set_dtypes(Config.user_desired_dtypes)
        self.set_metrics(Config.user_desired_metrics)

        if (
            hasattr(self, "set_more_shapes")
            and callable(getattr(self, "set_more_shapes"))
            and Config.bench_level == BenchLevel.COMPREHENSIVE
            and not Config.query
        ):
            more = self.set_more_shapes()
            if more:
                self.shapes = list(dict.fromkeys(list(self.shapes) + list(more)))

    def set_more_shapes(self):
        return [
            (128, 128, 128),
            (256, 512, 1024),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 1024),
        ]

    def get_input_iter(self, cur_dtype):
        for (m, n, k) in self.shapes:
            yield from self.input_fn(m, n, k, cur_dtype, self.device)

    def get_tflops(self, op, *args, **kwargs):
        # matmul flops: 2*M*N*K
        a_int8 = args[0]
        w_int8 = args[1]
        M = a_int8.shape[0]
        K = a_int8.shape[1]
        N = w_int8.shape[1]
        return M * N * K * 2

    def get_gbps(self, args, latency=None):
        raise NotImplementedError


# -----------------------------
# Input generator
# -----------------------------
def int8_gemm_input_fn(m, n, k, out_dtype, device):
    # int8 inputs
    a = torch.randint(-128, 127, (m, k), dtype=torch.int8, device=device)
    w = torch.randint(-128, 127, (k, n), dtype=torch.int8, device=device)

    # scales
    a_scale = 0.02

    if Config.bench_level == BenchLevel.COMPREHENSIVE and not Config.query:
        # per-channel
        w_scale = torch.rand((n,), device=device, dtype=torch.float32) * 0.05 + 0.001
    else:
        # scalar
        w_scale = 0.03

    # Optional bias
    if Config.bench_level == BenchLevel.COMPREHENSIVE and not Config.query:
        bias = torch.randn((n,), device=device, dtype=torch.float32)
    else:
        bias = None

    # IMPORTANT: yield exactly the args that BOTH baseline and flag_gems.int8_gemm accept
    # (a, w, a_scale, w_scale, bias, out_dtype) -> 6 positional args
    yield a, w, a_scale, w_scale, bias, out_dtype


# -----------------------------
# Wrappers (stable signature)
# -----------------------------
def _baseline_A_wrapper(a, w, a_scale, w_scale, bias, out_dtype):
    return int8_gemm_naive_baseline_A(
        a, w, a_scale, w_scale, bias=bias, out_dtype=out_dtype
    )


def _baseline_B_wrapper(a, w, a_scale, w_scale, bias, out_dtype):
    # baseline-B only: choose a fixed output quant scale for the extra quant/dequant
    return int8_gemm_naive_baseline_B_quant_dequant(
        a, w, a_scale, w_scale, bias=bias, out_dtype=out_dtype, out_qscale=0.02
    )


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
        torch_op=_baseline_A_wrapper,
        dtypes=[torch.float16, torch.float32],
        input_fn=int8_gemm_input_fn,
    )
    bench.set_gems(gems_int8_gemm)
    bench.run()


@pytest.mark.int8_gemm
def test_int8_gemm_benchmark_vs_baseline_B_quant_dequant():
    """
    Compare flag_gems.int8_gemm vs Baseline B:
      dequant -> matmul -> (+bias) -> quant(int8) -> dequant -> cast
    """
    bench = Int8GemmBenchmark(
        op_name="int8_gemm",
        torch_op=_baseline_B_wrapper,
        dtypes=[torch.float16, torch.float32],
        input_fn=int8_gemm_input_fn,
    )
    bench.set_gems(gems_int8_gemm)
    bench.run()