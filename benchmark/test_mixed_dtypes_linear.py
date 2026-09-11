import pytest
import torch

import flag_gems

from . import base
from .consts import FLOAT_DTYPES, BenchmarkMetrics

# FP16/BF16 only: CUDA quantized matmul requires half-precision activation (no float32 support)
FP16_BF16_DTYPES = [d for d in FLOAT_DTYPES if d != torch.float32]


# LLM-scale shapes: (M, K, N, mode, has_bias, activation)
# where M = tokens, K = input features, N = output features, mode in ('int8', 'int4')
MIXED_DTYPES_LINEAR_SHAPES = [
    (1, 4096, 4096, "int8", True, "silu"),
    (1, 4096, 11008, "int8", True, "silu"),
    (1, 11008, 4096, "int8", True, "none"),
    (4, 4096, 4096, "int8", True, "silu"),
    (4, 4096, 11008, "int4", True, "silu"),
    (16, 4096, 4096, "int8", False, "relu"),
    (16, 4096, 11008, "int4", True, "silu"),
    (32, 4096, 4096, "int8", True, "none"),
    (64, 4096, 11008, "int8", True, "silu"),
    (128, 4096, 4096, "int4", True, "silu"),
]


class MixedDtypesLinearBenchmark(base.Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gems_op = lambda inp, w, s, b, act: flag_gems.mixed_dtypes_linear(
            inp, w, s, bias=b, activation=act
        )

    def set_shapes(self, shape_file_path=None):
        self.shapes = MIXED_DTYPES_LINEAR_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from mixed_dtypes_linear_input_fn(shape, cur_dtype, self.device)

    def _run_metric(self, input_item):
        metric = BenchmarkMetrics()
        inp, weight, scale, bias, activation = input_item
        metric.shape_detail = self.record_shapes(inp, weight, scale)
        try:
            if "latency_base" in self.to_bench_metrics:
                metric.latency_base = self.get_latency(
                    self.torch_op, inp, weight, scale, bias, activation
                )
            if "latency" in self.to_bench_metrics:
                metric.latency = self.get_latency(
                    self.gems_op, inp, weight, scale, bias, activation
                )
            if "speedup" in self.to_bench_metrics:
                metric.speedup = metric.latency_base / metric.latency
        except (RuntimeError, Exception) as e:
            metric.error_msg = str(e)
            pytest.fail(str(e))
        return metric


def mixed_dtypes_linear_input_fn(shape, dtype, device):
    """Yield tuples of (input, weight_uint8, scale, bias, activation)."""
    M, K, N, mode, has_bias, activation = shape
    inp = torch.randn((M, K), dtype=dtype, device=device)
    ncols = N if mode == "int8" else N // 2
    weight = torch.randint(0, 256, (K, ncols), dtype=torch.uint8, device=device)
    scale = torch.randn((N,), dtype=dtype, device=device)
    bias = torch.randn((N,), dtype=dtype, device=device) if has_bias else None
    yield (inp, weight, scale, bias, activation)


def mixed_dtypes_linear_torch(inp, weight, scale, bias, activation):
    """Torch baseline: dequantize uint8 weights and compute matmul.

    Matches kernel order of operations (dtype-split scale placement).
    """
    K, ncols = weight.shape
    N = scale.shape[0]
    if ncols == N:
        # int8 mode
        w = weight.to(torch.int32) - 128
    else:
        # int4 mode
        lo = (weight.to(torch.int32) & 0xF) - 8
        hi = ((weight.to(torch.int32) >> 4) & 0xF) - 8
        w = torch.stack([lo, hi], dim=-1).reshape(K, N)

    if inp.dtype == torch.float16:
        # fp16: epilogue scale
        wf = w.to(inp.dtype)
        acc = torch.matmul(inp.to(torch.float32), wf.to(torch.float32))
        acc = acc * scale.to(torch.float32)[None, :]
    else:
        # bf16: per-tile in-loop scale
        wdq = w.to(torch.float32) * scale.to(torch.float32)[None, :]
        wdq = wdq.to(inp.dtype)
        acc = torch.matmul(inp.to(torch.float32), wdq.to(torch.float32))

    if bias is not None:
        acc = acc + bias.to(torch.float32)[None, :]
    if activation == "relu":
        acc = torch.relu(acc)
    elif activation == "silu":
        acc = acc * torch.sigmoid(acc)
    return acc.to(inp.dtype)


@pytest.mark.mixed_dtypes_linear
def test_mixed_dtypes_linear():
    bench = MixedDtypesLinearBenchmark(
        op_name="mixed_dtypes_linear",
        torch_op=mixed_dtypes_linear_torch,
        dtypes=FP16_BF16_DTYPES,
    )
    bench.run()
