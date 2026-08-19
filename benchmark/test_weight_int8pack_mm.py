import pytest
import torch

import flag_gems

from . import base
from .consts import BenchmarkMetrics

# FP16/BF16 only: int8 matmul requires half-precision activation
FP16_BF16_DTYPES = [torch.float16, torch.bfloat16]


# LLM-scale shapes: (M, N, K) where M = tokens, N = output features, K = input features.
# These cover typical weight matrix dimensions found in transformer models.
WEIGHT_INT8PACK_MM_SHAPES = [
    (1, 4096, 4096),
    (1, 4096, 11008),
    (1, 11008, 4096),
    (1, 8192, 8192),
    (1, 8192, 28672),
    (1, 28672, 8192),
    (4, 4096, 4096),
    (4, 4096, 11008),
    (4, 11008, 4096),
    (16, 4096, 4096),
    (16, 4096, 11008),
    (16, 11008, 4096),
    (32, 4096, 4096),
    (32, 4096, 11008),
    (32, 11008, 4096),
    (64, 4096, 4096),
    (64, 4096, 11008),
    (64, 11008, 4096),
    (128, 4096, 4096),
    (128, 4096, 11008),
    (128, 11008, 4096),
]


class WeightInt8packMMBenchmark(base.Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gems_op = flag_gems.weight_int8pack_mm

    def set_shapes(self, shape_file_path=None):
        self.shapes = WEIGHT_INT8PACK_MM_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from weight_int8pack_mm_input_fn(shape, cur_dtype, self.device)

    def _run_metric(self, input_item):
        metric = BenchmarkMetrics()
        A, B, scales = input_item
        metric.shape_detail = self.record_shapes(A, B, scales)
        try:
            if "latency_base" in self.to_bench_metrics:
                metric.latency_base = self.get_latency(self.torch_op, A, B, scales)
            if "latency" in self.to_bench_metrics:
                metric.latency = self.get_latency(self.gems_op, A, B, scales)
            if "speedup" in self.to_bench_metrics:
                metric.speedup = metric.latency_base / metric.latency
        except (RuntimeError, Exception) as e:
            metric.error_msg = str(e)
            pytest.fail(str(e))
        return metric


def weight_int8pack_mm_input_fn(shape, dtype, device):
    """Yield tuples of (A, B_int8, scales) for the given shape and dtype."""
    M, N, K = shape
    A = torch.randn((M, K), dtype=dtype, device=device)
    B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)
    scales = torch.randn((N,), dtype=dtype, device=device)
    yield (A, B, scales)


def weight_int8pack_mm_torch(A, B, scales):
    """Torch baseline: dequantize int8 weights and compute matmul with scaling."""
    B_fp = B.to(A.dtype)
    result = torch.matmul(A, B_fp.T)
    result = result * scales.unsqueeze(0)
    return result


@pytest.mark.weight_int8pack_mm
def test_weight_int8pack_mm():
    bench = WeightInt8packMMBenchmark(
        op_name="weight_int8pack_mm",
        torch_op=weight_int8pack_mm_torch,
        dtypes=FP16_BF16_DTYPES,
    )
    bench.run()
