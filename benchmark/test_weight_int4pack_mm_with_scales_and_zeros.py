import gc

import pytest
import torch

import flag_gems

from . import base, consts
from .conftest import Config, emit_record_logger, update_result
from .consts import BenchmarkMetrics, BenchmarkResult, OperationAttribute

# Representative shapes for the int4 matmul with scales and zeros.
# (M, K, N) where K must be even and K % qGroupSize == 0.
MM_SHAPES = [
    (4, 16, 8),
    (8, 32, 16),
    (4, 64, 16),
    (8, 128, 32),
]


QGROUP_SIZES = (2, 4, 8, 16)


def _naive_dequant_mm(A, mat2, qGroupSize, qScale, qZeros):
    """Baseline: naive dequantize then matmul in PyTorch."""
    M, K = A.shape
    N = mat2.shape[0]

    # Build dequantized weight matrix
    W_f32 = torch.empty((K, N), dtype=torch.float32, device=A.device)
    for k in range(K):
        byte_idx = k // 2
        group = k // qGroupSize
        bytes_val = mat2[:, byte_idx].to(torch.int32)
        if k % 2 == 0:
            int4_vals = bytes_val & 0xF
        else:
            int4_vals = (bytes_val >> 4) & 0xF
        scales = qScale[group, :]
        zeros = qZeros[group, :]
        w_k = (int4_vals.to(torch.float32) - zeros) * scales
        W_f32[k, :] = w_k

    return torch.mm(A.to(torch.float32), W_f32).to(A.dtype)


def _gems_op(A, mat2, qGroupSize, qScale, qZeros):
    """Call the GEMS registered operator within use_gems context."""
    with flag_gems.use_gems():
        return torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(
            A, mat2, qGroupSize, qScale, qZeros
        )


class WeightInt4packMmWithScalesAndZerosBenchmark(base.Benchmark):
    """Benchmark for _weight_int4pack_mm_with_scales_and_zeros."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gems_op = _gems_op

    def set_shapes(self, shape_file_path=None):
        self.shapes = MM_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from _input_fn(shape, self.device, cur_dtype)

    def run(self):
        if Config.query:
            self.init_default_config()
            attri = OperationAttribute(
                op_name=self.op_name,
                recommended_core_shapes=self.shapes,
                shape_desc=self.shape_desc,
            )
            print(attri)
            emit_record_logger(attri.to_dict())
            return

        self.init_user_config()
        for dtype in self.to_bench_dtypes:
            metrics = []
            input_iter = self.get_input_iter(dtype)
            done = False
            while not done:
                try:
                    input_item = next(input_iter)
                except StopIteration:
                    done = True
                    continue
                except (RuntimeError, Exception) as e:
                    print(
                        f"\033[31mFAILED\033[0m: Operator={self.op_name} "
                        f"dtype={dtype} err=<<<{e}>>>"
                    )
                    pytest.fail(str(e))

                metric = BenchmarkMetrics()
                try:
                    args = input_item
                    metric.shape_detail = self.record_shapes(*args)
                    if "latency_base" in self.to_bench_metrics:
                        metric.latency_base = self.get_latency(self.torch_op, *args)
                    if "latency" in self.to_bench_metrics:
                        metric.latency = self.get_latency(self.gems_op, *args)
                    if "speedup" in self.to_bench_metrics:
                        metric.speedup = metric.latency_base / metric.latency
                except (RuntimeError, Exception) as e:
                    metric.error_msg = str(e)
                    pytest.fail(str(e))

                metrics.append(metric)
                gc.collect()

            result = BenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode=Config.mode.value,
                result=metrics,
            )
            print(result)
            update_result(self.op_name, result.to_json())
            emit_record_logger(result.to_json())


def _input_fn(shape, device, dtype):
    """Yield inputs for the benchmark from a valid random test case."""
    M, K, N = shape
    K2 = K // 2

    for qGroupSize in QGROUP_SIZES:
        if K % qGroupSize != 0:
            continue

        A = torch.randn((M, K), dtype=dtype, device=device) * 0.1

        W_int32 = torch.randint(0, 16, (K, N), dtype=torch.int32, device=device)
        mat2 = torch.empty((N, K2), dtype=torch.uint8, device=device)
        for k in range(K):
            byte_idx = k // 2
            if k % 2 == 0:
                mat2[:, byte_idx] = (W_int32[k, :].to(torch.uint8) & 0xF).to(
                    torch.uint8
                )
            else:
                mat2[:, byte_idx] = (
                    mat2[:, byte_idx].to(torch.int32)
                    | ((W_int32[k, :].to(torch.uint8) & 0xF).to(torch.int32) << 4)
                ).to(torch.uint8)

        G = K // qGroupSize
        qScale = torch.randn((G, N), dtype=dtype, device=device).abs() + 0.01
        qZeros = torch.randn((G, N), dtype=dtype, device=device) * 2.0

        yield A, mat2, qGroupSize, qScale, qZeros


@pytest.mark.weight_int4pack_mm_with_scales_and_zeros
def test_weight_int4pack_mm_with_scales_and_zeros():
    bench = WeightInt4packMmWithScalesAndZerosBenchmark(
        op_name="weight_int4pack_mm_with_scales_and_zeros",
        torch_op=_naive_dequant_mm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
