import pytest
import torch

import flag_gems

from . import base, consts


def _cuda_fp8_available():
    if flag_gems.device != "cuda" or not torch.cuda.is_available():
        return False
    if not flag_gems.runtime.device.support_bf16:
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 89 and hasattr(torch, "float8_e4m3fn")


def _benchmark_dtypes():
    if _cuda_fp8_available() and hasattr(torch, "_scaled_mm"):
        return [torch.float8_e4m3fn]
    return []


def _dtype_id(dtype):
    return str(dtype).split(".")[-1]


class ScaledMMBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, op_name, torch_op, gems_op, dtype, use_out=False):
        super().__init__(op_name, torch_op, dtypes=[dtype])
        self.set_gems(gems_op)
        self.use_out = use_out

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (16, 16, 16),
            (128, 128, 128),
            (512, 512, 512),
        ]
        self.shape_desc = "M, N, K"

    def set_more_shapes(self):
        return [
            (1024, 1024, 1024),
            (256, 4096, 4096),
        ]

    def get_input_iter(self, dtype):
        for M, N, K in self.shapes:
            mat1 = torch.randn((M, K), dtype=torch.float32, device=flag_gems.device)
            mat2 = torch.randn((K, N), dtype=torch.float32, device=flag_gems.device)
            mat1 = (mat1 * 0.25).to(dtype)
            mat2 = (mat2 * 0.25).to(dtype).t().contiguous().t()
            out_dtype = torch.bfloat16
            bias_dtype = out_dtype

            scale_a = torch.linspace(0.75, 1.25, M, device=flag_gems.device).reshape(
                M, 1
            )
            scale_b = torch.linspace(1.25, 0.75, N, device=flag_gems.device).reshape(
                1, N
            )
            bias = torch.randn((N,), dtype=bias_dtype, device=flag_gems.device)
            if self.use_out:
                out = torch.empty((M, N), dtype=out_dtype, device=flag_gems.device)
                yield mat1, mat2, scale_a, scale_b, bias, None, out_dtype, False, {
                    "out": out
                }
            else:
                yield mat1, mat2, scale_a, scale_b, bias, None, out_dtype, False

    def get_tflops(self, op, *args, **kwargs):
        M, K = args[0].shape
        N = args[1].shape[1]
        return 2 * M * N * K


@pytest.mark.scaled_mm
@pytest.mark.parametrize("dtype", _benchmark_dtypes(), ids=_dtype_id)
def test_scaled_mm_benchmark(dtype):
    bench = ScaledMMBenchmark("scaled_mm", torch._scaled_mm, flag_gems.scaled_mm, dtype)
    bench.run()


@pytest.mark.scaled_mm_out
@pytest.mark.parametrize("dtype", _benchmark_dtypes(), ids=_dtype_id)
def test_scaled_mm_out_benchmark(dtype):
    bench = ScaledMMBenchmark(
        "scaled_mm_out",
        torch.ops.aten._scaled_mm.out,
        flag_gems.scaled_mm_out,
        dtype,
        use_out=True,
    )
    bench.run()
