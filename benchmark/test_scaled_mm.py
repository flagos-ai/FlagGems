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


def _is_fp8(dtype):
    return hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn


def _scale_for_output(scale, rows, cols, is_left_scale):
    if scale.numel() == 1:
        return scale
    if scale.ndim == 1:
        if is_left_scale and scale.shape[0] == rows:
            return scale.reshape(rows, 1)
        if not is_left_scale and scale.shape[0] == cols:
            return scale.reshape(1, cols)
    return scale


def torch_scaled_mm_baseline(
    mat1,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
):
    if _is_fp8(mat1.dtype) and flag_gems.device == "cuda":
        return torch._scaled_mm(
            mat1,
            mat2,
            scale_a,
            scale_b,
            bias=bias,
            scale_result=scale_result,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )

    rows = mat1.shape[0]
    cols = mat2.shape[1]
    out = torch.mm(mat1.float(), mat2.float())
    out = out * _scale_for_output(scale_a.float(), rows, cols, True)
    out = out * _scale_for_output(scale_b.float(), rows, cols, False)
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype or mat1.dtype)


class ScaledMMBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self):
        dtypes = [torch.float16, torch.float32]
        if flag_gems.runtime.device.support_bf16:
            dtypes.append(torch.bfloat16)
        if _cuda_fp8_available():
            dtypes.append(torch.float8_e4m3fn)

        super().__init__("scaled_mm", torch_scaled_mm_baseline, dtypes=dtypes)
        self.set_gems(flag_gems.scaled_mm)

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
            if _is_fp8(dtype):
                mat1 = (mat1 * 0.25).to(dtype)
                mat2 = (mat2 * 0.25).to(dtype).t().contiguous().t()
                out_dtype = torch.bfloat16
                bias_dtype = out_dtype
            else:
                mat1 = mat1.to(dtype)
                mat2 = mat2.to(dtype)
                out_dtype = dtype
                bias_dtype = dtype

            scale_a = torch.linspace(0.75, 1.25, M, device=flag_gems.device).reshape(
                M, 1
            )
            scale_b = torch.linspace(1.25, 0.75, N, device=flag_gems.device).reshape(
                1, N
            )
            bias = torch.randn((N,), dtype=bias_dtype, device=flag_gems.device)
            yield mat1, mat2, scale_a, scale_b, bias, None, out_dtype, False

    def get_tflops(self, op, *args, **kwargs):
        M, K = args[0].shape
        N = args[1].shape[1]
        return 2 * M * N * K


@pytest.mark.scaled_mm
def test_scaled_mm_benchmark():
    bench = ScaledMMBenchmark()
    bench.run()
