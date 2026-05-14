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


def _ascend_available():
    return flag_gems.vendor_name == "ascend"


def _benchmark_cases():
    if _cuda_fp8_available() and hasattr(torch, "_scaled_mm"):
        return [
            (torch.float8_e4m3fn, "scalar", torch.float16, True),
            (torch.float8_e4m3fn, "scalar", torch.bfloat16, True),
            (torch.float8_e4m3fn, "scalar", torch.float32, False),
            (torch.float8_e4m3fn, "rowwise", torch.bfloat16, True),
        ]

    if _ascend_available():
        cases = [
            (torch.float16, "rowwise", torch.float16, True),
            (torch.float32, "rowwise", torch.float32, True),
        ]
        if flag_gems.runtime.device.support_bf16:
            cases.append((torch.bfloat16, "rowwise", torch.bfloat16, True))
        return cases

    return []


def _case_id(case):
    dtype, scale_mode, out_dtype, use_bias = case
    return (
        f"{str(dtype).split('.')[-1]}-{scale_mode}-"
        f"{str(out_dtype).split('.')[-1]}-bias_{use_bias}"
    )


def _scale_for_output(scale, rows, cols, is_left_scale):
    if scale.numel() == 1:
        return scale
    if scale.ndim == 1:
        if is_left_scale and scale.shape[0] == rows:
            return scale.reshape(rows, 1)
        if not is_left_scale and scale.shape[0] == cols:
            return scale.reshape(1, cols)
    return scale


def torch_scaled_mm_reference(
    mat1,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
):
    rows = mat1.shape[0]
    cols = mat2.shape[1]
    out = torch.mm(mat1.float(), mat2.float())
    out = out * _scale_for_output(scale_a.float(), rows, cols, True)
    out = out * _scale_for_output(scale_b.float(), rows, cols, False)
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype or mat1.dtype)


def torch_scaled_mm_out_reference(
    mat1,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
    *,
    out,
):
    result = torch_scaled_mm_reference(
        mat1,
        mat2,
        scale_a,
        scale_b,
        bias=bias,
        scale_result=scale_result,
        out_dtype=out_dtype,
        use_fast_accum=use_fast_accum,
    )
    out.copy_(result)
    return out


class ScaledMMBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, op_name, torch_op, gems_op, case, use_out=False):
        dtype = case[0]
        super().__init__(op_name, torch_op, dtypes=[dtype])
        self.set_gems(gems_op)
        self.case = case
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
        _, scale_mode, out_dtype, use_bias = self.case
        for M, N, K in self.shapes:
            mat1 = torch.randn((M, K), dtype=torch.float32, device=flag_gems.device)
            mat2 = torch.randn((K, N), dtype=torch.float32, device=flag_gems.device)
            mat1 = (mat1 * 0.25).to(dtype)
            mat2 = (mat2 * 0.25).to(dtype).t().contiguous().t()

            if scale_mode == "scalar":
                scale_a = torch.tensor([0.75], device=flag_gems.device)
                scale_b = torch.tensor([1.25], device=flag_gems.device)
            else:
                scale_a = torch.linspace(
                    0.75, 1.25, M, device=flag_gems.device
                ).reshape(M, 1)
                scale_b = torch.linspace(
                    1.25, 0.75, N, device=flag_gems.device
                ).reshape(1, N)

            bias = None
            if use_bias:
                bias = torch.randn((N,), dtype=out_dtype, device=flag_gems.device)
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
@pytest.mark.parametrize("case", _benchmark_cases(), ids=_case_id)
def test_scaled_mm_benchmark(case):
    torch_op = torch_scaled_mm_reference if _ascend_available() else torch._scaled_mm
    bench = ScaledMMBenchmark("scaled_mm", torch_op, flag_gems.scaled_mm, case)
    bench.run()


@pytest.mark.scaled_mm_out
@pytest.mark.parametrize("case", _benchmark_cases(), ids=_case_id)
def test_scaled_mm_out_benchmark(case):
    torch_op = (
        torch_scaled_mm_out_reference
        if _ascend_available()
        else torch.ops.aten._scaled_mm.out
    )
    bench = ScaledMMBenchmark(
        "scaled_mm_out",
        torch_op,
        flag_gems.scaled_mm_out,
        case,
        use_out=True,
    )
    bench.run()
