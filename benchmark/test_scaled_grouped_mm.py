import pytest
import torch

import flag_gems

from . import base, consts


def _cuda_fp8_available():
    if flag_gems.device != "cuda" or not torch.cuda.is_available():
        return False
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 89


def _is_float8(dtype):
    return hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn


def _default_out_dtype(dtype):
    if _is_float8(dtype):
        return torch.bfloat16
    return dtype


def torch_scaled_grouped_mm_baseline(
    mat_a,
    mat_b,
    scale_a,
    scale_b,
    offs=None,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
):
    target_dtype = out_dtype or _default_out_dtype(mat_a.dtype)
    starts = [0] + offs.detach().cpu().tolist()
    chunks = []
    for group_idx in range(mat_b.shape[0]):
        m_start, m_end = starts[group_idx], starts[group_idx + 1]
        out = mat_a[m_start:m_end].float().mm(mat_b[group_idx].float())
        out = out * scale_a[m_start:m_end].reshape(-1, 1) * scale_b[group_idx]
        if bias is not None:
            out = out + bias
        chunks.append(out.to(target_dtype))
    return torch.cat(chunks, dim=0)


class ScaledGroupedMMBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self):
        dtypes = [torch.float16, torch.float32]
        if flag_gems.runtime.device.support_bf16:
            dtypes.append(torch.bfloat16)
        if _cuda_fp8_available():
            dtypes.append(torch.float8_e4m3fn)

        super().__init__(
            "scaled_grouped_mm", torch_scaled_grouped_mm_baseline, dtypes=dtypes
        )
        self.set_gems(flag_gems.scaled_grouped_mm)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (4, 64, 128, 128),
            (8, 128, 256, 256),
            (16, 256, 512, 512),
        ]
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes = list(dict.fromkeys(self.shapes + self.set_more_shapes()))
        self.shape_desc = "groups, M_per_group, N, K"

    def set_more_shapes(self):
        return [
            (16, 512, 1024, 1024),
            (32, 256, 2048, 1024),
        ]

    def get_input_iter(self, dtype):
        for groups, m_per_group, N, K in self.shapes:
            sizes = torch.arange(
                m_per_group,
                m_per_group + groups,
                dtype=torch.int32,
                device=flag_gems.device,
            )
            offs = torch.cumsum(sizes, dim=0).to(torch.int32)
            M = int(offs[-1].item())

            mat_a = torch.randn((M, K), dtype=torch.float32, device=flag_gems.device)
            mat_b = torch.randn(
                (groups, K, N), dtype=torch.float32, device=flag_gems.device
            )
            if _is_float8(dtype):
                mat_a = (mat_a * 0.25).to(dtype)
                mat_b = (mat_b * 0.25).to(dtype)
                out_dtype = torch.bfloat16
            else:
                mat_a = mat_a.to(dtype)
                mat_b = mat_b.to(dtype)
                out_dtype = dtype

            scale_a = torch.linspace(0.75, 1.25, M, device=flag_gems.device)
            scale_b = torch.linspace(
                1.25, 0.75, groups * N, device=flag_gems.device
            ).reshape(groups, N)
            bias = torch.randn((N,), dtype=torch.float32, device=flag_gems.device)
            yield mat_a, mat_b, scale_a, scale_b, offs, bias, None, out_dtype, False

    def get_tflops(self, op, *args, **kwargs):
        mat_b = args[1]
        offs = args[4]
        groups, K, N = mat_b.shape
        sizes = torch.diff(
            offs, prepend=torch.zeros(1, device=offs.device, dtype=offs.dtype)
        )
        total_flops = 0
        for group_idx in range(groups):
            total_flops += int(sizes[group_idx].item()) * N * K * 2
        return total_flops


@pytest.mark.scaled_grouped_mm
def test_scaled_grouped_mm_benchmark():
    bench = ScaledGroupedMMBenchmark()
    bench.run()
