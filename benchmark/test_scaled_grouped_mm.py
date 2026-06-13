import pytest
import torch

import flag_gems

from . import base, consts


def _native_scaled_grouped_mm_benchmark_available():
    if not hasattr(torch, "_scaled_grouped_mm"):
        return False
    if flag_gems.device != "cuda" or not torch.cuda.is_available():
        return False
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 89


class ScaledGroupedMMBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self):
        super().__init__(
            "scaled_grouped_mm", torch._scaled_grouped_mm, dtypes=[torch.float8_e4m3fn]
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
                (groups, N, K), dtype=torch.float32, device=flag_gems.device
            )
            mat_a = (mat_a * 0.25).to(dtype)
            mat_b = (mat_b * 0.25).to(dtype).transpose(-1, -2)
            out_dtype = torch.bfloat16

            scale_a = torch.linspace(0.75, 1.25, M, device=flag_gems.device)
            scale_b = torch.linspace(
                1.25, 0.75, groups * N, device=flag_gems.device
            ).reshape(groups, N)
            yield mat_a, mat_b, scale_a, scale_b, {
                "offs": offs,
                "bias": None,
                "out_dtype": out_dtype,
                "use_fast_accum": False,
            }

    def get_tflops(self, op, *args, **kwargs):
        mat_b = args[1]
        offs = kwargs["offs"]
        groups, K, N = mat_b.shape
        sizes = torch.diff(
            offs, prepend=torch.zeros(1, device=offs.device, dtype=offs.dtype)
        )
        total_flops = 0
        for group_idx in range(groups):
            total_flops += int(sizes[group_idx].item()) * N * K * 2
        return total_flops


@pytest.mark.scaled_grouped_mm
@pytest.mark.skipif(
    not _native_scaled_grouped_mm_benchmark_available(),
    reason="native torch._scaled_grouped_mm benchmark requires CUDA FP8 on SM89+.",
)
def test_scaled_grouped_mm_benchmark():
    bench = ScaledGroupedMMBenchmark()
    bench.run()
