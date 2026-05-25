from typing import Generator

import pytest
import torch

from . import base, consts

ROUTER_GEMM_SHAPES = [
    (1, 8, 4096),
    (16, 8, 4096),
    (64, 8, 4096),
    (256, 8, 4096),
    (1024, 8, 4096),
    (4096, 8, 4096),
    (64, 64, 7168),
    (256, 64, 7168),
    (1024, 128, 7168),
    (4096, 128, 7168),
]


def torch_router_gemm(x, weight):
    return torch.mm(x, weight.t()).to(torch.float32)


try:
    from flag_gems.runtime.backend._nvidia.hopper.ops.mm import router_gemm

    ROUTER_GEMM_AVAILABLE = True
except Exception:
    router_gemm = None
    ROUTER_GEMM_AVAILABLE = False


class RouterGemmBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape_desc = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = ROUTER_GEMM_SHAPES[:]
        self.shape_desc = "M, N, K"

    def get_input_iter(self, dtype) -> Generator:
        for m, n, k in self.shapes:
            x = torch.randn((m, k), dtype=torch.bfloat16, device=self.device)
            weight = torch.randn((n, k), dtype=torch.bfloat16, device=self.device)
            yield x, weight

    def get_tflops(self, op, *args, **kwargs):
        x, weight = args[0], args[1]
        m, k = x.shape
        n = weight.shape[0]
        return 2 * m * n * k


@pytest.mark.router_gemm
@pytest.mark.skipif(
    not ROUTER_GEMM_AVAILABLE,
    reason="router_gemm benchmark requires NVIDIA Hopper backend",
)
def test_perf_router_gemm():
    bench = RouterGemmBenchmark(
        op_name="router_gemm",
        torch_op=torch_router_gemm,
        gems_op=router_gemm,
        dtypes=[torch.bfloat16],
    )
    bench.run()
