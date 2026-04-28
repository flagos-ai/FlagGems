from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts

try:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm as vllm_w8a8_triton_block_scaled_mm,
    )

    VLLM_W8A8_BLOCK_FP8_AVAILABLE = True
except Exception:
    vllm_w8a8_triton_block_scaled_mm = None
    VLLM_W8A8_BLOCK_FP8_AVAILABLE = False


W8A8_BLOCK_FP8_MNK_SHAPES = [
    (64, 128, 128),
    (128, 256, 512),
    (1, 4096, 7168),
    (16, 4096, 7168),
    (64, 4096, 7168),
    (83, 7748, 3884),
    (84, 7168, 3884),
]

W8A8_BLOCK_FP8_BLOCK_SIZE = [128, 128]


def get_w8a8_block_fp8_dtype():
    if flag_gems.device != "cuda" or not torch.cuda.is_available():
        return None

    major, _ = torch.cuda.get_device_capability()

    if major > 8 and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn

    if major == 8 and hasattr(torch, "float8_e5m2"):
        return torch.float8_e5m2

    return None


def rand_fp8_tensor(shape, device, dtype):
    finfo = torch.finfo(dtype)
    return (
        torch.randn(shape, device=device, dtype=torch.float32)
        .clamp(min=finfo.min, max=finfo.max)
        .to(dtype)
    )


class W8A8BlockFP8MatmulBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, block_size=None, **kwargs):
        super().__init__(*args, **kwargs)

        if block_size is None:
            self.block_size = W8A8_BLOCK_FP8_BLOCK_SIZE[:]
        else:
            self.block_size = list(block_size)

        self.shape_desc = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = W8A8_BLOCK_FP8_MNK_SHAPES[:]
        self.shape_desc = "M, N, K"

    def get_input_iter(self, dtype) -> Generator:
        fp8_dtype = get_w8a8_block_fp8_dtype()

        if fp8_dtype is None:
            raise RuntimeError(
                "w8a8_block_fp8_matmul benchmark requires CUDA device with FP8 support"
            )

        block_n, block_k = self.block_size
        for m, n, k in self.shapes:
            num_k_groups = (k + block_k - 1) // block_k
            num_n_groups = (n + block_n - 1) // block_n

            A = rand_fp8_tensor((m, k), self.device, fp8_dtype).contiguous()
            B = rand_fp8_tensor((n, k), self.device, fp8_dtype).contiguous()
            As = (
                0.01
                * torch.rand((m, num_k_groups), dtype=torch.float32, device=self.device)
                + 0.005
            ).contiguous()
            Bs = (
                0.01
                * torch.rand(
                    (num_n_groups, num_k_groups),
                    dtype=torch.float32,
                    device=self.device,
                )
                + 0.005
            ).contiguous()

            yield A, B, As, Bs, self.block_size[:], torch.float16

    def get_tflops(self, op, *args, **kwargs):
        A, B = args[0], args[1]
        m, k = A.shape
        n = B.shape[0]
        return 2 * m * n * k


@pytest.mark.w8a8_block_fp8_matmul
@pytest.mark.skipif(
    not VLLM_W8A8_BLOCK_FP8_AVAILABLE,
    reason="w8a8_block_fp8_matmul benchmark requires vLLM baseline operator",
)
def test_perf_w8a8_block_fp8_matmul():
    if get_w8a8_block_fp8_dtype() is None:
        pytest.skip(
            "w8a8_block_fp8_matmul benchmark requires CUDA device with FP8 support"
        )

    bench = W8A8BlockFP8MatmulBenchmark(
        op_name="w8a8_block_fp8_matmul",
        torch_op=vllm_w8a8_triton_block_scaled_mm,
        gems_op=flag_gems.w8a8_block_fp8_matmul,
        dtypes=["fp8"],
    )
    bench.run()
