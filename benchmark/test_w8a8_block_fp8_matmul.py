from typing import Generator

import pytest
import torch

import flag_gems

from .attri_util import DEFAULT_METRICS
from .performance_utils import Benchmark


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


def get_fp8_dtype():
    if flag_gems.device != "cuda" or not torch.cuda.is_available():
        return None

    major, _ = torch.cuda.get_device_capability()
    if major > 8 and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    if major == 8 and hasattr(torch, "float8_e5m2"):
        return torch.float8_e5m2
    return None


FP8_DTYPE = get_fp8_dtype()
FP8_DTYPES = [FP8_DTYPE] if FP8_DTYPE is not None else []


def rand_fp8_tensor(shape, device, dtype):
    finfo = torch.finfo(dtype)
    return (
        torch.randn(shape, device=device, dtype=torch.float32)
        .clamp(min=finfo.min, max=finfo.max)
        .to(dtype)
    )


def torch_w8a8_block_fp8_matmul_ref(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    block_n, block_k = block_size
    K = A.shape[-1]
    M = A.numel() // K
    N = B.shape[0]

    A_flat = A.reshape(M, K).to(torch.float32)
    B_fp32 = B.to(torch.float32)

    A_scale = (
        As.reshape(M, -1)
        .to(torch.float32)
        .repeat_interleave(block_k, dim=-1)[:, :K]
    )
    B_scale = (
        Bs.to(torch.float32)
        .repeat_interleave(block_n, dim=0)[:N]
        .repeat_interleave(block_k, dim=1)[:, :K]
    )

    out = torch.matmul(A_flat * A_scale, (B_fp32 * B_scale).T)
    return out.to(output_dtype).reshape(A.shape[:-1] + (N,))


class W8A8BlockFP8MatmulBenchmark(Benchmark):
    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, block_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = (
            W8A8_BLOCK_FP8_BLOCK_SIZE[:] if block_size is None else list(block_size)
        )
        self.shape_desc = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = W8A8_BLOCK_FP8_MNK_SHAPES[:]
        self.shape_desc = "M, N, K"

    def get_input_iter(self, dtype) -> Generator:
        block_n, block_k = self.block_size
        for m, n, k in self.shapes:
            num_k_groups = (k + block_k - 1) // block_k
            num_n_groups = (n + block_n - 1) // block_n

            A = rand_fp8_tensor((m, k), self.device, dtype).contiguous()
            B = rand_fp8_tensor((n, k), self.device, dtype).contiguous()
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
def test_perf_w8a8_block_fp8_matmul():
    if not FP8_DTYPES:
        pytest.skip(
            "w8a8_block_fp8_matmul benchmark requires CUDA device with FP8 support"
        )

    bench = W8A8BlockFP8MatmulBenchmark(
        op_name="w8a8_block_fp8_matmul",
        torch_op=torch_w8a8_block_fp8_matmul_ref,
        dtypes=FP8_DTYPES,
    )
    bench.set_gems(flag_gems.w8a8_block_fp8_matmul)
    bench.run()
