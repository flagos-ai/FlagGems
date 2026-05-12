import pytest
import torch

from flag_gems.fused.fp8_einsum import fp8_einsum, fp8_einsum_ref

from . import base


class FP8EinsumBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "B, H, R, D"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 8, 4096, 1024),
            (4, 8, 4096, 1024),
            (16, 8, 4096, 1024),
            (32, 8, 4096, 1024),
            (64, 8, 4096, 1024),
            (128, 8, 4096, 1024),
            (256, 8, 4096, 1024),
        ]

    def get_input_iter(self, dtype):
        BLOCK_K = 128
        for B, H, R, D in self.shapes:
            n_k_blocks = (R + BLOCK_K - 1) // BLOCK_K
            n_d_blocks = (D + BLOCK_K - 1) // BLOCK_K

            torch.manual_seed(42)
            a = torch.randn((B, H, R), device=self.device).to(torch.float8_e4m3fn)
            b = torch.randn((H, D, R), device=self.device).to(torch.float8_e4m3fn)
            a_scale = (
                torch.rand((B, H, n_k_blocks), dtype=torch.float32, device=self.device)
                + 0.5
            )
            b_scale = (
                torch.rand(
                    (H, n_d_blocks, n_k_blocks), dtype=torch.float32, device=self.device
                )
                + 0.5
            )
            yield a, a_scale, b, b_scale


@pytest.mark.fp8_einsum
def test_fp8_einsum():
    bench = FP8EinsumBenchmark(
        op_name="fp8_einsum",
        torch_op=fp8_einsum_ref,
        gems_op=fp8_einsum,
        dtypes=[torch.float8_e4m3fn],
    )
    bench.run()
