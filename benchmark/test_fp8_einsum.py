import pytest
import torch

from flag_gems.fused.fp8_einsum import fp8_einsum

from . import base

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="fp8e4nv (torch.float8_e4m3fn) requires SM89+ (Hopper H100/H200)",
)

BLOCK_K = 128


def _torch_fp8_einsum_ref(a, a_scale, b, b_scale, out):
    """Pure-PyTorch reference: block-scaled FP8 einsum bhr,hdr->bhd."""
    B, H, R = a.shape
    _, D, _ = b.shape

    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    n_k_blocks = (R + BLOCK_K - 1) // BLOCK_K

    acc = torch.zeros((B, H, D), dtype=torch.float32, device=a.device)
    for k_blk in range(n_k_blocks):
        k_start = k_blk * BLOCK_K
        k_end = min(k_start + BLOCK_K, R)

        a_block = a_f32[:, :, k_start:k_end]
        b_block = b_f32[:, :, k_start:k_end]
        dot = torch.einsum("bhr,hdr->bhd", a_block, b_block)

        a_s = a_scale[:, :, k_blk].unsqueeze(-1)
        b_s = b_scale[:, :, k_blk].repeat_interleave(BLOCK_K, dim=1)[:, :D].unsqueeze(0)
        acc += dot * (a_s * b_s)

    out.copy_(acc.to(torch.bfloat16))


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
            out = torch.empty((B, H, D), device=self.device, dtype=torch.bfloat16)
            yield a, a_scale, b, b_scale, out


@pytest.mark.fp8_einsum
def test_fp8_einsum():
    bench = FP8EinsumBenchmark(
        op_name="fp8_einsum",
        torch_op=_torch_fp8_einsum_ref,
        gems_op=fp8_einsum,
        dtypes=[torch.float8_e4m3fn],
    )
    bench.run()
