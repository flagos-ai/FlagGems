"""
Accuracy tests for FP8 block-scaled einsum (DeepSeek V4 O-projection).

Equation: bhr,hdr->bhd
Tests the Triton kernel against a pure-PyTorch reference implementation.
"""

from itertools import product

import pytest
import torch

import flag_gems
from flag_gems.fused.fp8_einsum import fp8_einsum

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="fp8e4nv (torch.float8_e4m3fn) requires SM89+ (Hopper H100/H200)",
)

BLOCK_K = 128


def fp8_einsum_ref(a, a_scale, b, b_scale, out):
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


def generate_fp8_einsum_data(
    B: int, H: int, R: int, D: int, device: str = flag_gems.device
):
    torch.manual_seed(42)
    n_k_blocks = (R + BLOCK_K - 1) // BLOCK_K
    n_d_blocks = (D + BLOCK_K - 1) // BLOCK_K

    a = torch.randn((B, H, R), device=device).to(torch.float8_e4m3fn)
    b = torch.randn((H, D, R), device=device).to(torch.float8_e4m3fn)
    a_scale = torch.rand((B, H, n_k_blocks), dtype=torch.float32, device=device) + 0.5
    b_scale = (
        torch.rand((H, n_d_blocks, n_k_blocks), dtype=torch.float32, device=device)
        + 0.5
    )
    out = torch.empty((B, H, D), device=device, dtype=torch.bfloat16)
    return dict(a=a, a_scale=a_scale, b=b, b_scale=b_scale, out=out)


FP8_EINSUM_CONFIGS = list(
    product(
        [1, 4, 16, 32, 64, 128, 256],
        [8],
        [4096],
        [1024],
    )
)


@pytest.mark.fp8_einsum
@pytest.mark.parametrize(
    "B, H, R, D",
    FP8_EINSUM_CONFIGS,
    ids=[f"B{B}_H{H}_R{R}_D{D}" for B, H, R, D in FP8_EINSUM_CONFIGS],
)
def test_fp8_einsum(B, H, R, D):
    """Test Triton fp8_einsum against pure-PyTorch reference."""
    data = generate_fp8_einsum_data(B, H, R, D)
    out_ref = torch.empty_like(data["out"])

    fp8_einsum(**data)
    fp8_einsum_ref(data["a"], data["a_scale"], data["b"], data["b_scale"], out_ref)

    torch.testing.assert_close(data["out"], out_ref, rtol=1e-2, atol=1e-2)


SMALL_CONFIGS = list(
    product(
        [1, 4, 16],
        [2],
        [256, 512],
        [128, 256],
    )
)


@pytest.mark.fp8_einsum
@pytest.mark.parametrize(
    "B, H, R, D",
    SMALL_CONFIGS,
    ids=[f"B{B}_H{H}_R{R}_D{D}" for B, H, R, D in SMALL_CONFIGS],
)
def test_fp8_einsum_small(B, H, R, D):
    """Test with smaller dimensions for quick validation."""
    data = generate_fp8_einsum_data(B, H, R, D)
    out_ref = torch.empty_like(data["out"])

    fp8_einsum(**data)
    fp8_einsum_ref(data["a"], data["a_scale"], data["b"], data["b_scale"], out_ref)

    torch.testing.assert_close(data["out"], out_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.fp8_einsum
def test_fp8_einsum_output_dtype():
    """Verify output is bfloat16."""
    data = generate_fp8_einsum_data(4, 8, 4096, 1024)
    fp8_einsum(**data)
    assert data["out"].dtype == torch.bfloat16


@pytest.mark.fp8_einsum
def test_fp8_einsum_output_shape():
    """Verify output shape is [B, H, D]."""
    B, H, R, D = 16, 8, 4096, 1024
    data = generate_fp8_einsum_data(B, H, R, D)
    fp8_einsum(**data)
    assert data["out"].shape == (B, H, D)
