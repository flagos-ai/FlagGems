"""
Accuracy tests for FP8 block-scaled einsum (DeepSeek V4 O-projection).

Equation: bhr,hdr->bhd
Tests the Triton kernel against the DeepGEMM CUDA reference implementation.
"""
from itertools import product

import pytest
import torch

import flag_gems
from flag_gems.fused.fp8_einsum import fp8_einsum, fp8_einsum_ref


def generate_fp8_einsum_data(
    B: int, H: int, R: int, D: int, device: str = flag_gems.device
):
    torch.manual_seed(42)
    BLOCK_K = 128
    n_k_blocks = (R + BLOCK_K - 1) // BLOCK_K
    n_d_blocks = (D + BLOCK_K - 1) // BLOCK_K

    a_fp32 = torch.randn((B, H, R), device=device)
    b_fp32 = torch.randn((H, D, R), device=device)

    a = a_fp32.to(torch.float8_e4m3fn)
    b = b_fp32.to(torch.float8_e4m3fn)

    a_scale = torch.rand((B, H, n_k_blocks), dtype=torch.float32, device=device) + 0.5
    b_scale = (
        torch.rand((H, n_d_blocks, n_k_blocks), dtype=torch.float32, device=device) + 0.5
    )

    out = torch.empty((B, H, D), device=device, dtype=torch.bfloat16)

    return dict(a=a, a_scale=a_scale, b=b, b_scale=b_scale, out=out)


FP8_EINSUM_CONFIGS = list(
    product(
        [1, 4, 16, 32, 64, 128, 256],  # B (batch)
        [8],  # H (heads)
        [4096],  # R (reduction dim)
        [1024],  # D (output dim)
    )
)


@pytest.mark.fp8_einsum
@pytest.mark.parametrize(
    "B, H, R, D",
    FP8_EINSUM_CONFIGS,
    ids=[f"B{B}_H{H}_R{R}_D{D}" for B, H, R, D in FP8_EINSUM_CONFIGS],
)
def test_fp8_einsum_vs_ref(B, H, R, D):
    """Test Triton fp8_einsum against DeepGEMM CUDA reference."""
    data = generate_fp8_einsum_data(B, H, R, D)
    out_ref = torch.empty_like(data["out"])

    fp8_einsum(**data)
    fp8_einsum_ref(data["a"], data["a_scale"], data["b"], data["b_scale"], out_ref)

    torch.testing.assert_close(data["out"], out_ref, rtol=1e-2, atol=1e-2)


SMALL_CONFIGS = list(
    product(
        [1, 4, 16],  # B
        [2],  # H
        [256, 512],  # R
        [128, 256],  # D
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
