"""Tests for router_gemm_bf16_fp32 (DeepSeek V4 MoE routing GEMM)."""

import pytest
import torch

from flag_gems.fused.DSA.router_gemm_bf16_fp32 import router_gemm_bf16_fp32


def torch_router_gemm_bf16_fp32_ref(input: torch.Tensor, weight: torch.Tensor):
    """Reference implementation using PyTorch native ops."""
    # Compute: output = input @ weight.T
    # input: [M, K] bf16, weight: [N, K] bf16 -> output: [M, N] fp32
    return torch.matmul(input.float(), weight.T.float())


@pytest.mark.router_gemm_bf16_fp32
@pytest.mark.parametrize(
    "M, N, K",
    [
        # Decode shapes (small M, latency-critical)
        (1, 384, 7168),  # single token decode
        (8, 384, 7168),
        (32, 384, 7168),
        # Prefill shapes (large M, throughput-oriented)
        (128, 384, 7168),
        (512, 384, 7168),
        (2048, 384, 7168),
        # Edge cases
        (7, 384, 7168),  # non-power-of-2 M
    ],
)
def test_router_gemm_bf16_fp32_correctness(M, N, K):
    """Test correctness against PyTorch reference across various shapes."""
    torch.manual_seed(42)

    # Generate random inputs
    input = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # Compute outputs
    output_triton = router_gemm_bf16_fp32(input, weight)
    output_torch = torch_router_gemm_bf16_fp32_ref(input, weight)

    # Check shape and dtype
    assert output_triton.shape == (
        M,
        N,
    ), f"Expected shape ({M}, {N}), got {output_triton.shape}"
    assert (
        output_triton.dtype == torch.float32
    ), f"Expected dtype float32, got {output_triton.dtype}"

    # Check numerical accuracy
    # Use relative tolerance for bf16 -> fp32 accumulation
    torch.testing.assert_close(
        output_triton,
        output_torch,
        rtol=1e-2,  # 1% relative tolerance (bf16 has ~3 decimal digits precision)
        atol=1e-1,  # absolute tolerance for small values
    )


@pytest.mark.router_gemm_bf16_fp32
@pytest.mark.parametrize(
    "M, N, K",
    [
        # Non-standard shapes (not DeepSeek V4 config)
        (16, 128, 4096),  # smaller N and K
        (32, 512, 8192),  # larger N and K
        (64, 256, 5120),  # arbitrary dimensions
    ],
)
def test_router_gemm_bf16_fp32_general_shapes(M, N, K):
    """Test generalization to non-DeepSeek-V4 shapes."""
    torch.manual_seed(42)

    input = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    output_triton = router_gemm_bf16_fp32(input, weight)
    output_torch = torch_router_gemm_bf16_fp32_ref(input, weight)

    assert output_triton.shape == (M, N)
    assert output_triton.dtype == torch.float32

    torch.testing.assert_close(output_triton, output_torch, rtol=1e-2, atol=1e-1)


@pytest.mark.router_gemm_bf16_fp32
def test_router_gemm_bf16_fp32_stride():
    """Test with non-contiguous (strided) tensors."""
    torch.manual_seed(42)
    M, N, K = 32, 384, 7168

    # Create non-contiguous input via transpose
    input_base = torch.randn(K, M, dtype=torch.bfloat16, device="cuda")
    input = input_base.T  # [M, K], non-contiguous

    # Create non-contiguous weight via slicing
    weight_base = torch.randn(N * 2, K, dtype=torch.bfloat16, device="cuda")
    weight = weight_base[::2, :]  # [N, K], non-contiguous

    assert not input.is_contiguous()
    assert not weight.is_contiguous()

    output_triton = router_gemm_bf16_fp32(input, weight)
    output_torch = torch_router_gemm_bf16_fp32_ref(input, weight)

    torch.testing.assert_close(output_triton, output_torch, rtol=1e-2, atol=1e-1)


@pytest.mark.router_gemm_bf16_fp32
def test_router_gemm_bf16_fp32_zero_input():
    """Test with zero input (edge case for numerical stability)."""
    M, N, K = 16, 384, 7168

    input = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    output = router_gemm_bf16_fp32(input, weight)

    # Output should be all zeros
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)


@pytest.mark.router_gemm_bf16_fp32
def test_router_gemm_bf16_fp32_zero_weight():
    """Test with zero weight (edge case for numerical stability)."""
    M, N, K = 16, 384, 7168

    input = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros(N, K, dtype=torch.bfloat16, device="cuda")

    output = router_gemm_bf16_fp32(input, weight)

    # Output should be all zeros
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)


@pytest.mark.router_gemm_bf16_fp32
def test_router_gemm_bf16_fp32_dtype_check():
    """Test that dtype assertions work correctly."""
    M, N, K = 16, 384, 7168

    # Test with wrong input dtype
    input_fp32 = torch.randn(M, K, dtype=torch.float32, device="cuda")
    weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(AssertionError, match="input must be bf16"):
        router_gemm_bf16_fp32(input_fp32, weight_bf16)

    # Test with wrong weight dtype
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight_fp32 = torch.randn(N, K, dtype=torch.float32, device="cuda")

    with pytest.raises(AssertionError, match="weight must be bf16"):
        router_gemm_bf16_fp32(input_bf16, weight_fp32)


@pytest.mark.router_gemm_bf16_fp32
def test_router_gemm_bf16_fp32_shape_check():
    """Test that shape assertions work correctly."""
    M, N, K = 16, 384, 7168

    # Test with mismatched K dimension
    input = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(N, K + 1, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(AssertionError, match="K dimension must match"):
        router_gemm_bf16_fp32(input, weight)

    # Test with 1D input
    input_1d = torch.randn(K, dtype=torch.bfloat16, device="cuda")
    weight_2d = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(AssertionError, match="must be 2D"):
        router_gemm_bf16_fp32(input_1d, weight_2d)


if __name__ == "__main__":
    # Run a quick smoke test
    test_router_gemm_bf16_fp32_correctness(32, 384, 7168)
