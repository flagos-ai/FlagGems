import random

import numpy as np
import pytest
import torch

from flag_gems.fused.DSA.sparse_mla import triton_sparse_mla_fwd_interface

from .torch_src.sparse_mla_fwd import (
    ref_sparse_mla_fwd_interface,  # , sparse_mla_fwd_interface
)

try:
    from ..accuracy_utils import gems_assert_close, init_seed, to_reference
except ImportError:
    # Accuracy check function
    def gems_assert_close(
        actual, expected, dtype, equal_nan=False, atol=None, rtol=None
    ):
        # For bfloat16 and float16, use more relaxed tolerance
        if atol is None:
            if dtype in [torch.bfloat16, torch.float16]:
                atol = 1e-2
                rtol = 1e-2
            else:
                atol = 1e-4
                rtol = 1e-4

        print(f"Actual shape: {actual.shape}, Expected shape: {expected.shape}")
        print(f"Actual dtype: {actual.dtype}, Expected dtype: {expected.dtype}")

        # Calculate difference statistics
        diff = torch.abs(actual - expected)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()

        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")
        print(
            f"Actual range: [{torch.min(actual).item():.6f}, {torch.max(actual).item():.6f}]"
        )
        print(
            f"Expected range: [{torch.min(expected).item():.6f}, {torch.max(expected).item():.6f}]"
        )

        torch.testing.assert_close(
            actual, expected, atol=atol, rtol=rtol, equal_nan=equal_nan
        )

    def init_seed(seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def to_reference(tensor, requires_grad=False):
        result = tensor.detach().clone()
        if requires_grad:
            result.requires_grad_()
        return result


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_sparse_mla_input(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    num_kv_heads: int,
    qk_dim: int,
    topk: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool = False,
):
    """Create input data for sparse MLA operator"""
    init_seed(42)
    B = batch_size
    S = seq_len_q
    H = num_heads
    DQK = qk_dim
    SKV = seq_len_kv
    HKV = num_kv_heads

    q = torch.randn((B, S, H, DQK), dtype=dtype, device=device).requires_grad_(
        requires_grad
    )
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device=device).requires_grad_(
        requires_grad
    )

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device=device)
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    return q, kv, indices


def reference_sparse_mla_implementation(q, kv, indices, sm_scale=None, d_v=512):
    """Reference implementation - using provided reference function"""
    return ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=sm_scale, d_v=d_v)


@pytest.mark.sparse_mla_forward
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len_q", [64, 128, 512])
@pytest.mark.parametrize("seq_len_kv", [1024, 2048, 4096])
@pytest.mark.parametrize("num_heads", [64, 128, 256])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("qk_dim", [576])  # Your operator is fixed at 576
@pytest.mark.parametrize("d_v", [512])  # Output dimension
@pytest.mark.parametrize("topk", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_sparse_mla_forward(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    num_kv_heads: int,
    qk_dim: int,
    d_v: int,
    topk: int,
    dtype: torch.dtype,
):
    """Sparse MLA forward propagation test"""
    # Skip unsupported cases
    if num_heads % num_kv_heads != 0:
        pytest.skip("num_heads must be divisible by num_kv_heads")

    if topk > seq_len_kv:
        pytest.skip("topk cannot be larger than seq_len_kv")

    # Create input
    q, kv, indices = make_sparse_mla_input(
        batch_size,
        seq_len_q,
        seq_len_kv,
        num_heads,
        num_kv_heads,
        qk_dim,
        topk,
        dtype,
        device,
    )

    # Reference implementation
    ref_q = to_reference(q, False)
    ref_kv = to_reference(kv, False)
    ref_indices = to_reference(indices, False)

    ref_output = reference_sparse_mla_implementation(
        ref_q, ref_kv, ref_indices, d_v=d_v
    )

    # Your operator implementation
    your_output, your_lse = triton_sparse_mla_fwd_interface(q, kv, indices, d_v=d_v)

    # Accuracy comparison
    gems_assert_close(your_output, ref_output, dtype, atol=1e-2)


@pytest.mark.sparse_mla_forward_edge_cases
@pytest.mark.parametrize(
    "config",
    [
        # Edge case tests
        {
            "batch_size": 1,
            "seq_len_q": 1,
            "seq_len_kv": 1,
            "num_heads": 1,
            "num_kv_heads": 1,
            "topk": 1,
        },
        {
            "batch_size": 1,
            "seq_len_q": 2,
            "seq_len_kv": 100,
            "num_heads": 4,
            "num_kv_heads": 1,
            "topk": 50,
        },
        {
            "batch_size": 1,
            "seq_len_q": 17,
            "seq_len_kv": 1030,
            "num_heads": 8,
            "num_kv_heads": 1,
            "topk": 256,
        },
    ],
)
def test_sparse_mla_forward_edge_cases(config):
    """Sparse MLA edge case tests"""
    dtype = torch.bfloat16
    qk_dim = 576
    d_v = 512

    q, kv, indices = make_sparse_mla_input(
        config["batch_size"],
        config["seq_len_q"],
        config["seq_len_kv"],
        config["num_heads"],
        config["num_kv_heads"],
        qk_dim,
        config["topk"],
        dtype,
        device,
    )

    # Reference implementation
    ref_output = reference_sparse_mla_implementation(
        to_reference(q), to_reference(kv), to_reference(indices), d_v=d_v
    )

    # Your operator implementation
    your_output, your_lse = triton_sparse_mla_fwd_interface(q, kv, indices, d_v=d_v)

    gems_assert_close(your_output, ref_output, dtype, atol=1e-2)


# Device compatibility test
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.sparse_mla_device
def test_sparse_mla_device_compatibility():
    """Test device compatibility"""
    config = {
        "batch_size": 1,
        "seq_len_q": 128,
        "seq_len_kv": 1024,
        "num_heads": 8,
        "num_kv_heads": 1,
        "topk": 64,
    }
    dtype = torch.bfloat16
    qk_dim = 576
    d_v = 512

    q, kv, indices = make_sparse_mla_input(
        config["batch_size"],
        config["seq_len_q"],
        config["seq_len_kv"],
        config["num_heads"],
        config["num_kv_heads"],
        qk_dim,
        config["topk"],
        dtype,
        device,
    )

    # Run on CUDA device
    your_output, your_lse = triton_sparse_mla_fwd_interface(q, kv, indices, d_v=d_v)

    # Verify output shape is correct
    expected_shape = (
        config["batch_size"],
        config["seq_len_q"],
        config["num_heads"],
        d_v,
    )
    assert (
        your_output.shape == expected_shape
    ), f"Output shape incorrect: {your_output.shape} != {expected_shape}"


if __name__ == "__main__":
    # Can directly run this file for testing
    pytest.main([__file__, "-v"])
