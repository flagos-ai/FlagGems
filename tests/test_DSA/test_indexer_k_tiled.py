import random

import numpy as np
import pytest
import torch

from flag_gems.fused.DSA.indexer_k_tiled import (
    triton_lighting_indexer_k_tiled_interface,
)

from .torch_src.fp8_lighting_indexer import (
    ref_fp8_mqa_logits,  # , mqa_attn_return_logits_interface
)
from .utils import generate_random_cu_seqlens


# Accuracy check function
def display_error_message(msg):
    print(f"\033[31mWARNING: {msg}\033[0m")


def compute_correlation(a, b, label="tensor"):
    a, b = a.data.double(), b.data.double()
    norm_sum = (a * a + b * b).sum()
    if norm_sum == 0:
        display_error_message(f"{label} all zero")
        return 1
    correlation = 2 * (a * b).sum() / norm_sum
    return correlation


def assert_close_inf(
    a, b, tolerance=1e-8, tensor_name="tensor", should_raise=True, ks=None, ke=None
):
    a_finite = torch.isfinite(a)
    b_finite = torch.isfinite(b)
    if not torch.all(a_finite == b_finite):
        mismatch_mask = a_finite != b_finite
        mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False)

        # Display first few inconsistent positions
        num_to_show = min(10000, len(mismatch_indices))
        display_error_message(
            f"{tensor_name} Error: isfinite mask mismatch: {torch.sum(mismatch_mask)} positions"
        )

        for i in range(num_to_show):
            idx = mismatch_indices[i]
            coord = tuple(idx.tolist())
            a_val = a[coord].item()
            b_val = b[coord].item()
            a_finite_val = a_finite[coord].item()
            b_finite_val = b_finite[coord].item()
            idx_val = idx.tolist()[0]
            ks_val = ks[idx_val]
            ke_val = ke[idx_val]
            display_error_message(
                f"  Position {coord}: "
                f"a={a_val:.6f} (finite={a_finite_val}), "
                f"b={b_val:.6f} (finite={b_finite_val}), "
                f"ks={ks_val}, ke={ke_val}"
            )

        if len(mismatch_indices) > num_to_show:
            display_error_message(
                f"  ... and {len(mismatch_indices) - num_to_show} more positions"
            )

        if should_raise:
            assert False
    if not torch.isclose(
        a.masked_fill(a_finite, 0),
        b.masked_fill(b_finite, 0),
        rtol=0,
        atol=0,
        equal_nan=True,
    ).all():
        display_error_message(f"{tensor_name} Error: nonfinite value mismatch")
        if should_raise:
            assert False
    a = a.masked_fill(~a_finite, 0)
    b = b.masked_fill(~b_finite, 0)
    correlation = compute_correlation(a, b, tensor_name)
    difference = 1.0 - correlation
    if not (0 <= difference <= tolerance):
        display_error_message(f"{tensor_name} Error: {difference}")
        if should_raise:
            assert False
    return difference


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


def make_lighting_indexer_input(
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    qk_dim: int,
    detype: torch.dtype,
    device: torch.device,
    kv_stride: int = 1,
):
    """Create input data for sparse MLA operator"""
    init_seed(42)
    S = seq_len_q
    H = num_heads
    D = qk_dim
    SKV = seq_len_kv
    q = torch.randn((S, H, D), dtype=detype, device=device).requires_grad_(False)
    kv = torch.randn((SKV, D), dtype=detype, device=device).requires_grad_(False)
    weights = torch.randn((S, H), dtype=torch.float32, device=device).requires_grad_(
        False
    )
    # p = (torch.randn(S, SKV, device="cuda", dtype=torch.float32) * 4).softmax(dim=-1)

    ks, ke = generate_random_cu_seqlens(
        per_cp_seqlen=S, cp_size=3, cp_rank=4, kv_stride=kv_stride, average_q_len=2048
    )

    return q, kv, weights, ks, ke


def reference_lighting_indexer_implementation(q, kv, weights, ks, ke):
    """Reference implementation - using provided reference function"""
    return ref_fp8_mqa_logits(
        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
    )


@pytest.mark.lighting_indexer_forward
@pytest.mark.parametrize("seq_len_q", [1024, 2048, 4096])
@pytest.mark.parametrize("seq_len_kv", [2048, 4096, 8192])
@pytest.mark.parametrize("num_heads", [16, 32, 64])
@pytest.mark.parametrize("qk_dim", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_lighting_indexer_forward(
    seq_len_q: int, seq_len_kv: int, num_heads: int, qk_dim: int, dtype: torch.dtype
):
    # Create input
    q, kv, weights, ks, ke = make_lighting_indexer_input(
        seq_len_q, seq_len_kv, num_heads, qk_dim, dtype, device
    )

    # Reference implementation
    ref_q = to_reference(q, False)
    ref_kv = to_reference(kv, False)
    ref_weights = to_reference(weights, False)

    ref_output, cost_ref = reference_lighting_indexer_implementation(
        ref_q, ref_kv, ref_weights, ks, ke
    )

    # Your operator implementation
    your_output = triton_lighting_indexer_k_tiled_interface(q, kv, weights, ks, ke)

    # Accuracy comparison
    assert_close_inf(your_output, ref_output, 1e-2)


if __name__ == "__main__":
    # Can directly run this file for testing
    pytest.main([__file__, "-v"])
