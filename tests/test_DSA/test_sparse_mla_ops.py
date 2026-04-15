"""
Sparse MLA accuracy tests (vLLM-compatible 3D interface).

Compares FlagGems Triton kernel against vLLM flash-mla CUDA kernel.

Model configurations:
- GLM-5 CONFIG1: d_qk=512, h_q=64, topk=512
- GLM-5 CONFIG2: d_qk=512, h_q=128, topk=1024
- DeepSeek-V3.2: d_qk=576, h_q=128, topk=2048

Usage:
    pytest tests/test_DSA/test_sparse_mla_ops.py -v
"""

import pytest
import torch
import vllm._flashmla_C  # noqa: F401

from flag_gems.fused.DSA.sparse_mla import sparse_prefill_fwd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# vLLM CUDA baseline
_flash_mla_cuda = torch.ops._flashmla_C


def vllm_sparse_prefill_fwd(q, kv, indices, sm_scale, d_v):
    """vLLM flash-mla CUDA sparse prefill forward (baseline)."""
    result = _flash_mla_cuda.sparse_prefill_fwd(
        q, kv, indices, sm_scale, d_v, None, None
    )
    return result[0], result[1], result[2]


# ============================================================
# Accuracy tests
# ============================================================


@pytest.mark.sparse_mla_forward
@pytest.mark.parametrize(
    "d_qk,h_q,topk,s_kv",
    [
        # GLM-5 CONFIG1: d_qk=512, h_q=64, topk=512
        (512, 64, 512, 8192),
        (512, 64, 512, 32768),
        (512, 64, 512, 49152),
        (512, 64, 512, 65536),
        # GLM-5 CONFIG2: d_qk=512, h_q=128, topk=1024
        (512, 128, 1024, 8192),
        (512, 128, 1024, 32768),
        (512, 128, 1024, 49152),
        (512, 128, 1024, 65536),
        # DeepSeek-V3.2: d_qk=576, h_q=128, topk=2048
        (576, 128, 2048, 8192),
        (576, 128, 2048, 32768),
        (576, 128, 2048, 65536),
        (576, 128, 2048, 98304),
        (576, 128, 2048, 131072),
    ],
)
@pytest.mark.parametrize("s_q", [1, 4096])
def test_sparse_mla_accuracy(s_q, d_qk, h_q, topk, s_kv):
    """Sparse MLA accuracy: Triton kernel vs vLLM CUDA kernel."""
    h_kv = 1
    d_v = 512
    dtype = torch.bfloat16
    sm_scale = d_qk ** (-0.5)
    actual_topk = min(topk, s_kv)

    torch.manual_seed(42)
    q = torch.randn(s_q, h_q, d_qk, dtype=dtype, device=device)
    kv = torch.randn(s_kv, h_kv, d_qk, dtype=dtype, device=device)
    indices = torch.randint(
        0, s_kv, (s_q, h_kv, actual_topk), dtype=torch.int32, device=device
    )

    # vLLM CUDA baseline
    ref_out, _, ref_lse = vllm_sparse_prefill_fwd(q, kv, indices, sm_scale, d_v)

    # Triton kernel
    act_out, _, act_lse = sparse_prefill_fwd(q, kv, indices, sm_scale, d_v)

    # Check correctness
    assert not torch.isnan(act_out).any(), "Output contains NaN"
    max_diff = (ref_out.float() - act_out.float()).abs().max().item()
    assert max_diff < 1e-2, f"Output max diff {max_diff:.6f} exceeds tolerance 1e-2"

    lse_diff = (ref_lse - act_lse).abs().max().item()
    assert lse_diff < 0.1, f"LSE max diff {lse_diff:.6f} exceeds tolerance 0.1"


# ============================================================
# Edge case tests
# ============================================================


@pytest.mark.sparse_mla_forward_edge_cases
@pytest.mark.parametrize(
    "config",
    [
        {"s_q": 1, "s_kv": 1024, "h_q": 64, "topk": 512, "d_qk": 576},
        {"s_q": 17, "s_kv": 2048, "h_q": 64, "topk": 512, "d_qk": 576},
        {"s_q": 128, "s_kv": 4096, "h_q": 128, "topk": 512, "d_qk": 512},
    ],
)
def test_sparse_mla_edge_cases(config):
    """Sparse MLA edge cases: Triton kernel vs vLLM CUDA kernel."""
    d_v = 512
    dtype = torch.bfloat16
    sm_scale = config["d_qk"] ** (-0.5)

    torch.manual_seed(42)
    q = torch.randn(
        config["s_q"], config["h_q"], config["d_qk"], dtype=dtype, device=device
    )
    kv = torch.randn(config["s_kv"], 1, config["d_qk"], dtype=dtype, device=device)
    actual_topk = min(config["topk"], config["s_kv"])
    indices = torch.randint(
        0,
        config["s_kv"],
        (config["s_q"], 1, actual_topk),
        dtype=torch.int32,
        device=device,
    )

    ref_out, _, _ = vllm_sparse_prefill_fwd(q, kv, indices, sm_scale, d_v)
    act_out, _, _ = sparse_prefill_fwd(q, kv, indices, sm_scale, d_v)

    assert not torch.isnan(act_out).any(), "Output contains NaN"
    max_diff = (ref_out.float() - act_out.float()).abs().max().item()
    assert max_diff < 1e-2, f"Output max diff {max_diff:.6f} exceeds tolerance 1e-2"
