import pytest
import torch

import flag_gems.testing as fg_testing
from flag_gems.fused.deepseek_v4_attention_fused_q_kv_rmsnorm import fused_q_kv_rmsnorm

try:
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        fused_q_kv_rmsnorm as vllm_fused_q_kv_rmsnorm,
    )

    _HAS_VLLM_FUSED_Q_KV_RMSNORM = True
except Exception:
    vllm_fused_q_kv_rmsnorm = None
    _HAS_VLLM_FUSED_Q_KV_RMSNORM = False


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _rmsnorm_ref(x, weight, eps):
    y = x.float() * torch.rsqrt(
        torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps
    )
    return (y * weight.float()).to(x.dtype)


@pytest.mark.parametrize(
    ("tokens", "qdim", "kvdim"),
    [
        (4, 256, 128),
        (8, 1536, 512),
    ],
)
@pytest.mark.skipif(not _has_cuda(), reason="requires cuda")
def test_fused_q_kv_rmsnorm_accuracy(tokens, qdim, kvdim):
    device = "cuda"
    qr = torch.randn((tokens, qdim), device=device, dtype=torch.bfloat16)
    kv = torch.randn((tokens, kvdim), device=device, dtype=torch.bfloat16)
    q_weight = torch.randn((qdim,), device=device, dtype=torch.bfloat16)
    kv_weight = torch.randn((kvdim,), device=device, dtype=torch.bfloat16)
    eps = 1e-6

    q_out, kv_out = fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)

    fg_testing.assert_close(
        q_out, _rmsnorm_ref(qr, q_weight, eps), dtype=torch.bfloat16, equal_nan=True
    )
    fg_testing.assert_close(
        kv_out, _rmsnorm_ref(kv, kv_weight, eps), dtype=torch.bfloat16, equal_nan=True
    )


@pytest.mark.skipif(
    (not _has_cuda()) or (not _HAS_VLLM_FUSED_Q_KV_RMSNORM),
    reason="requires cuda and vllm deepseek_v4_ops.fused_q_kv_rmsnorm",
)
def test_fused_q_kv_rmsnorm_vllm_accuracy():
    device = "cuda"
    qr = torch.randn((8, 1536), device=device, dtype=torch.bfloat16)
    kv = torch.randn((8, 512), device=device, dtype=torch.bfloat16)
    q_weight = torch.randn((1536,), device=device, dtype=torch.bfloat16)
    kv_weight = torch.randn((512,), device=device, dtype=torch.bfloat16)
    eps = 1e-6

    q_out, kv_out = fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)
    q_expected, kv_expected = vllm_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)

    fg_testing.assert_close(q_out, q_expected, dtype=torch.bfloat16, equal_nan=True)
    fg_testing.assert_close(kv_out, kv_expected, dtype=torch.bfloat16, equal_nan=True)
