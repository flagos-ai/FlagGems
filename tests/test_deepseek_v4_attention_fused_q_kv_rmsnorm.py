import pytest
import torch

import flag_gems.testing as fg_testing
from flag_gems.fused.deepseek_v4_attention_fused_q_kv_rmsnorm import fused_q_kv_rmsnorm


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _rmsnorm_ref(x, weight, eps):
    y = x.float() * torch.rsqrt(
        torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps
    )
    return (y * weight.float()).to(x.dtype)


@pytest.mark.skipif(not _has_cuda(), reason="requires cuda")
def test_fused_q_kv_rmsnorm_accuracy():
    device = "cuda"
    qr = torch.randn((4, 256), device=device, dtype=torch.bfloat16)
    kv = torch.randn((4, 128), device=device, dtype=torch.bfloat16)
    q_weight = torch.randn((256,), device=device, dtype=torch.bfloat16)
    kv_weight = torch.randn((128,), device=device, dtype=torch.bfloat16)
    eps = 1e-6

    q_out, kv_out = fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)

    fg_testing.assert_close(
        q_out, _rmsnorm_ref(qr, q_weight, eps), dtype=torch.bfloat16, equal_nan=True
    )
    fg_testing.assert_close(
        kv_out, _rmsnorm_ref(kv, kv_weight, eps), dtype=torch.bfloat16, equal_nan=True
    )
