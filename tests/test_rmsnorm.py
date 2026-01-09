import torch
import pytest

from flag_gems.ops.rmsnorm import rmsnorm


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Triton RMSNorm"
)
def test_rmsnorm_correctness():
    def rmsnorm_ref(x, w, eps=1e-6):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return (x / rms) * w

    x = torch.randn(32, 256, device="cuda")
    w = torch.randn(256, device="cuda")

    y1 = rmsnorm(x, w)
    y2 = rmsnorm_ref(x, w)

    assert torch.max(torch.abs(y1 - y2)) < 1e-3

