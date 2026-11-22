import torch
import pytest
from flag_gems.ops import mm_conv_glu_fused, iluvatar_addmm, iluvatar_mm, iluvatar_bmm

HAS_ILUVATAR = False
DEV = None
try:
    dev = torch.device("privateuseone:0")
    _tmp = torch.empty((1,), device=dev)
    HAS_ILUVATAR = True
    DEV = dev
except Exception:
    HAS_ILUVATAR = False

@pytest.mark.skipif(not HAS_ILUVATAR, reason="No Iluvatar PrivateUse1 device available")
def test_mm_conv_glu_fused_basic():
    torch.manual_seed(42)
    M, K, D = 32, 16, 8
    x_cpu = torch.randn(M, K, dtype=torch.float32)
    w_cpu = torch.randn(K, D, dtype=torch.float32)
    bias_cpu = torch.randn(D, dtype=torch.float32)
    ref_y = x_cpu @ w_cpu + bias_cpu
    ref = ref_y[..., :D//2] * torch.sigmoid(ref_y[..., D//2:])

    x = x_cpu.to(DEV).contiguous()
    w = w_cpu.to(DEV).contiguous()
    bias = bias_cpu.to(DEV).contiguous()
    out = mm_conv_glu_fused(x, w, bias)
    out_cpu = out.to("cpu")
    torch.testing.assert_close(out_cpu, ref, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("M,K,D", [(65,65,64), (1,17,30), (64,64,128)])
@pytest.mark.skipif(not HAS_ILUVATAR, reason="No Iluvatar PrivateUse1 device available")
def test_mm_conv_glu_non_aligned(M,K,D):
    torch.manual_seed(42)
    x_cpu = torch.randn(M, K, dtype=torch.float32)
    w_cpu = torch.randn(K, D, dtype=torch.float32)
    bias_cpu = torch.randn(D, dtype=torch.float32)
    ref_y = x_cpu @ w_cpu + bias_cpu
    ref = ref_y[..., :D//2] * torch.sigmoid(ref_y[..., D//2:])

    x = x_cpu.to(DEV).contiguous()
    w = w_cpu.to(DEV).contiguous()
    bias = bias_cpu.to(DEV).contiguous()
    out = mm_conv_glu_fused(x, w, bias)
    out_cpu = out.to("cpu")
    torch.testing.assert_close(out_cpu, ref, atol=1e-2, rtol=1e-2)

@pytest.mark.skipif(not HAS_ILUVATAR, reason="No Iluvatar PrivateUse1 device available")
def test_addmm_interface_order():
    torch.manual_seed(42)
    M,N,K = 64,128,32
    bias_cpu = torch.randn(N, dtype=torch.float32)
    m1_cpu = torch.randn(M,K,dtype=torch.float32)
    m2_cpu = torch.randn(K,N,dtype=torch.float32)
    ref = torch.addmm(bias_cpu, m1_cpu, m2_cpu, beta=1.0, alpha=1.0)

    bias = bias_cpu.to(DEV)
    m1 = m1_cpu.to(DEV)
    m2 = m2_cpu.to(DEV)
    out = iluvatar_addmm(bias, m1, m2)
    torch.testing.assert_close(out.cpu(), ref, atol=1e-2, rtol=1e-2)

@pytest.mark.skipif(not HAS_ILUVATAR, reason="No Iluvatar PrivateUse1 device available")
def test_mm_non_aligned():
    torch.manual_seed(42)
    a_cpu = torch.randn(65,37)
    b_cpu = torch.randn(37,71)
    ref = a_cpu @ b_cpu
    a = a_cpu.to(DEV)
    b = b_cpu.to(DEV)
    out = iluvatar_mm(a,b)
    torch.testing.assert_close(out.cpu(), ref, atol=1e-2, rtol=1e-2)

@pytest.mark.skipif(not HAS_ILUVATAR, reason="No Iluvatar PrivateUse1 device available")
def test_bmm_non_aligned():
    torch.manual_seed(42)
    a_cpu = torch.randn(3,65,37)
    b_cpu = torch.randn(3,37,71)
    ref = a_cpu @ b_cpu
    a = a_cpu.to(DEV)
    b = b_cpu.to(DEV)
    out = iluvatar_bmm(a,b)
    torch.testing.assert_close(out.cpu(), ref, atol=1e-2, rtol=1e-2)
