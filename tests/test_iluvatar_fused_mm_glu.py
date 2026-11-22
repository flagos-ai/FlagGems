import torch
import pytest
from flag_gems.ops import iluvatar_mm, iluvatar_addmm, iluvatar_bmm, fused_mm_glu

def test_glu_shape_correct():
    x = torch.randn(2,8, device='cuda')
    y = fused_mm_glu(x)
    assert y.shape == (2,4)

@pytest.mark.parametrize("M,N,K", [
    (65,65,65),
    (64,64,64),
    (1, 17, 31),
])
def test_mm_mask_protect(M,N,K):
    a = torch.randn(M,K, device='cuda')
    b = torch.randn(K,N, device='cuda')
    ref = a @ b
    out = iluvatar_mm(a,b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("M,N,K", [
    (65,65,65),
    (3,15,27),
])
def test_addmm_interface_order(M,N,K):
    bias = torch.randn(M,N, device='cuda')
    m1 = torch.randn(M,K, device='cuda')
    m2 = torch.randn(K,N, device='cuda')
    ref = torch.addmm(bias, m1, m2)
    out = iluvatar_addmm(bias, m1, m2)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

def test_bmm_mask():
    B,M,N,K = 2, 65, 65, 33
    a = torch.randn(B,M,K, device='cuda')
    b = torch.randn(B,K,N, device='cuda')
    ref = torch.bmm(a,b)
    out = iluvatar_bmm(a,b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
