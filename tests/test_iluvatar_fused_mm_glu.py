import torch
import pytest
from flag_gems.ops import iluvatar_mm, iluvatar_addmm, iluvatar_bmm, fused_mm_glu

def test_glu_shape_correct():
    x = torch.randn(2, 8, device='cuda')
    y = fused_mm_glu(x)
    assert y.shape == (2, 4)
    ref_a = x[..., :4]
    ref_b = x[..., 4:]
    ref = ref_a * torch.sigmoid(ref_b)
    torch.testing.assert_close(y, ref, atol=1e-2, rtol=1e-2)

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

def test_mm_conv_glu_fused():
    import torch
    from flag_gems.ops import mm_conv_glu_fused
    M, K, N = 32, 16, 8
    x = torch.randn(M, K, device="cuda")
    w = torch.randn(K, N, device="cuda")
    bias = torch.randn(N, device="cuda")
    ref_y = x @ w + bias
    ref = ref_y[..., :N//2] * torch.sigmoid(ref_y[..., N//2:])
    out = mm_conv_glu_fused(x, w, bias)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
