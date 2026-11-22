import torch
import pytest
from flag_gems.ops import mm_conv_glu_fused, iluvatar_addmm, iluvatar_mm, iluvatar_bmm

def test_mm_conv_glu_fused_basic():
    M, K, D = 32, 16, 8
    x = torch.randn(M, K, dtype=torch.float32)
    w = torch.randn(K, D, dtype=torch.float32)
    bias = torch.randn(D, dtype=torch.float32)
    ref_y = x @ w + bias
    ref = ref_y[..., :D//2] * torch.sigmoid(ref_y[..., D//2:])
    out = mm_conv_glu_fused(x, w, bias)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("M,K,D", [(65,65,64), (1,17,30), (64,64,128)])
def test_mm_conv_glu_non_aligned(M,K,D):
    if D % 2 !=0:
        D +=1
    x = torch.randn(M, K, dtype=torch.float32)
    w = torch.randn(K, D, dtype=torch.float32)
    bias = torch.randn(D, dtype=torch.float32)
    ref_y = x @ w + bias
    ref = ref_y[..., :D//2] * torch.sigmoid(ref_y[..., D//2:])
    out = mm_conv_glu_fused(x, w, bias)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

def test_addmm_interface_order():
    # FlagGems规范顺序 addmm(bias, mat1, mat2)
    M,N,K = 64,128,32
    bias = torch.randn(M,N)
    m1 = torch.randn(M,K)
    m2 = torch.randn(K,N)
    beta, alpha = 0.5,1.2
    ref = beta * bias + alpha * (m1 @ m2)
    out = iluvatar_addmm(bias, m1, m2, beta, alpha)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

def test_mm_non_aligned():
    a = torch.randn(65, 33)
    b = torch.randn(33, 71)
    ref = a @ b
    out = iluvatar_mm(a,b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

def test_bmm_non_aligned():
    a = torch.randn(2, 65, 33)
    b = torch.randn(2, 33, 71)
    ref = torch.bmm(a,b)
    out = iluvatar_bmm(a,b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
