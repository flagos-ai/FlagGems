import torch
import pytest
from flag_gems.ops import iluvatar_mm, iluvatar_bmm, iluvatar_addmm, iluvatar_mm_conv_glu

@pytest.mark.parametrize("M,N,K", [(65,65,64), (1,17,30), (64,64,128)])
def test_mm_non_aligned(M,N,K):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    ref = torch.matmul(a,b)
    out = iluvatar_mm(a,b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("B,M,N,K", [(2,65,33,17), (1,1,1,1), (4,64,64,64)])
def test_bmm_non_aligned(B,M,N,K):
    a = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(B, K, N, device="cuda", dtype=torch.float16)
    ref = torch.bmm(a,b)
    out = iluvatar_bmm(a,b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("M,N,K", [(65,32,17), (64,128,32)])
def test_addmm_interface_order(M,N,K):
    bias = torch.randn(M, N, device="cuda", dtype=torch.float16)
    mat1 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    mat2 = torch.randn(K, N, device="cuda", dtype=torch.float16)
    ref = torch.addmm(bias, mat1, mat2, beta=1.0, alpha=1.0)
    out = iluvatar_addmm(bias, mat1, mat2, beta=1.0, alpha=1.0)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("M,K,Cout", [(2,8,4), (65,32,128), (1,17,60)])
def test_mm_conv_glu_fused_basic(M,K,Cout):
    Cout2 = Cout * 2
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(K, Cout2, device="cuda", dtype=torch.float16)
    b = torch.randn(Cout2, device="cuda", dtype=torch.float16)
    # 参考实现：拆分w,b为a/gate分支
    w_a = w[:, :Cout]
    w_g = w[:, Cout:]
    b_a = b[:Cout]
    b_g = b[Cout:]
    a = x @ w_a + b_a
    g = x @ w_g + b_g
    ref = a * torch.sigmoid(g)
    out = iluvatar_mm_conv_glu(x, w, b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
