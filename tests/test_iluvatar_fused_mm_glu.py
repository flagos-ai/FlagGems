import torch
import pytest
from flag_gems.ops import iluvatar_mm, iluvatar_bmm, iluvatar_addmm, iluvatar_mm_conv_glu

@pytest.mark.parametrize("M,K,N", [(65,65,64), (1,17,30), (64,64,128)])
def test_mm_non_aligned(M,K,N):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    ref = torch.matmul(a,b)
    res = iluvatar_mm(a,b)
    torch.testing.assert_close(res, ref, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("B,M,K,N", [(2,65,33,17), (1,1,1,1), (4,64,64,64)])
def test_bmm_non_aligned(B,M,K,N):
    a = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(B, K, N, device="cuda", dtype=torch.float16)
    ref = torch.bmm(a,b)
    res = iluvatar_bmm(a,b)
    torch.testing.assert_close(res, ref, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("M,K,N", [(65,32,17), (64,128,32)])
def test_addmm_interface_order(M,K,N):
    bias = torch.randn(M, N, device="cuda", dtype=torch.float16)
    mat1 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    mat2 = torch.randn(K, N, device="cuda", dtype=torch.float16)
    ref = torch.addmm(bias, mat1, mat2)
    res = iluvatar_addmm(bias, mat1, mat2)
    torch.testing.assert_close(res, ref, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("M,K,O", [(2,8,4), (65,32,128), (1,17,60)])
def test_mm_conv_glu_fused_basic(M,K,O):
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    weight = torch.randn(K, O*2, device="cuda", dtype=torch.float16)
    mm = torch.matmul(x, weight)
    a_ref, b_ref = mm.split(O, dim=-1)
    ref = a_ref * torch.sigmoid(b_ref)
    res = iluvatar_mm_conv_glu(x, weight)
    torch.testing.assert_close(res, ref, atol=1e-2, rtol=1e-2)
