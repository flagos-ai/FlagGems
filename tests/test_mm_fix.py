import torch
import pytest
from flag_gems.ops import mm, addmm, bmm, glu

def test_mm_non_aligned():
    torch.manual_seed(0)
    a = torch.randn(65, 33, device="cuda")
    b = torch.randn(33, 65, device="cuda")
    ref = a @ b
    res = mm(a,b)
    torch.testing.assert_close(res, ref, atol=1e-3, rtol=1e-3)

def test_addmm_correct_order():
    torch.manual_seed(0)
    bias = torch.randn(64,128, device="cuda")
    mat1 = torch.randn(64,32, device="cuda")
    mat2 = torch.randn(32,128, device="cuda")
    ref = torch.addmm(bias, mat1, mat2, beta=1.0, alpha=1.0)
    res = addmm(bias, mat1, mat2, beta=1.0, alpha=1.0)
    torch.testing.assert_close(res, ref, atol=1e-3, rtol=1e-3)

def test_bmm_non_aligned():
    torch.manual_seed(0)
    a = torch.randn(2,65,41, device="cuda")
    b = torch.randn(2,41,65, device="cuda")
    ref = torch.bmm(a,b)
    res = bmm(a,b)
    torch.testing.assert_close(res, ref, atol=1e-3, rtol=1e-3)

def test_glu_shape_and_calc():
    torch.manual_seed(0)
    x = torch.randn(2,8, device="cuda")
    res = glu(x)
    assert res.shape == (2,4)
    a,b = x.chunk(2, dim=-1)
    ref = a * torch.sigmoid(b)
    torch.testing.assert_close(res, ref, atol=1e-3, rtol=1e-3)

if __name__ == "__main__":
    test_mm_non_aligned()
    print("✅ test_mm_non_aligned pass")
    test_addmm_correct_order()
    print("✅ test_addmm_correct_order pass")
    test_bmm_non_aligned()
    print("✅ test_bmm_non_aligned pass")
    test_glu_shape_and_calc()
    print("✅ test_glu_shape_and_calc pass")
