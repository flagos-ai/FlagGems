import torch
from flag_gems.ops import iluvatar_mm, iluvatar_bmm, iluvatar_addmm, iluvatar_mm_conv_glu

def test_mm_non_aligned():
    M,N,K = 65,65,65
    a = torch.randn(M,K, device="cuda", dtype=torch.float16)
    b = torch.randn(K,N, device="cuda", dtype=torch.float16)
    ref = torch.matmul(a,b)
    out = iluvatar_mm(a,b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("✅ test_mm_non_aligned pass")

def test_bmm_non_aligned():
    B,M,N,K = 2,65,65,33
    a = torch.randn(B,M,K, device="cuda", dtype=torch.float16)
    b = torch.randn(B,K,N, device="cuda", dtype=torch.float16)
    ref = torch.bmm(a,b)
    out = iluvatar_bmm(a,b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("✅ test_bmm_non_aligned pass")

def test_addmm_correct_order():
    bias = torch.randn(64,128, device="cuda", dtype=torch.float16)
    mat1 = torch.randn(64,32, device="cuda", dtype=torch.float16)
    mat2 = torch.randn(32,128, device="cuda", dtype=torch.float16)
    ref = torch.addmm(bias, mat1, mat2)
    out = iluvatar_addmm(bias, mat1, mat2)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("✅ test_addmm_correct_order pass")

def test_mm_conv_glu_shape_logic():
    x = torch.randn(2,8, device="cuda", dtype=torch.float16)
    w = torch.randn(8,8, device="cuda", dtype=torch.float16)
    ref_raw = x @ w
    ref_val = ref_raw[:, :4]
    ref_gate = ref_raw[:, 4:]
    ref_glu = ref_val * torch.sigmoid(ref_gate)
    out = iluvatar_mm_conv_glu(x, w)
    assert out.shape == (2,4), f"shape wrong {out.shape}"
    torch.testing.assert_close(out, ref_glu, atol=1e-2, rtol=1e-2)
    print("✅ test_mm_conv_glu_shape_logic pass")

if __name__ == "__main__":
    test_mm_non_aligned()
    test_bmm_non_aligned()
    test_addmm_correct_order()
    test_mm_conv_glu_shape_logic()
    print("\n🎉 All Iluvatar mm/bmm/addmm/mm-conv-glu tests passed!")
