import torch
from flag_gems.ops import iluvatar_mm, iluvatar_bmm, iluvatar_addmm, iluvatar_mm_glu

torch.manual_seed(42)
device = torch.device("cuda")

def test_mm_non_aligned():
    a = torch.randn(65, 33, device=device, dtype=torch.float16)
    b = torch.randn(33, 65, device=device, dtype=torch.float16)
    ref = torch.matmul(a, b)
    out = iluvatar_mm(a, b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("✅ test_mm_non_aligned pass")

def test_bmm_non_aligned():
    a = torch.randn(2, 65, 33, device=device, dtype=torch.float16)
    b = torch.randn(2, 33, 65, device=device, dtype=torch.float16)
    ref = torch.bmm(a, b)
    out = iluvatar_bmm(a, b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("✅ test_bmm_non_aligned pass")

def test_addmm_correct_order():
    bias = torch.randn(64, 128, device=device, dtype=torch.float16)
    mat1 = torch.randn(64, 32, device=device, dtype=torch.float16)
    mat2 = torch.randn(32, 128, device=device, dtype=torch.float16)
    ref = torch.addmm(bias, mat1, mat2, beta=0.5, alpha=2.0)
    out = iluvatar_addmm(bias, mat1, mat2, beta=0.5, alpha=2.0)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("✅ test_addmm_correct_order pass")

def test_mm_glu_shape_logic():
    x = torch.randn(2, 8, device=device, dtype=torch.float16)
    w = torch.randn(8, 8, device=device, dtype=torch.float16)
    out = iluvatar_mm_glu(x, w)
    assert out.shape == (2, 4), f"GLU shape wrong {out.shape}"
    raw = x @ w
    x1, x2 = raw.chunk(2, dim=-1)
    ref = x1 * torch.sigmoid(x2)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("✅ test_mm_glu_shape_logic pass")

if __name__ == "__main__":
    test_mm_non_aligned()
    test_bmm_non_aligned()
    test_addmm_correct_order()
    test_mm_glu_shape_logic()
    print("\n🎉 All Iluvatar mm/bmm/addmm/mm-glu tests passed!")
