import torch

from src.flag_gems.ops.triton_src.weight_norm import weight_norm_triton


def test_weight_norm(dtype=torch.float32, out_dim=128, in_dim=512):
    torch.manual_seed(0)
    v = torch.randn(out_dim, in_dim, device="cuda", dtype=dtype, requires_grad=True)
    g = torch.randn(out_dim, device="cuda", dtype=dtype, requires_grad=True)

    # triton
    w_tr = weight_norm_triton(v, g)
    loss_tr = w_tr.sum()
    loss_tr.backward()
    dv_tr = v.grad.clone()
    dg_tr = g.grad.clone()
    v.grad.zero_()
    g.grad.zero_()

    # torch reference
    norm_v = torch.norm(v, p=2, dim=1, keepdim=True)
    norm_v = torch.clamp(norm_v, min=1e-12)
    w_ref = g[:, None] * v / norm_v
    loss_ref = w_ref.sum()
    loss_ref.backward()

    diff_w = (w_tr - w_ref).abs()
    diff_dv = (dv_tr - v.grad).abs()
    diff_dg = (dg_tr - g.grad).abs()

    max_w = diff_w.max().item()
    max_dv = diff_dv.max().item()
    max_dg = diff_dg.max().item()

    print(f"\n==== Test dtype={dtype}, out={out_dim}, in={in_dim} ====")
    print(f"w     max diff: {max_w:.2e}")
    print(f"dv    max diff: {max_dv:.2e}")
    print(f"dg    max diff: {max_dg:.2e}")

    tol = 1e-5 if dtype == torch.float32 else 1e-2
    assert max_w < tol, f"w forward error exceed tol {tol}"
    assert max_dv < tol, f"dv grad error exceed tol {tol}"
    assert max_dg < tol, f"dg grad error exceed tol {tol}"
    print("✅ PASS")


if __name__ == "__main__":
    test_weight_norm(torch.float32, out_dim=128, in_dim=512)
    test_weight_norm(torch.bfloat16, out_dim=128, in_dim=512)
    test_weight_norm(torch.float32, out_dim=8, in_dim=32)
