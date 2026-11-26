"""Correctness tests comparing Triton kernels against PyTorch reference.

Run from the repository root:
    PYTHONPATH=src:$PYTHONPATH python tests/test_ascend_mhc.py
    or
    PYTHONPATH=src:$PYTHONPATH python -m tests.test_ascend_mhc
    or with sudo:
    sudo PYTHONPATH=/workspace/FlagGems/src:$PYTHONPATH python tests/test_ascend_mhc.py

On Ascend NPU this expects `torch_npu` importable and CANN environment set.
On CUDA / CPU it will fall back if triton supports the device.

Note: If the mhc module is not found, ensure the package is installed in editable mode
      or set PYTHONPATH to include the src directory.
"""

from __future__ import annotations

import torch
from flag_gems.runtime.backend._ascend.fused.mhc.mhc_post import mhc_post, mhc_post_ref
from flag_gems.runtime.backend._ascend.fused.mhc.mhc_post_backward import mhc_post_backward, mhc_post_backward_ref
from flag_gems.runtime.backend._ascend.fused.mhc.mhc_pre_clamp_sinkhorn import mhc_pre_clamp_sinkhorn, mhc_pre_clamp_sinkhorn_ref
from flag_gems.runtime.backend._ascend.fused.mhc.mhc_pre_clamp_sinkhorn_backward import (
    mhc_pre_clamp_sinkhorn_backward,
    mhc_pre_clamp_sinkhorn_backward_ref,
)

def _pick_device():
    try:
        import torch_npu  # noqa: F401
        return "npu"
    except ImportError:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


def _close(a, b, atol=1e-2, rtol=1e-2, name=""):
    a = a.detach().to(torch.float32).cpu()
    b = b.detach().to(torch.float32).cpu()
    diff = (a - b).abs()
    rel = diff / (b.abs() + 1e-6)
    ok = (diff <= atol + rtol * b.abs()).all().item()
    print(f"  {name:24s} max_abs={diff.max():.4e} max_rel={rel.max():.4e} pass={ok}")
    return ok


def test_mhc_post(device):
    torch.manual_seed(0)
    B, S, N, D = 2, 8, 4, 64
    x = torch.randn(B, S, N, D, dtype=torch.bfloat16, device=device)
    h_res = torch.randn(B, S, N, N, dtype=torch.float32, device=device)
    h_out = torch.randn(B, S, D, dtype=torch.bfloat16, device=device)
    h_post = torch.randn(B, S, N, dtype=torch.float32, device=device)

    y_ref = mhc_post_ref(x, h_res, h_out, h_post)
    y = mhc_post(x, h_res, h_out, h_post)
    print("mhc_post:")
    return _close(y, y_ref, name="y")


def test_mhc_post_backward(device):
    torch.manual_seed(0)
    B, S, N, D = 2, 8, 4, 64
    x = torch.randn(B, S, N, D, dtype=torch.bfloat16, device=device)
    h_res = torch.randn(B, S, N, N, dtype=torch.float32, device=device)
    h_out = torch.randn(B, S, D, dtype=torch.bfloat16, device=device)
    h_post = torch.randn(B, S, N, dtype=torch.float32, device=device)
    grad_y = torch.randn(B, S, N, D, dtype=torch.bfloat16, device=device)

    gx_r, ghres_r, ghout_r, ghpost_r = mhc_post_backward_ref(grad_y, x, h_res, h_out, h_post)
    gx, ghres, ghout, ghpost = mhc_post_backward(grad_y, x, h_res, h_out, h_post)
    print("mhc_post_backward:")
    ok = True
    ok &= _close(gx, gx_r, name="grad_x")
    ok &= _close(ghres, ghres_r, name="grad_h_res")
    ok &= _close(ghout, ghout_r, name="grad_h_out")
    ok &= _close(ghpost, ghpost_r, name="grad_h_post")
    return ok


def test_mhc_pre_clamp_sinkhorn(device):
    torch.manual_seed(0)
    B, S, N, D = 2, 8, 4, 64
    hc_mix = N * (N + 2)
    x = torch.randn(B, S, N, D, dtype=torch.bfloat16, device=device) * 0.1
    phi = torch.randn(hc_mix, N * D, dtype=torch.float32, device=device) * 0.1
    alpha = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=device)
    base = torch.randn(hc_mix, dtype=torch.float32, device=device) * 0.01

    out_ref = mhc_pre_clamp_sinkhorn_ref(x, phi, alpha, base,
                                         norm_eps=1e-6, hc_eps=1e-6,
                                         clamp_min=-3.0, clamp_max=3.0,
                                         iter_times=20)
    out = mhc_pre_clamp_sinkhorn(x, phi, alpha, base,
                                 norm_eps=1e-6, hc_eps=1e-6,
                                 clamp_min=-3.0, clamp_max=3.0,
                                 iter_times=20,
                                 need_backward=False)
    print("mhc_pre_clamp_sinkhorn:")
    ok = True
    ok &= _close(out["y"], out_ref["y"], name="y")
    ok &= _close(out["post_out"], out_ref["post_out"], name="post_out")
    ok &= _close(out["comb_frag"], out_ref["comb_frag"], name="comb_frag")
    return ok


def test_mhc_pre_clamp_sinkhorn_backward(device):
    torch.manual_seed(0)
    B, S, N, D = 2, 8, 4, 64
    hc_mix = N * (N + 2)
    x = torch.randn(B, S, N, D, dtype=torch.bfloat16, device=device) * 0.1
    phi = torch.randn(hc_mix, N * D, dtype=torch.float32, device=device) * 0.1
    alpha = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=device)
    base = torch.randn(hc_mix, dtype=torch.float32, device=device) * 0.01

    # grad_y matches aclnn "hin" contract: shape (B, S, D).
    grad_y = torch.randn(B, S, D, dtype=torch.bfloat16, device=device)
    grad_post_out = torch.randn(B * S, N, dtype=torch.float32, device=device)
    grad_comb_frag = torch.randn(B * S, N, N, dtype=torch.float32, device=device)

    out = mhc_pre_clamp_sinkhorn(x, phi, alpha, base,
                                 norm_eps=1e-6, hc_eps=1e-6,
                                 clamp_min=-3.0, clamp_max=3.0,
                                 iter_times=20,
                                 need_backward=True)
    gx, gphi, ga, gb = mhc_pre_clamp_sinkhorn_backward(
        x, phi, alpha, base,
        out["inv_rms"], out["x_scaled"], out["mixes"],
        out.get("h_res_logits"), out["pre"],
        grad_y, grad_post_out, grad_comb_frag,
        norm_eps=1e-6, hc_eps=1e-6,
        clamp_min=-3.0, clamp_max=3.0, iter_times=20,
    )
    gx_r, gphi_r, ga_r, gb_r = mhc_pre_clamp_sinkhorn_backward_ref(
        x, phi, alpha, base,
        grad_y, grad_post_out, grad_comb_frag,
        norm_eps=1e-6, hc_eps=1e-6,
        clamp_min=-3.0, clamp_max=3.0, iter_times=20,
    )
    print("mhc_pre_clamp_sinkhorn_backward:")
    ok = True
    ok &= _close(gx, gx_r, name="grad_x", atol=5e-2, rtol=5e-2)
    ok &= _close(gphi, gphi_r, name="grad_phi", atol=5e-2, rtol=5e-2)
    ok &= _close(ga, ga_r, name="grad_alpha", atol=5e-2, rtol=5e-2)
    ok &= _close(gb, gb_r, name="grad_base", atol=5e-2, rtol=5e-2)
    return ok


if __name__ == "__main__":
    device = _pick_device()
    print(f"[device] {device}")
    results = {}
    for name, fn in [
        ("mhc_post", test_mhc_post),
        ("mhc_post_backward", test_mhc_post_backward),
        ("mhc_pre_clamp_sinkhorn", test_mhc_pre_clamp_sinkhorn),
        ("mhc_pre_clamp_sinkhorn_backward", test_mhc_pre_clamp_sinkhorn_backward),
    ]:
        try:
            results[name] = fn(device)
        except Exception as e:
            print(f"{name}: FAILED with {type(e).__name__}: {e}")
            results[name] = False
    print("\nSummary:")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
