"""Correctness test for optimized hc_head_fused Triton kernel."""

import pytest
import torch

from flag_gems.fused.mhc.hc_head_fused_kernel import (
    hc_head_fused_kernel,
    hc_head_fused_kernel_ref,
)


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 64, 256])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_hc_head_fused_kernel_vs_ref(num_tokens, hidden_size, hc_mult, dtype):
    torch.manual_seed(42)

    hs_flat = torch.randn(num_tokens, hc_mult, hidden_size, device="cuda", dtype=dtype)
    fn = torch.randn(hc_mult, hc_mult * hidden_size, device="cuda", dtype=torch.float32) * 0.01
    hc_scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    hc_base = torch.randn(hc_mult, device="cuda", dtype=torch.float32) * 0.1
    rms_eps = 1e-6
    hc_eps = 1e-4

    out_triton = torch.empty(num_tokens, hidden_size, device="cuda", dtype=dtype)
    out_ref = torch.empty(num_tokens, hidden_size, device="cuda", dtype=dtype)

    hc_head_fused_kernel(hs_flat, fn, hc_scale, hc_base, out_triton, hidden_size, rms_eps, hc_eps, hc_mult)
    hc_head_fused_kernel_ref(hs_flat, fn, hc_scale, hc_base, out_ref, hidden_size, rms_eps, hc_eps, hc_mult)

    torch.testing.assert_close(out_triton, out_ref, rtol=2e-2, atol=2e-2)
