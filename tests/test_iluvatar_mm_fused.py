import pytest
import torch

from flag_gems.ops import addmm, bmm, glu, mm, mm_conv_glu


def test_mm_non64():
    torch.manual_seed(0)
    a = torch.randn(65, 33, device="cuda", dtype=torch.float16)
    b = torch.randn(33, 65, device="cuda", dtype=torch.float16)
    ref = a @ b
    res = mm(a, b)
    torch.testing.assert_close(res, ref, atol=1e-2, rtol=1e-2)


def test_addmm_order_non64():
    torch.manual_seed(0)
    bias = torch.randn(65, 45, device="cuda", dtype=torch.float16)
    m1 = torch.randn(65, 27, device="cuda", dtype=torch.float16)
    m2 = torch.randn(27, 45, device="cuda", dtype=torch.float16)
    ref = torch.addmm(bias, m1, m2, beta=1.0, alpha=1.0)
    res = addmm(bias, m1, m2)
    torch.testing.assert_close(res, ref, atol=1e-2, rtol=1e-2)


def test_bmm_non64():
    torch.manual_seed(0)
    a = torch.randn(3, 65, 31, device="cuda", dtype=torch.float16)
    b = torch.randn(3, 31, 55, device="cuda", dtype=torch.float16)
    ref = torch.bmm(a, b)
    res = bmm(a, b)
    torch.testing.assert_close(res, ref, atol=1e-2, rtol=1e-2)


def test_glu_shape_logic():
    torch.manual_seed(0)
    x = torch.randn(2, 8, device="cuda", dtype=torch.float16)
    x0, x1 = x.chunk(2, dim=-1)
    ref = x0 * torch.sigmoid(x1)
    res = glu(x)
    torch.testing.assert_close(res, ref, atol=1e-2, rtol=1e-2)


def test_mm_conv_glu_fused():
    torch.manual_seed(0)
    M, K, Cout2 = 16, 32, 8
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(K, Cout2 * 2, device="cuda", dtype=torch.float16)
    w1, w2 = w.split(Cout2, dim=-1)
    ref = (a @ w1) * torch.sigmoid(a @ w2)
    res = mm_conv_glu(a, w)
    torch.testing.assert_close(res, ref, atol=1e-2, rtol=1e-2)
