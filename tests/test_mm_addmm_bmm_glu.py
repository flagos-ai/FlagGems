import pytest
import torch

from flag_gems.ops.addmm import addmm
from flag_gems.ops.bmm import bmm
from flag_gems.ops.glu import glu
from flag_gems.ops.mm import mm
from flag_gems.ops.mm_conv_glu_fused import mm_conv_glu_fused


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_mm_non_aligned():
    M, N, K = 127, 113, 97
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    ref = torch.matmul(a, b)
    out = mm(a, b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_addmm_correct_order():
    M, N, K = 64, 64, 64
    c = torch.randn(M, N, device="cuda", dtype=torch.float16)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    ref = torch.addmm(c, a, b)
    out = addmm(c, a, b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_bmm_non_aligned():
    B, M, N, K = 2, 31, 29, 27
    a = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(B, K, N, device="cuda", dtype=torch.float16)
    ref = torch.bmm(a, b)
    out = bmm(a, b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_glu_basic():
    M, N = 128, 256
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    x1, x2 = x.chunk(2, dim=-1)
    ref = x1 * torch.sigmoid(x2)
    out = glu(x)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skip(reason="Iluvatar triton backend segfault risk, skip fused")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_mm_conv_glu_fused():
    M, K, N = 128, 128, 256
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(K, N, device="cuda", dtype=torch.float16)
    y = torch.matmul(x, w)
    y1, y2 = y.chunk(2, dim=-1)
    ref = y1 * torch.sigmoid(y2)
    out = mm_conv_glu_fused(x, w)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
