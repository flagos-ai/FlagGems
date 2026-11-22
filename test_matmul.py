import pytest
import torch

from addmm import addmm
from bmm import bmm
from mm import mm

torch.manual_seed(42)
DEV = torch.device("cuda")
ATOL = 1e-5
RTOL = 1e-5


@pytest.mark.parametrize(
    "M,N,K",
    [
        (256, 512, 128),
        (129, 257, 65),  # 非对齐
        (32, 32, 32),
        (1024, 128, 256),
    ],
)
def test_mm(M, N, K):
    a = torch.randn(M, K, device=DEV, dtype=torch.float32)
    b = torch.randn(K, N, device=DEV, dtype=torch.float32)
    out = mm(a, b)
    ref = a @ b
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "M,N,K",
    [
        (256, 512, 128),
        (131, 241, 71),
        (32, 32, 32),
    ],
)
def test_addmm(M, N, K):
    a = torch.randn(M, K, device=DEV, dtype=torch.float32)
    b = torch.randn(K, N, device=DEV, dtype=torch.float32)
    bias = torch.randn(N, device=DEV, dtype=torch.float32)
    out = addmm(bias, a, b)
    ref = torch.addmm(bias, a, b)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "B,M,N,K",
    [
        (8, 128, 256, 64),
        (3, 127, 251, 61),
        (1, 32, 32, 32),
    ],
)
def test_bmm(B, M, N, K):
    a = torch.randn(B, M, K, device=DEV, dtype=torch.float32)
    b = torch.randn(B, K, N, device=DEV, dtype=torch.float32)
    out = bmm(a, b)
    ref = torch.bmm(a, b)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)
