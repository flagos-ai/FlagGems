import torch
from addmm import addmm
from bmm import bmm
from mm import mm
from swiglu import dswiglu, swiglu


def test_mm():
    test_shapes = [
        (64, 64),
        (65, 65),
        (65, 32),
        (32, 65),
        (1, 1),
        (1, 8),
        (7, 7),
        (128, 128),
    ]
    print("==== Test MM ====")
    for M, N in test_shapes:
        K = 32
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)
        ref = torch.matmul(a, b)
        ours = mm(a, b)
        diff = torch.max(torch.abs(ref - ours))
        print(f"mm {M}x{K} @ {K}x{N}, max diff: {diff.item():.6f}")
        assert diff < 1e-4


def test_addmm():
    test_shapes = [
        (64, 64, 32),
        (65, 65, 32),
        (65, 32, 16),
        (32, 65, 8),
        (1, 1, 1),
        (7, 7, 7),
    ]
    print("\n==== Test AddMM ====")
    for M, N, K in test_shapes:
        bias = torch.randn(M, N, device="cuda", dtype=torch.float32)
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)
        ref = torch.addmm(bias, a, b, alpha=1.0, beta=1.0)
        ours = addmm(bias, a, b, alpha=1.0, beta=1.0)
        diff = torch.max(torch.abs(ref - ours))
        print(f"addmm bias({M},{N}) + {M}x{K}@{K}x{N}, max diff: {diff.item():.6f}")
        assert diff < 1e-4


def test_addmm_compat():
    print("\n==== Test AddMM Compatibility Case ====")
    bias = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    m1 = torch.randn(64, 32, device="cuda", dtype=torch.float32)
    m2 = torch.randn(32, 128, device="cuda", dtype=torch.float32)
    ref = torch.addmm(bias, m1, m2, alpha=1.0, beta=1.0)
    ours = addmm(bias, m1, m2, alpha=1.0, beta=1.0)
    diff = torch.max(torch.abs(ref - ours))
    print(f"compat case max diff: {diff.item():.6f}")
    assert diff < 1e-4


def test_bmm():
    test_shapes = [
        (2, 64, 64, 32),
        (2, 65, 65, 32),
        (1, 65, 32, 16),
        (4, 7, 7, 7),
        (1, 1, 1, 1),
    ]
    print("\n==== Test BMM ====")
    for B, M, N, K in test_shapes:
        a = torch.randn(B, M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(B, K, N, device="cuda", dtype=torch.float32)
        ref = torch.bmm(a, b)
        ours = bmm(a, b)
        diff = torch.max(torch.abs(ref - ours))
        print(f"bmm B={B}, {M}x{K}@{K}x{N}, max diff: {diff.item():.6f}")
        assert diff < 1e-4


def test_swiglu_fwd():
    print("\n==== Test SwiGLU Forward ====")
    x = torch.randn(2, 8, device="cuda", dtype=torch.float32)
    out = swiglu(x)
    assert out.shape == (2, 4)
    a, b = x.chunk(2, dim=-1)
    ref = a * torch.sigmoid(b)
    diff = torch.max(torch.abs(ref - out))
    print(f"swiglu fwd max diff: {diff.item():.6f}")
    assert diff < 1e-4


def test_swiglu_bwd():
    print("\n==== Test SwiGLU Backward ====")
    x = torch.randn(2, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    a, b = x.chunk(2, dim=-1)
    y = a * torch.sigmoid(b)
    dout = torch.randn_like(y)
    y.backward(dout)
    ref_dx = x.grad.clone()

    our_dx = dswiglu(dout, x)
    diff = torch.max(torch.abs(ref_dx - our_dx))
    print(f"swiglu bwd max diff: {diff.item():.6f}")
    assert diff < 1e-4


if __name__ == "__main__":
    torch.manual_seed(42)
    test_mm()
    test_addmm()
    test_addmm_compat()
    test_bmm()
    test_swiglu_fwd()
    test_swiglu_bwd()
    print("\n✅ All tests passed!")
