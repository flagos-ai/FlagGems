import torch

from flag_gems.fused.addmm import addmm
from flag_gems.fused.glu import glu
from flag_gems.fused.mm import mm


def test_mm():
    test_cases = [
        (64, 32, 64),
        (65, 32, 65),
        (65, 32, 32),
        (32, 32, 65),
        (1, 32, 1),
        (1, 32, 8),
        (7, 32, 7),
        (128, 32, 128),
        (57, 41, 73),
        (13, 17, 29),
    ]
    print("\n==== Test MM ====")
    for M, K, N in test_cases:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)
        ref = torch.matmul(a, b)
        ours = mm(a, b)
        diff = torch.max(torch.abs(ref - ours))
        print(f"mm {M}x{K} @ {K}x{N}, max diff: {diff:.6f}")


def test_addmm():
    test_cases = [
        (64, 32, 64),
        (65, 32, 65),
        (65, 32, 32),
        (32, 32, 65),
        (1, 32, 1),
        (1, 32, 8),
        (7, 32, 7),
        (128, 32, 128),
        (57, 41, 73),
        (13, 17, 29),
    ]
    print("\n==== Test AddMM ====")
    for M, K, N in test_cases:
        mat1 = torch.randn(M, K, device="cuda", dtype=torch.float32)
        mat2 = torch.randn(K, N, device="cuda", dtype=torch.float32)
        bias = torch.randn(N, device="cuda", dtype=torch.float32)
        ref = torch.addmm(bias, mat1, mat2, beta=1.0, alpha=1.0)
        ours = addmm(bias, mat1, mat2, beta=1.0, alpha=1.0)
        diff = torch.max(torch.abs(ref - ours))
        print(f"addmm bias({N},) + {M}x{K} @ {K}x{N}, beta=1 alpha=1, max diff: {diff:.6f}")

    print("\n==== Test AddMM alpha=0.5 beta=2.0 ====")
    M, K, N = 48, 16, 32
    mat1 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    mat2 = torch.randn(K, N, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.float32)
    ref = torch.addmm(bias, mat1, mat2, beta=2.0, alpha=0.5)
    ours = addmm(bias, mat1, mat2, beta=2.0, alpha=0.5)
    diff = torch.max(torch.abs(ref - ours))
    print(f"addmm bias({N},) + {M}x{K} @ {K}x{N}, beta=2 alpha=0.5, max diff: {diff:.6f}")


def test_glu():
    print("\n==== Test GLU ====")
    test_cases = [(2, 8), (4, 16), (5, 128), (7, 2)]
    for shape in test_cases:
        x = torch.randn(*shape, device="cuda", dtype=torch.float32)
        d = x.shape[-1] // 2
        ref = x[..., :d] * torch.sigmoid(x[..., d:])
        ours = glu(x)
        diff = torch.max(torch.abs(ref - ours))
        print(f"glu input {shape}, output {(*shape[:-1], d)}, max diff: {diff:.6f}")


if __name__ == "__main__":
    test_mm()
    test_addmm()
    test_glu()
