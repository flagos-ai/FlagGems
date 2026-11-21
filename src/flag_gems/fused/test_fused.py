import torch

from flag_gems.fused.addmm import addmm
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
    ]
    print("\n==== Test AddMM ====")
    for M, K, N in test_cases:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)
        bias = torch.randn(N, device="cuda", dtype=torch.float32)
        ref = torch.addmm(bias, a, b, beta=1.0, alpha=1.0)
        ours = addmm(bias, a, b, alpha=1.0, beta=1.0)
        diff = torch.max(torch.abs(ref - ours))
        print(
            f"addmm bias({N},) + {M}x{K} @ {K}x{N}, alpha=1 beta=1, max diff: {diff:.6f}"
        )

    print("\n==== Test AddMM alpha=0.5 beta=2.0 ====")
    M, K, N = 48, 16, 32
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.float32)
    ref = torch.addmm(bias, a, b, beta=2.0, alpha=0.5)
    ours = addmm(bias, a, b, alpha=0.5, beta=2.0)
    diff = torch.max(torch.abs(ref - ours))
    print(
        f"addmm bias({N},) + {M}x{K} @ {K}x{N}, alpha=0.5 beta=2.0, max diff: {diff:.6f}"
    )


if __name__ == "__main__":
    test_mm()
    test_addmm()
