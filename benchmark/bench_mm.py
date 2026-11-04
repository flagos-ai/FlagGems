import time

import torch

import flag_gems


def bench_mm(M, N, K, warmup=10, repeat=100):
    a = torch.randn((M, K), dtype=torch.float16, device="cuda").contiguous()
    b = torch.randn((K, N), dtype=torch.float16, device="cuda").contiguous()
    # warmup
    for _ in range(warmup):
        _ = flag_gems.mm(a, b)
        torch.cuda.synchronize()
    # flaggems
    t0 = time.time()
    for _ in range(repeat):
        _ = flag_gems.mm(a, b)
        torch.cuda.synchronize()
    t_gems = (time.time() - t0) / repeat
    # aten ref
    for _ in range(warmup):
        _ = a @ b
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        _ = a @ b
        torch.cuda.synchronize()
    t_ref = (time.time() - t0) / repeat
    speedup = t_ref / t_gems
    print(
        f"MM M={M},N={N},K={K} | FlagGems: {t_gems*1e6:.2f} us | ATen: {t_ref*1e6:.2f} us | speedup={speedup:.3f}"
    )
    return speedup


if __name__ == "__main__":
    for s in [128, 256, 512, 1024]:
        bench_mm(s, s, s)
