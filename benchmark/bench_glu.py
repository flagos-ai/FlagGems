# benchmark/bench_glu.py
import time

import torch

import flag_gems


def bench_glu(B, D, warmup=10, repeat=100):
    x = torch.randn((B, D), dtype=torch.float16, device="cuda").contiguous()
    for _ in range(warmup):
        _ = flag_gems.glu(x)
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        _ = flag_gems.glu(x)
        torch.cuda.synchronize()
    t_gems = (time.time() - t0) / repeat

    for _ in range(warmup):
        _ = x[..., : D // 2] * torch.sigmoid(x[..., D // 2 :])
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        _ = x[..., : D // 2] * torch.sigmoid(x[..., D // 2 :])
        torch.cuda.synchronize()
    t_ref = (time.time() - t0) / repeat
    speedup = t_ref / t_gems
    print(
        f"GLU B={B},D={D} | FlagGems: {t_gems*1e6:.2f} us | ATen: {t_ref*1e6:.2f} us | speedup={speedup:.3f}"
    )
    return t_gems, t_ref, speedup


if __name__ == "__main__":
    bench_glu(2048, 2048)
