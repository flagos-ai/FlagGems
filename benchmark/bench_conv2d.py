# benchmark/bench_conv2d.py
import time

import torch

import flag_gems


def bench_conv2d(N, C, H, W, warmup=10, repeat=50):
    x = torch.randn((N, C, H, W), dtype=torch.float16, device="cuda").contiguous()
    w = torch.randn((C, C, 3, 3), dtype=torch.float16, device="cuda").contiguous()
    # warmup
    for _ in range(warmup):
        _ = flag_gems.conv2d(x, w)
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        _ = flag_gems.conv2d(x, w)
        torch.cuda.synchronize()
    t_gems = (time.time() - t0) / repeat

    for _ in range(warmup):
        _ = torch.nn.functional.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        _ = torch.nn.functional.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
    t_ref = (time.time() - t0) / repeat
    speedup = t_ref / t_gems
    print(
        f"Conv3x3 N={N},C={C},H={H},W={W} | FlagGems: {t_gems*1e6:.2f} us | ATen: {t_ref*1e6:.2f} us | speedup={speedup:.3f}"
    )
    return t_gems, t_ref, speedup


if __name__ == "__main__":
    bench_conv2d(8, 128, 64, 64)
