import time
import torch
from flag_gems import conv_depthwise2d


def bench_conv2d():
    device = "cuda"
    dtype = torch.float16
    N, C, H, W = 2, 512, 64, 64
    K = 3
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    x = torch.randn((N, C, H, W), device=device, dtype=dtype)
    w = torch.randn((C, 1, K, K), device=device, dtype=dtype)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        conv_depthwise2d(x, w, stride, padding, dilation, groups=C)
    torch.cuda.synchronize()
    print("time:", time.time() - start)


if __name__ == "__main__":
    bench_conv2d()
