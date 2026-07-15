import pytest
import torch
import time

import flag_gems


def benchmark(func, inp, dim, warmup=20, repeat=100):

    for _ in range(warmup):
        func(inp, dim)

    torch.npu.synchronize()

    start = time.time()

    for _ in range(repeat):
        func(inp, dim)

    torch.npu.synchronize()

    end = time.time()

    return (end - start) * 1000 / repeat


@pytest.mark.unsqueeze_copy
def test_unsqueeze_copy():

    shapes = [
        (1024,),
        (1024, 1024),
        (4096, 4096),
        (32, 1024, 1024),
    ]

    dtype = torch.float16

    print("\nUnsqueeze_copy Performance (Ascend)")

    for shape in shapes:

        inp = torch.randn(
            shape,
            dtype=dtype,
            device=flag_gems.device,
        )

        aten_time = benchmark(
            torch.ops.aten.unsqueeze_copy.default,
            inp,
            -1,
        )

        gems_time = benchmark(
            flag_gems.unsqueeze_copy,
            inp,
            -1,
        )

        speedup = aten_time / gems_time

        print("=" * 60)
        print(f"Shape: {shape}")
        print(f"Aten: {aten_time:.6f} ms")
        print(f"Gems: {gems_time:.6f} ms")
        print(f"Speedup: {speedup:.3f}")
