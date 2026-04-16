import os
import sys
import time

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.__rshift__ import rshift_tensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

ALL_INT_DTYPES = [torch.int16, torch.int32, torch.int64]

TEST_SHAPES = [
    (1024,),
    (16, 1024),
    (16, 512, 256),
    (4, 128, 1024),
]


@pytest.mark.rshift_performance
@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES)
def test_rshift_performance(shape, dtype):
    inp_a = torch.randint(0, 100, shape, dtype=dtype, device=flag_gems.device)
    inp_b = torch.randint(0, 8, shape, dtype=dtype, device=flag_gems.device)

    # Warmup
    for _ in range(10):
        _ = rshift_tensor(inp_a, inp_b)
    torch.cuda.synchronize()

    # Benchmark FlagGems
    start_time = time.time()
    for _ in range(100):
        _ = rshift_tensor(inp_a, inp_b)
    torch.cuda.synchronize()
    end_time = time.time()
    gems_time = (end_time - start_time) / 100

    # Benchmark PyTorch
    start_time = time.time()
    for _ in range(100):
        _ = inp_a >> inp_b
    torch.cuda.synchronize()
    end_time = time.time()
    torch_time = (end_time - start_time) / 100

    speedup = torch_time / gems_time
    print(f"__rshift__ {shape} {dtype}:")
    print(f"  FlagGems: {gems_time * 1000:.3f}ms")
    print(f"  PyTorch: {torch_time * 1000:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
