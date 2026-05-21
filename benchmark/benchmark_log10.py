
import sys
import os
import time

sys.path.append(os.path.abspath("src"))

import torch

from flag_gems.ops.log10 import log10


x = torch.rand(10_000_000, device="cuda") + 0.01


# ---------------- Triton Benchmark ----------------

torch.cuda.synchronize()

start = time.time()

for _ in range(100):
    y = log10(x)

torch.cuda.synchronize()

triton_time = time.time() - start


# ---------------- PyTorch Benchmark ----------------

torch.cuda.synchronize()

start = time.time()

for _ in range(100):
    y = torch.log10(x)

torch.cuda.synchronize()

torch_time = time.time() - start


print("=" * 50)
print(f"Triton Time : {triton_time:.4f} sec")
print(f"PyTorch Time: {torch_time:.4f} sec")
print(f"Speedup     : {torch_time / triton_time:.2f}x")
print("=" * 50)
