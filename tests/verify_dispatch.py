import sys
sys.path.insert(0, "/workspace/FlagGems")

import torch
import torch.profiler as profiler

device = 'cuda'
M = 1048576
torch.manual_seed(42)
x = torch.randn(M, device=device)

print("=" * 60)
print("1. Profile 原生 PyTorch: torch.argmax(x)")
print("=" * 60)
with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU],
    record_shapes=True,
    with_stack=True
) as p:
    torch.argmax(x)

print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

print("\n" + "=" * 60)
print("2. Profile FlagGems Triton: gems.argmax(x, dim=None)")
print("=" * 60)
from flag_gems import argmax as gems_argmax
with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU],
    record_shapes=True,
    with_stack=True
) as p:
    gems_argmax(x, dim=None)

print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

