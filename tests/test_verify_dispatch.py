import torch
import torch.profiler as profiler

device = "cuda"
M = 1048576
torch.manual_seed(42)
x = torch.randn(M, device=device)

def test_torch_argmax_dispatch():
    print("\n" + "=" * 60)
    print("Profile 原生 PyTorch: torch.argmax(x)")
    print("=" * 60)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        record_shapes=True,
    ) as p:
        result = torch.argmax(x)
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
    print(f"result = {result.item()}")

def test_gems_argmax_dispatch():
    from flag_gems import argmax as gems_argmax
    print("\n" + "=" * 60)
    print("Profile FlagGems: gems.argmax(x, dim=None)")
    print("=" * 60)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        record_shapes=True,
    ) as p:
        result = gems_argmax(x, dim=None)
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
    print(f"result = {result.item()}")
