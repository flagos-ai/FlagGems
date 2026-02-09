import torch
import pytest
from src.flag_gems.ops.roll import roll

@pytest.mark.parametrize("shape", [(1024, 1024), (100, 100, 100)])
@pytest.mark.parametrize("shift", [1, 50, -10])
@pytest.mark.parametrize("dim", [0, 1])
def test_roll_correctness(shape, shift, dim):
    x = torch.randn(shape, device="cuda")
    out_pt = torch.roll(x, shift, dim)
    out_tri = roll(x, shift, dim)
    assert torch.allclose(out_tri, out_pt)

def test_roll_benchmark():
    x = torch.randn((2048, 2048), device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100): torch.roll(x, 100, 1)
    end.record()
    torch.cuda.synchronize()
    ms_pt = start.elapsed_time(end)
    start.record()
    for _ in range(100): roll(x, 100, 1)
    end.record()
    torch.cuda.synchronize()
    print(f"Roll Speedup: {ms_pt/start.elapsed_time(end):.2f}x")