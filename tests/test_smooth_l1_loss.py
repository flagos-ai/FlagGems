import torch
import torch.nn.functional as F
import pytest
from src.flag_gems.ops.smooth_l1_loss import smooth_l1_loss

@pytest.mark.parametrize("shape", [(1024, 1024), (100, 100), (1, 10)])
@pytest.mark.parametrize("reduction", ['mean', 'sum', 'none'])
def test_smooth_l1_correctness(shape, reduction):
    input = torch.randn(shape, device='cuda', dtype=torch.float32)
    target = torch.randn(shape, device='cuda', dtype=torch.float32)
    
    # PyTorch
    out_pt = F.smooth_l1_loss(input, target, reduction=reduction, beta=1.0)
    
    # Triton
    out_triton = smooth_l1_loss(input, target, beta=1.0, reduction=reduction)
    
    # Check
    assert torch.allclose(out_triton, out_pt, atol=1e-4)

@pytest.mark.parametrize("shape", [(4096, 4096)]) # Large tensor
def test_smooth_l1_benchmark(shape):
    input = torch.randn(shape, device='cuda', dtype=torch.float32)
    target = torch.randn(shape, device='cuda', dtype=torch.float32)
    
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    # PT
    start.record()
    for _ in range(100): F.smooth_l1_loss(input, target, reduction='mean')
    end.record()
    torch.cuda.synchronize()
    pt_ms = start.elapsed_time(end) / 100
    
    # Triton
    start.record()
    for _ in range(100): smooth_l1_loss(input, target, reduction='mean')
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / 100
    
    print(f"Shape {shape}: PT {pt_ms:.3f}ms, Triton {triton_ms:.3f}ms, Speedup {pt_ms/triton_ms:.2f}x")