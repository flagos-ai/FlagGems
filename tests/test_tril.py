import torch
import pytest
from src.flag_gems.ops.tril import tril

def test_tril():
    x = torch.randn(128, 128, device="cuda")
    assert torch.allclose(tril(x, 0), torch.tril(x, 0))
    assert torch.allclose(tril(x, 2), torch.tril(x, 2))
    assert torch.allclose(tril(x, -2), torch.tril(x, -2))