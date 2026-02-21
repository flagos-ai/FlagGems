import torch

def test_log10_basic():
    x = torch.tensor([1.0, 10.0, 100.0])
    y = torch.log10(x)
    assert torch.allclose(y, torch.tensor([0.0, 1.0, 2.0]))

