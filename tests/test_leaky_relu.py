import torch
from flag_gems.ops.leaky_relu import leaky_relu


def test_leaky_relu_float32():
    x = torch.randn(1024, device="cuda", dtype=torch.float32)

    out = leaky_relu(x)

    ref = torch.nn.functional.leaky_relu(x)

    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-6)


def test_leaky_relu_float16():
    x = torch.randn(1024, device="cuda", dtype=torch.float16)

    out = leaky_relu(x)

    ref = torch.nn.functional.leaky_relu(x)

    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-3)
