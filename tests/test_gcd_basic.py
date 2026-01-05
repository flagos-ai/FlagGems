import pytest
import torch
import flag_gems

pytestmark = pytest.mark.gcd


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(1,), (8,), (257,), (64, 64), (256, 256)])
def test_gcd_basic(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    a = torch.randint(-1000, 1000, shape, device="cuda", dtype=dtype)
    b = torch.randint(-1000, 1000, shape, device="cuda", dtype=dtype)

    ref = torch.gcd(a, b)
    with flag_gems.use_gems():
        out = torch.gcd(a, b)

    assert torch.equal(out, ref)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_gcd_zeros(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    a = torch.tensor([0, 0, 5, -5, 12, -12], device="cuda", dtype=dtype)
    b = torch.tensor([0, 7, 0, 10, -18, 18], device="cuda", dtype=dtype)

    ref = torch.gcd(a, b)
    with flag_gems.use_gems():
        out = torch.gcd(a, b)

    assert torch.equal(out, ref)
