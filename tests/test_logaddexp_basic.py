import pytest
import torch
import flag_gems

DTYPES = [torch.float16, torch.float32, torch.bfloat16]


@pytest.mark.logaddexp
@pytest.mark.parametrize("dtype", DTYPES)
def test_logaddexp_basic(dtype):
    a = torch.tensor([0.1, 1.0, 10.0], dtype=dtype, device="cuda")
    b = torch.tensor([0.2, 2.0, 20.0], dtype=dtype, device="cuda")
    ref = torch.logaddexp(a, b)

    with flag_gems.use_gems():
        y = torch.logaddexp(a, b)

    assert torch.allclose(y, ref, rtol=1e-4, atol=1e-3 if dtype == torch.float16 else 1.3e-6, equal_nan=True)


@pytest.mark.logaddexp
@pytest.mark.parametrize("dtype", DTYPES)
def test_logaddexp_shapes(dtype):
    a = torch.rand((16, 32, 8), dtype=dtype, device="cuda") * 10 - 5
    b = torch.rand((16, 32, 8), dtype=dtype, device="cuda") * 10 - 5
    ref = torch.logaddexp(a, b)

    with flag_gems.use_gems():
        y = torch.logaddexp(a, b)

    # tolerancias razonables por dtype
    atol = 1e-3 if dtype == torch.float16 else (0.016 if dtype == torch.bfloat16 else 1.3e-6)
    assert torch.allclose(y, ref, rtol=1e-4, atol=atol, equal_nan=True)


@pytest.mark.logaddexp
@pytest.mark.parametrize("dtype", DTYPES)
def test_logaddexp_special_values(dtype):
    a = torch.tensor([0.0, -1.0, float("inf"), float("nan"), 100.0], dtype=dtype, device="cuda")
    b = torch.tensor([0.0,  2.0, float("inf"), 1.0,         -100.0], dtype=dtype, device="cuda")
    ref = torch.logaddexp(a, b)

    with flag_gems.use_gems():
        y = torch.logaddexp(a, b)

    atol = 1e-3 if dtype == torch.float16 else (0.016 if dtype == torch.bfloat16 else 1.3e-6)
    assert torch.allclose(y, ref, rtol=1e-4, atol=atol, equal_nan=True)
