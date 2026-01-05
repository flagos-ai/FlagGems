import torch
import pytest
import flag_gems

pytestmark = pytest.mark.log10

DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
]


@pytest.mark.parametrize("dtype", DTYPES)
def test_log10_basic(dtype):
    x = torch.tensor([0.1, 1.0, 10.0], dtype=dtype, device="cuda")
    ref = torch.log10(x)

    with flag_gems.use_gems():
        y = torch.log10(x)

    assert torch.allclose(y, ref, equal_nan=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_log10_shapes(dtype):
    x = torch.rand((16, 32, 8), dtype=dtype, device="cuda") + 0.01
    ref = torch.log10(x)

    with flag_gems.use_gems():
        y = torch.log10(x)

    assert torch.allclose(y, ref, rtol=1e-3, atol=1e-3, equal_nan=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_log10_special_values(dtype):
    x = torch.tensor(
        [0.1, 1.0, 10.0, 0.0, -1.0, float("inf"), float("nan")],
        dtype=dtype,
        device="cuda",
    )
    ref = torch.log10(x)

    with flag_gems.use_gems():
        y = torch.log10(x)

    assert torch.allclose(y, ref, equal_nan=True)
