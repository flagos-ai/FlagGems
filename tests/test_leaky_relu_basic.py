import pytest
import torch
import flag_gems

pytestmark = pytest.mark.leaky_relu


def _tol(dtype):
    if dtype == torch.float16:
        return 1e-4, 1e-3
    if dtype == torch.bfloat16:
        return 1e-4, 1.6e-2
    if dtype == torch.float32:
        return 1e-4, 1.3e-6
    raise RuntimeError(f"unsupported dtype: {dtype}")


@pytest.mark.parametrize("negative_slope", [0.01, 0.0, 0.2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1,), (8,), (257,), (64, 64), (256, 256)])
def test_leaky_relu_basic(negative_slope, dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x = torch.randn(*shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    with flag_gems.use_gems():
        out = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)

    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("negative_slope", [0.01, 0.2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_leaky_relu_special_values(negative_slope, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x = torch.tensor(
        [0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan")],
        device="cuda",
        dtype=dtype,
    )

    ref = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    with flag_gems.use_gems():
        out = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)

    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("negative_slope", [0.01, 0.2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_leaky_relu_empty(negative_slope, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x = torch.empty((0,), device="cuda", dtype=dtype)

    ref = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    with flag_gems.use_gems():
        out = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)

    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol, equal_nan=True)
