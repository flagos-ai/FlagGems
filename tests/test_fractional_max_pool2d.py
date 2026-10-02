import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

from . import accuracy_utils as utils

# Custom shapes for fractional_max_pool2d: (batch, channels, height, width), kernel_size, output_size
FRACTIONAL_MAXPOOL2D_CONFIGS = [
    # Classic case: 2x2 kernel, output size 16x16
    ((4, 3, 32, 32), 2, (16, 16)),
    # Different output sizes
    ((8, 16, 28, 28), 2, (14, 14)),
    # Test different output sizes
    ((2, 4, 20, 20), 2, (10, 10)),
    # Larger case
    ((1, 64, 56, 56), 2, (28, 28)),
    # No padding, different kernel sizes
    ((2, 8, 16, 16), 2, (8, 8)),
    # Non-square output
    ((2, 8, 16, 20), 2, (8, 10)),
]


@pytest.mark.fractional_max_pool2d
@pytest.mark.parametrize(
    "shape, kernel_size, output_size", FRACTIONAL_MAXPOOL2D_CONFIGS
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fractional_max_pool2d(shape, kernel_size, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)

    # Use the same random_samples for both to ensure identical pooling regions.
    # PyTorch C++ requires _random_samples dtype == input dtype.
    # Our implementation converts to double internally, so we pass float32 to ours.
    # Both paths ultimately compute intervals with the same float64 values.
    N, C = shape[0], shape[1]
    random_samples = torch.rand(N, C, 2, dtype=torch.float32, device=flag_gems.device)
    ref_random_samples = random_samples.to(device=ref_inp.device, dtype=ref_inp.dtype)

    ref_out, ref_indices = torch.nn.functional.fractional_max_pool2d(
        ref_inp,
        kernel_size=kernel_size,
        output_size=output_size,
        return_indices=True,
        _random_samples=ref_random_samples,
    )

    res_out, res_indices = flag_gems.fractional_max_pool2d(
        inp,
        kernel_size=kernel_size,
        output_size=output_size,
        _random_samples=random_samples,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    # Indices should match exactly since we use the same random_samples
    torch.testing.assert_close(res_indices.to(ref_indices.device), ref_indices)


@pytest.mark.fractional_max_pool2d_backward
@pytest.mark.parametrize(
    "shape, kernel_size, output_size", FRACTIONAL_MAXPOOL2D_CONFIGS
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fractional_max_pool2d_backward(shape, kernel_size, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, upcast=True)

    # Use the same random_samples for both.
    N, C = shape[0], shape[1]
    random_samples = torch.rand(N, C, 2, dtype=torch.float32, device=flag_gems.device)
    ref_random_samples = random_samples.to(device=ref_inp.device, dtype=ref_inp.dtype)

    ref_out, ref_indices = torch.nn.functional.fractional_max_pool2d(
        ref_inp,
        kernel_size=kernel_size,
        output_size=output_size,
        return_indices=True,
        _random_samples=ref_random_samples,
    )
    res_out, res_indices = flag_gems.fractional_max_pool2d(
        inp,
        kernel_size=kernel_size,
        output_size=output_size,
        _random_samples=random_samples,
    )

    out_grad = torch.randn_like(res_out, device=flag_gems.device)
    ref_grad = utils.to_reference(out_grad, upcast=True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    res_in_grad = flag_gems.fractional_max_pool2d_backward(
        out_grad,
        inp,
        kernel_size=kernel_size,
        output_size=output_size,
        indices=res_indices,
    )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.fractional_max_pool2d_backward
@pytest.mark.parametrize(
    "grad_shape, indices_shape, message",
    [
        (
            (2, 4, 4, 4),
            (2, 4, 16, 16),
            r"fractional_max_pool2d_backward\(\): gradOutput sizes unexpected",
        ),
        (
            (2, 4, 16, 16),
            (2, 4, 4, 4),
            r"fractional_max_pool2d_backward\(\): indices sizes unexpected",
        ),
    ],
)
def test_fractional_max_pool2d_backward_rejects_unexpected_sizes(
    grad_shape, indices_shape, message
):
    inp = torch.randn((2, 4, 16, 16), device=flag_gems.device)
    grad_output = torch.randn(grad_shape, device=flag_gems.device)
    indices = torch.zeros(indices_shape, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(RuntimeError, match=message):
        flag_gems.fractional_max_pool2d_backward(
            grad_output,
            inp,
            kernel_size=(3, 3),
            output_size=(16, 16),
            indices=indices,
        )

    health = torch.ones(4, device=flag_gems.device)
    health.add_(1)
    torch_device_fn.synchronize()
    assert health.cpu().tolist() == [2.0] * 4
