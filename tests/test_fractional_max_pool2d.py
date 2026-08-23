import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Custom shapes for fractional_max_pool2d: shape, kernel_size, output_size.
# 4D shapes are batched (N, C, H, W); 3D shapes are unbatched (C, H, W),
# both of which torch.nn.functional.fractional_max_pool2d supports.
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
    # Unbatched (C, H, W) inputs
    ((3, 32, 32), 2, (16, 16)),
    ((8, 16, 20), 2, (8, 10)),
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
    # random_samples is (N, C, 2) for batched input and (C, 2) for unbatched.
    sample_shape = (*shape[:-2], 2)
    random_samples = torch.rand(
        sample_shape, dtype=torch.float32, device=flag_gems.device
    )
    ref_random_samples = random_samples.to(device=ref_inp.device, dtype=ref_inp.dtype)
    # PyTorch's reference requires _random_samples to be 3D even for 3D input,
    # so add a leading batch dim for the unbatched case.
    if len(shape) == 3:
        ref_random_samples = ref_random_samples.unsqueeze(0)

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
    # random_samples is (N, C, 2) for batched input and (C, 2) for unbatched.
    sample_shape = (*shape[:-2], 2)
    random_samples = torch.rand(
        sample_shape, dtype=torch.float32, device=flag_gems.device
    )
    ref_random_samples = random_samples.to(device=ref_inp.device, dtype=ref_inp.dtype)
    # PyTorch's reference requires _random_samples to be 3D even for 3D input,
    # so add a leading batch dim for the unbatched case.
    if len(shape) == 3:
        ref_random_samples = ref_random_samples.unsqueeze(0)

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
