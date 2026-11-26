import pytest
import torch

import flag_gems


def check_fft(name, dtype, length, norm, *, backward=True):
    high_precision = dtype in (torch.float64, torch.complex128)
    if high_precision and not flag_gems.runtime.device.support_fp64:
        pytest.skip("Device does not support FP64")

    generator = torch.Generator().manual_seed(123)
    host = torch.randn((2, 3, 9), dtype=dtype, generator=generator).transpose(1, 2)
    device = host.to(flag_gems.device)
    host.requires_grad_(backward)
    device.requires_grad_(backward)
    reference = getattr(torch.fft, name)(host, n=length, dim=-2, norm=norm)
    actual = getattr(flag_gems, f"fft_{name}")(device, n=length, dim=-2, norm=norm)
    tolerance = 2e-12 if high_precision else 2e-5
    torch.testing.assert_close(actual.cpu(), reference, atol=tolerance, rtol=tolerance)
    if backward:
        grad = torch.randn(reference.shape, dtype=reference.dtype, generator=generator)
        device_grad = grad.to(device.device).requires_grad_()
        grad.requires_grad_()
        (expected_grad,) = torch.autograd.grad(reference, host, grad, create_graph=True)
        (actual_grad,) = torch.autograd.grad(
            actual, device, device_grad, create_graph=True
        )
        torch.testing.assert_close(
            actual_grad.cpu(), expected_grad, atol=tolerance, rtol=tolerance
        )
        vector = torch.randn(host.shape, dtype=host.dtype, generator=generator)
        (expected_second,) = torch.autograd.grad(expected_grad, grad, vector)
        (actual_second,) = torch.autograd.grad(
            actual_grad, device_grad, vector.to(device.device)
        )
        torch.testing.assert_close(
            actual_second.cpu(), expected_second, atol=tolerance, rtol=tolerance
        )


@pytest.mark.fft_fft
@pytest.mark.parametrize(
    "dtype,length,norm",
    [
        (torch.float32, 8, None),
        (torch.float32, 256, "ortho"),
        (torch.float32, 513, "ortho"),
        (torch.complex64, 1, "ortho"),
        (torch.complex64, None, "backward"),
        (torch.complex128, 7, "forward"),
    ],
)
def test_fft_fft(dtype, length, norm):
    check_fft("fft", dtype, length, norm, backward=length != 513)


@pytest.mark.fft_irfft
@pytest.mark.parametrize(
    "dtype,length,norm",
    [
        (torch.float32, 8, "forward"),
        (torch.complex64, 1, "ortho"),
        (torch.complex64, None, None),
        (torch.complex64, 240, "forward"),
        (torch.complex128, 7, "backward"),
        (torch.complex128, 32, "ortho"),
    ],
)
def test_fft_irfft(dtype, length, norm):
    check_fft("irfft", dtype, length, norm)
