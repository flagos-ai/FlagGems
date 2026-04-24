import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(
    promotion_methods=[
        ((0, 1), "DEFAULT"),
        ((0, 1), "DEFAULT"),
    ],
    num_outputs=2,
)
@triton.jit
def polar_kernel(abs, angle):
    real = abs * tl.cos(angle)
    imag = abs * tl.sin(angle)
    return real, imag


def polar(abs, angle):
    # Compute real and imaginary parts separately on TPU
    real = torch.empty_like(abs)
    imag = torch.empty_like(abs)

    polar_kernel(abs, angle, out0=real, out1=imag)

    # TPU does not support complex type, return stacked tensor of real and imaginary parts
    # Shape: (..., 2), where last dimension [0] is real part, [1] is imaginary part
    return torch.stack([real, imag], dim=-1)
