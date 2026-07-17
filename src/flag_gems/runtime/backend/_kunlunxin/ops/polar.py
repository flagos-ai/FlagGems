import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic


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
    output = torch.empty((*abs.shape, 2), dtype=abs.dtype, device=abs.device)

    polar_kernel(abs, angle, out0=output[..., 0], out1=output[..., 1])

    return torch.view_as_complex(output)
