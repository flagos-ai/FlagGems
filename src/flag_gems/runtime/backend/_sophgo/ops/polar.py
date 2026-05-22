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
    real = torch.empty_like(abs)
    imag = torch.empty_like(abs)
    polar_kernel(abs, angle, out0=real, out1=imag)

    rcpu, icpu = real.cpu(), imag.cpu()
    stacked = torch.stack([rcpu, icpu], dim=-1)
    return torch.view_as_complex(stacked)
