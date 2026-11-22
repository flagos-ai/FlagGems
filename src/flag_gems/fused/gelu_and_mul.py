import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

erf = tl_extra_shim.erf
exp = tl_extra_shim.exp


@triton.jit
def gelu_and_mul_kernel(
    x_ptr,
    y_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + pid * n_cols + offs, mask=mask, other=0.0)
    half = n_cols // 2
    x0 = x[:half]
    x1 = x[half:]
    cdf = 0.5 * (1.0 + erf(x0 / 1.41421356237))
    out = x0 * cdf * x1
    tl.store(y_ptr + pid * n_cols + offs, out, mask=mask)


def gelu_and_mul(x: torch.Tensor) -> torch.Tensor:
    B, N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    gelu_and_mul_kernel[(B,)](x, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out
