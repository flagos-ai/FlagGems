import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("leaky_relu_f"),
    key=["n_elements"],
)
@triton.jit
def leaky_relu_fwd_kernel(
    x_ptr,
    y_ptr,
    negative_slope,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x > 0, x, x * negative_slope)
    tl.store(y_ptr + offsets, y, mask=mask)


def leaky_relu(
    x: torch.Tensor, negative_slope: float = 0.01, inplace: bool = False
) -> torch.Tensor:
    if not x.is_contiguous():
        if inplace:
            raise ValueError("Inplace operation requires contiguous tensor.")
        x = x.contiguous()
    n_elements = x.numel()
    if inplace:
        y = x
    else:
        y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    leaky_relu_fwd_kernel[grid](x, y, negative_slope, n_elements)
    return y


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("leaky_relu_f"),
    key=["n_elements"],
)
@triton.jit
def leaky_relu_bwd_kernel(
    grad_out_ptr,
    x_ptr,
    grad_in_ptr,
    negative_slope,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
    grad_in = tl.where(x > 0, grad_out, grad_out * negative_slope)
    tl.store(grad_in_ptr + offsets, grad_in, mask=mask)


def leaky_relu_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    negative_slope: float = 0.01,
    self_is_result: bool = False,
) -> torch.Tensor:
    grad_output = grad_output.contiguous()
    grad_input = torch.empty_like(self)
    n_elements = self.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    leaky_relu_bwd_kernel[grid](
        grad_output, self, grad_input, negative_slope, n_elements
    )
    return grad_input
