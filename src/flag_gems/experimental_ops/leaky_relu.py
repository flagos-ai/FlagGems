import torch
import triton
import triton.language as tl


@triton.jit
def _leaky_relu_kernel(
    x_ptr, y_ptr, n_elements, negative_slope, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    zero = tl.zeros([BLOCK_SIZE], dtype=x.dtype)
    slope = tl.full([BLOCK_SIZE], negative_slope, dtype=x.dtype)
    # y = x if x >= 0 else slope * x
    # Equivalent, branchless:
    y = tl.maximum(x, zero) + slope * tl.minimum(x, zero)

    tl.store(y_ptr + offsets, y, mask=mask)


def _launch_leaky_relu_kernel(
    x: torch.Tensor, out: torch.Tensor, negative_slope: float
):
    if not x.is_cuda or not out.is_cuda:
        raise ValueError("Input and output tensors must be on CUDA device.")
    if x.numel() != out.numel():
        raise ValueError("Input and output must have the same number of elements.")
    if x.dtype != out.dtype:
        raise ValueError("Input and output tensors must have the same dtype.")
    if not x.is_contiguous():
        x = x.contiguous()
    if not out.is_contiguous():
        raise ValueError("Output tensor must be contiguous.")

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _leaky_relu_kernel[grid](x, out, n_elements, float(negative_slope), BLOCK_SIZE=1024)
    return out


def leaky_relu(input: torch.Tensor, negative_slope: float = 0.01):
    """
    ATen: ('leaky_relu', <Autograd.disable: False>)
    """
    out = torch.empty_like(input)
    return _launch_leaky_relu_kernel(input, out, negative_slope)


def leaky_relu_out(
    input: torch.Tensor, negative_slope: float = 0.01, out: torch.Tensor = None
):
    """
    ATen: ('leaky_relu.out', <Autograd.disable: False>)
    """
    if out is None:
        raise ValueError("Argument 'out' must be provided for leaky_relu_out.")
    return _launch_leaky_relu_kernel(input, out, negative_slope)
