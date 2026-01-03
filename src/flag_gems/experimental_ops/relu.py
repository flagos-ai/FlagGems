import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.maximum(x, 0)
    tl.store(out_ptr + offsets, y, mask=mask)


def _launch_relu_kernel(x: torch.Tensor, out: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Tensors must be on CUDA device"
    assert x.device == out.device, "Input and output must be on the same device"
    assert x.dtype == out.dtype, "Input and output must have the same dtype"
    assert (
        x.numel() == out.numel()
    ), "Input and output must have the same number of elements"

    n_elements = x.numel()
    if n_elements == 0:
        return

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)


def relu(*args, **kwargs):
    # Accept common calling patterns: relu(input)
    x = None
    if len(args) > 0:
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", kwargs.get("x")))
    if x is None:
        raise TypeError("relu expected a Tensor argument 'input'")

    # Ensure tensor type and device
    if not x.is_cuda:
        raise AssertionError("Input tensor must be on CUDA device")

    out = torch.empty_like(x, memory_format=torch.contiguous_format)

    if x.is_contiguous():
        _launch_relu_kernel(x, out)
    else:
        x_contig = x.contiguous()
        _launch_relu_kernel(x_contig, out)

    return out


def relu_out(*args, **kwargs):
    # Accept common calling patterns:
    # relu_out(input, out) or relu_out(input, out=...)
    x = None
    out = None

    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", kwargs.get("x")))
    if len(args) >= 2:
        out = args[1]
    else:
        out = kwargs.get("out")

    if x is None or out is None:
        raise TypeError("relu_out expected arguments: input Tensor and out Tensor")

    if not x.is_cuda or not out.is_cuda:
        raise AssertionError("Input and output tensors must be on CUDA device")

    if x.shape != out.shape:
        raise AssertionError("Input and output must have the same shape")
    if x.dtype != out.dtype:
        raise AssertionError("Input and output must have the same dtype")

    # Compute into a contiguous destination, then copy if needed
    if out.is_contiguous():
        x_src = x.contiguous() if not x.is_contiguous() else x
        _launch_relu_kernel(x_src, out)
    else:
        temp = torch.empty_like(out, memory_format=torch.contiguous_format)
        x_src = x.contiguous() if not x.is_contiguous() else x
        _launch_relu_kernel(x_src, temp)
        out.copy_(temp)

    return out
