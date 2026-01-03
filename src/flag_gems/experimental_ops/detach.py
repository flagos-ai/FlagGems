import torch
import triton
import triton.language as tl


@triton.jit
def detach(x_ptr,  # Pointer to input tensor
           out_ptr,  # Pointer to output tensor
           n_elements,  # Number of elements
           BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


_detach_kernel = detach


def detach(*args, **kwargs):
    # Extract input tensor from positional or keyword arguments
    x = None
    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
        x = args[0]
    elif 'input' in kwargs and isinstance(kwargs['input'], torch.Tensor):
        x = kwargs['input']
    elif 'self' in kwargs and isinstance(kwargs['self'], torch.Tensor):
        x = kwargs['self']
    else:
        raise ValueError("detach expects a Tensor as the first positional argument or as 'input'/'self' keyword.")

    # Ensure tensor is on a CUDA device for Triton
    if not x.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")

    # Create contiguous views for kernel processing
    x_c = x.contiguous()
    out_c = torch.empty_like(x_c, memory_format=torch.contiguous_format)

    n_elements = x_c.numel()
    if n_elements == 0:
        return out_c.view_as(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _detach_kernel[grid](x_c, out_c, n_elements, BLOCK_SIZE=1024)

    return out_c.view_as(x)