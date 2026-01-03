import torch
import triton
import triton.language as tl


@triton.jit
def positive(x_ptr,  # *Pointer* to input tensor.
             output_ptr,  # *Pointer* to output tensor.
             n_elements,  # Number of elements.
             BLOCK_SIZE: tl.constexpr,  # Elements processed per program.
             ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)


_positive_kernel = positive


def positive(*args, **kwargs):
    # Extract the tensor argument
    x = None
    if args:
        x = args[0]
    else:
        x = kwargs.get('input', kwargs.get('self', kwargs.get('x', None)))

    if x is None:
        raise ValueError("positive expects a torch.Tensor as the first argument")
    if not isinstance(x, torch.Tensor):
        raise TypeError("positive expects a torch.Tensor")

    # Fallback for non-CUDA or unsupported types
    if (not x.is_cuda) or x.is_complex():
        return torch.ops.aten.positive(x)

    x_contig = x.contiguous()
    n_elements = x_contig.numel()
    out = torch.empty_like(x_contig)

    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _positive_kernel[grid](x_contig, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out