import torch
import triton
import triton.language as tl


@triton.jit
def cosh_(x_ptr,  # *Pointer* to input vector (modified in-place).
          n_elements,  # Size of the vector.
          BLOCK_SIZE: tl.constexpr  # Number of elements each program should process.
          ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x32 = tl.cast(x, tl.float32)
    e_pos = tl.exp(x32)
    e_neg = tl.exp(-x32)
    y32 = 0.5 * (e_pos + e_neg)
    tl.store(x_ptr + offsets, y32, mask=mask)


_triton_cosh_kernel = cosh_


def cosh_(*args, **kwargs):
    x = args[0] if len(args) > 0 else kwargs.get('input', None) or kwargs.get('x', None) or kwargs.get('self', None)
    if x is None:
        raise ValueError("cosh_ expects a tensor as the first positional argument.")
    if not isinstance(x, torch.Tensor):
        raise TypeError("cosh_ expects a torch.Tensor input.")
    if not x.is_cuda:
        raise ValueError("cosh_ Triton kernel requires a CUDA tensor.")
    if not x.is_contiguous():
        raise ValueError("cosh_ Triton kernel currently supports contiguous tensors only.")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("cosh_ Triton kernel supports float16, bfloat16, and float32 tensors.")

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _triton_cosh_kernel[grid](x, n_elements, BLOCK_SIZE=1024)
    return x