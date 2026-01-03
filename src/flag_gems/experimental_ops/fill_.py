import torch
import triton
import triton.language as tl


@triton.jit
def fill_kernel(out_ptr, value_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    val = tl.load(value_ptr)  # load scalar value (1-element tensor)
    tl.store(out_ptr + offsets, val, mask=mask)


def fill__Scalar(*args, **kwargs):
    # Expecting (self, value)
    if len(args) >= 2:
        self = args[0]
        value = args[1]
    else:
        self = kwargs.get('self') or kwargs.get('input') or kwargs.get('tensor')
        value = kwargs.get('value')
    if self is None:
        raise ValueError("fill__Scalar requires a target tensor 'self'.")
    if value is None:
        raise ValueError("fill__Scalar requires a scalar 'value'.")
    assert self.is_cuda, "fill__Scalar requires a CUDA tensor."
    assert self.is_contiguous(), "fill__Scalar currently supports only contiguous tensors."
    # Create a 1-element tensor on device with matching dtype
    value_t = torch.tensor([value], dtype=self.dtype, device=self.device)
    n_elements = self.numel()
    if n_elements == 0:
        return self
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fill_kernel[grid](self, value_t, n_elements, BLOCK_SIZE=1024)
    return self


def fill__Tensor(*args, **kwargs):
    # Expecting (self, other)
    if len(args) >= 2:
        self = args[0]
        other = args[1]
    else:
        self = kwargs.get('self') or kwargs.get('input') or kwargs.get('tensor')
        other = kwargs.get('other') or kwargs.get('value') or kwargs.get('src')
    if self is None:
        raise ValueError("fill__Tensor requires a target tensor 'self'.")
    if other is None:
        raise ValueError("fill__Tensor requires a source tensor 'other'.")
    assert self.is_cuda, "fill__Tensor requires a CUDA tensor."
    assert self.is_contiguous(), "fill__Tensor currently supports only contiguous tensors."
    if other.numel() != 1:
        raise RuntimeError("fill__Tensor expects 'other' to be a 0-dim or 1-element tensor.")
    # Move/convert 'other' to match device and dtype of 'self'
    value_t = other.to(device=self.device, dtype=self.dtype).reshape(1)
    n_elements = self.numel()
    if n_elements == 0:
        return self
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fill_kernel[grid](self, value_t, n_elements, BLOCK_SIZE=1024)
    return self