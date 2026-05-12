import triton
import triton.language as tl


@triton.jit
def fill_scalar_kernel(out_ptr, value_scalar, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, value_scalar, mask=mask)


@triton.jit
def fill_tensor_kernel(out_ptr, value_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(value_ptr)
    tl.store(out_ptr + offsets, value, mask=mask)
