import torch
import triton
import triton.language as tl


@triton.jit
def lt_inplace_kernel(x_ptr, y_ptr, n_elements, IS_SCALAR: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if IS_SCALAR:
        s = tl.load(y_ptr)  # y_ptr points to a single element tensor
        cmp = x < s
    else:
        y = tl.load(y_ptr + offsets, mask=mask)
        cmp = x < y

    # Cast boolean comparison result to the dtype of x for in-place store
    one = tl.full(x.shape, 1, x.dtype)
    zero = tl.full(x.shape, 0, x.dtype)
    out = tl.where(cmp, one, zero)

    tl.store(x_ptr + offsets, out, mask=mask)


def lt__Tensor(self: torch.Tensor, other: torch.Tensor):
    assert self.is_cuda, "Input tensor must be on CUDA device"
    assert other.is_cuda, "Other tensor must be on CUDA device"
    assert not self.is_complex(), "Complex dtypes are not supported"
    assert self.is_contiguous(), "Only contiguous tensors are supported for in-place lt_"
    # Support either same numel or scalar-like other (numel == 1)
    if other.numel() == 1:
        # Ensure dtype matches self for comparison
        scalar_buf = other.to(self.dtype).reshape(1).contiguous()
        n_elements = self.numel()
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        lt_inplace_kernel[grid](
            self, scalar_buf, n_elements, IS_SCALAR=True, BLOCK_SIZE=BLOCK_SIZE
        )
        return self
    else:
        assert self.numel() == other.numel(), "Shapes must match or other must be scalar"
        assert other.is_contiguous(), "Only contiguous tensors are supported for other"
        # Cast other to self dtype for comparison
        other_buf = other.to(self.dtype).contiguous()
        n_elements = self.numel()
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        lt_inplace_kernel[grid](
            self, other_buf, n_elements, IS_SCALAR=False, BLOCK_SIZE=BLOCK_SIZE
        )
        return self


def lt__Scalar(self: torch.Tensor, other):
    assert self.is_cuda, "Input tensor must be on CUDA device"
    assert not self.is_complex(), "Complex dtypes are not supported"
    assert self.is_contiguous(), "Only contiguous tensors are supported for in-place lt_"
    # Create a 1-element tensor on device with same dtype as self for scalar compare
    scalar_buf = torch.tensor([other], device=self.device, dtype=self.dtype).contiguous()
    n_elements = self.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    lt_inplace_kernel[grid](
        self, scalar_buf, n_elements, IS_SCALAR=True, BLOCK_SIZE=BLOCK_SIZE
    )
    return self