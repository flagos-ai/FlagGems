import torch
import triton
import triton.language as tl


@triton.jit
def greater_inplace_kernel(
    x_ptr,  # *Pointer* to self tensor (modified in-place)
    y_ptr,  # *Pointer* to other tensor (if tensor path)
    scalar_ptr,  # *Pointer* to 1-element scalar tensor (if scalar path)
    n_elements,  # Number of elements
    IS_SCALAR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    if IS_SCALAR:
        s = tl.load(scalar_ptr)  # scalar value
        y = s
    else:
        y = tl.load(y_ptr + offsets, mask=mask, other=0)

    cmp = x > y
    one = tl.full([BLOCK_SIZE], 1, x.dtype)
    zero = tl.full([BLOCK_SIZE], 0, x.dtype)
    out = tl.where(cmp, one, zero)
    tl.store(x_ptr + offsets, out, mask=mask)


def _launch_greater_inplace_tensor(self: torch.Tensor, other: torch.Tensor):
    assert self.is_cuda and other.is_cuda, "Tensors must be on CUDA device"
    assert (
        self.is_contiguous()
    ), "Only contiguous tensors are supported for in-place operation"
    # Prepare other: broadcast to self and ensure same dtype and contiguity
    other_exp = (
        other.to(dtype=self.dtype, device=self.device).expand_as(self).contiguous()
    )
    n_elements = self.numel()
    if n_elements == 0:
        return self
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    greater_inplace_kernel[grid](
        self, other_exp, self, n_elements, IS_SCALAR=False, BLOCK_SIZE=1024
    )
    return self


def _launch_greater_inplace_scalar(self: torch.Tensor, other):
    assert self.is_cuda, "Tensor must be on CUDA device"
    assert (
        self.is_contiguous()
    ), "Only contiguous tensors are supported for in-place operation"
    # Make a 1-element tensor holding the scalar, converted to self's dtype/device
    if isinstance(other, torch.Tensor):
        assert other.numel() == 1, "Scalar variant expects a single value"
        scalar_t = other.to(dtype=self.dtype, device=self.device).reshape(())
    else:
        scalar_t = torch.tensor(other, dtype=self.dtype, device=self.device)
    n_elements = self.numel()
    if n_elements == 0:
        return self
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    greater_inplace_kernel[grid](
        self, self, scalar_t, n_elements, IS_SCALAR=True, BLOCK_SIZE=1024
    )
    return self


def greater__Scalar(*args, **kwargs):
    # Expected: (self, other)
    self = args[0]
    other = args[1]
    return _launch_greater_inplace_scalar(self, other)


def greater__Tensor(*args, **kwargs):
    # Expected: (self, other)
    self = args[0]
    other = args[1]
    return _launch_greater_inplace_tensor(self, other)
