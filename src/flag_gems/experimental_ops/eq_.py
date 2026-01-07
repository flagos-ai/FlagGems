import torch
import triton
import triton.language as tl


@triton.jit
def eq_inplace_scalar_kernel(
    self_ptr, scalar_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(self_ptr + offsets, mask=mask)
    s = tl.load(scalar_ptr)  # scalar
    eq = x == s
    result = eq.to(x.dtype)
    tl.store(self_ptr + offsets, result, mask=mask)


@triton.jit
def eq_inplace_tensor_kernel(self_ptr, other_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(self_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)
    eq = x == y
    result = eq.to(x.dtype)
    tl.store(self_ptr + offsets, result, mask=mask)


def eq__Scalar(*args, **kwargs):
    # Parse inputs: expect (self, other)
    if len(args) >= 2:
        self, other = args[0], args[1]
    else:
        # Support keyword styles
        self = kwargs.get("self", kwargs.get("input"))
        other = kwargs.get("other", None)
    if self is None:
        raise ValueError("eq__Scalar expects a 'self' tensor as the first argument")
    if other is None:
        raise ValueError("eq__Scalar expects 'other' scalar as the second argument")
    if not isinstance(self, torch.Tensor):
        raise TypeError("eq__Scalar: 'self' must be a torch.Tensor")
    if not self.is_cuda:
        raise ValueError("eq__Scalar requires CUDA tensors")

    # Convert other to a scalar tensor of the same dtype/device as self
    if isinstance(other, torch.Tensor):
        if other.numel() != 1:
            # If other is not scalar, redirect to tensor variant
            return eq__Tensor(self, other)
        other_val = other.item()
    else:
        other_val = other

    scalar_tensor = torch.tensor(other_val, dtype=self.dtype, device=self.device)

    # For simplicity, require contiguous self (common for in-place ops in kernels)
    if not self.is_contiguous():
        raise ValueError("eq__Scalar currently supports only contiguous 'self' tensors")

    n_elements = self.numel()
    if n_elements == 0:
        return self

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    eq_inplace_scalar_kernel[grid](self, scalar_tensor, n_elements, BLOCK_SIZE=1024)
    return self


def eq__Tensor(*args, **kwargs):
    # Parse inputs: expect (self, other)
    if len(args) >= 2:
        self, other = args[0], args[1]
    else:
        # Support keyword styles
        self = kwargs.get("self", kwargs.get("input"))
        other = kwargs.get("other", None)
    if self is None or other is None:
        raise ValueError("eq__Tensor expects 'self' and 'other' tensors")
    if not isinstance(self, torch.Tensor) or not isinstance(other, torch.Tensor):
        raise TypeError("eq__Tensor: both 'self' and 'other' must be torch.Tensors")
    if not self.is_cuda:
        raise ValueError("eq__Tensor requires CUDA tensors")

    # Match device and dtype to self for comparison in kernel
    other = other.to(device=self.device, dtype=self.dtype)

    # Broadcast other to self's shape, then make contiguous for simple 1D indexing
    try:
        other_b = other.expand_as(self)
    except Exception as e:
        raise ValueError(f"eq__Tensor: tensors are not broadcastable: {e}")
    other_c = other_b.contiguous()

    # Require contiguous self (in-place on original storage)
    if not self.is_contiguous():
        raise ValueError("eq__Tensor currently supports only contiguous 'self' tensors")

    n_elements = self.numel()
    if other_c.numel() != n_elements:
        raise ValueError(
            "eq__Tensor: broadcasted 'other' does not match number of elements in 'self'"
        )
    if n_elements == 0:
        return self

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    eq_inplace_tensor_kernel[grid](self, other_c, n_elements, BLOCK_SIZE=1024)
    return self
