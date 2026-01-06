import torch
import triton
import triton.language as tl


@triton.jit
def less_equal_scalar_kernel(
    x_ptr, out_ptr, scalar_val, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    cond = x <= scalar_val

    one = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    zero = tl.full([BLOCK_SIZE], 0.0, dtype=tl.float32)
    out = tl.where(cond, one, zero)

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def less_equal_tensor_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    cond = x <= y

    one = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    zero = tl.full([BLOCK_SIZE], 0.0, dtype=tl.float32)
    out = tl.where(cond, one, zero)

    tl.store(out_ptr + offsets, out, mask=mask)


def less_equal__Scalar(*args, **kwargs):
    # Expected usage: less_equal__Scalar(self, other_scalar)
    if len(args) >= 2:
        self, other = args[0], args[1]
    else:
        self = kwargs.get("self", None)
        other = kwargs.get("other", None)
    assert isinstance(self, torch.Tensor), "self must be a torch.Tensor"
    assert self.is_cuda, "Tensor must be on CUDA device for Triton kernel"
    assert self.numel() >= 0

    # Prepare working buffer in float32 contiguous
    n_elements = self.numel()
    if n_elements == 0:
        return self

    self_contig_f32 = self.contiguous().to(torch.float32)

    # Convert scalar to float32
    # Accept Python numbers and 0-dim tensors
    if isinstance(other, torch.Tensor):
        assert other.numel() == 1, "Scalar variant expects a scalar value"
        scalar_f32 = other.to(device=self.device, dtype=torch.float32).item()
    else:
        scalar_f32 = float(other)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    less_equal_scalar_kernel[grid](
        self_contig_f32,  # x_ptr
        self_contig_f32,  # out_ptr (in-place on temp buffer)
        scalar_f32,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Copy results back to original tensor dtype/layout
    if self.is_contiguous():
        # Direct copy if same storage layout
        self.copy_(self_contig_f32.to(self.dtype))
    else:
        self.copy_(self_contig_f32.to(self.dtype).view(self.shape))

    return self


def less_equal__Tensor(*args, **kwargs):
    # Expected usage: less_equal__Tensor(self, other_tensor)
    if len(args) >= 2:
        self, other = args[0], args[1]
    else:
        self = kwargs.get("self", None)
        other = kwargs.get("other", None)
    assert isinstance(self, torch.Tensor) and isinstance(
        other, torch.Tensor
    ), "Inputs must be torch.Tensors"
    assert (
        self.is_cuda and other.is_cuda
    ), "Tensors must be on CUDA device for Triton kernel"

    # Prepare working buffers in float32 contiguous
    n_elements = self.numel()
    if n_elements == 0:
        return self

    self_contig_f32 = self.contiguous().to(torch.float32)
    other_expanded = other.to(device=self.device, dtype=torch.float32).expand_as(self)
    other_contig_f32 = other_expanded.contiguous()

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    less_equal_tensor_kernel[grid](
        self_contig_f32,  # x_ptr
        other_contig_f32,  # y_ptr
        self_contig_f32,  # out_ptr (in-place on temp buffer)
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Copy results back to original tensor dtype/layout
    if self.is_contiguous():
        self.copy_(self_contig_f32.to(self.dtype))
    else:
        self.copy_(self_contig_f32.to(self.dtype).view(self.shape))

    return self
