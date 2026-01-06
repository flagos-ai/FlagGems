import torch
import triton
import triton.language as tl


def _torch_dtype_to_triton(dtype: torch.dtype):
    mapping = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
        torch.float64: tl.float64,
        torch.int8: tl.int8,
        torch.uint8: tl.uint8,
        torch.int16: tl.int16,
        torch.int32: tl.int32,
        torch.int64: tl.int64,
        torch.bool: tl.int1,
    }
    if dtype not in mapping:
        raise NotImplementedError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


@triton.jit
def _not_equal_scalar_kernel(
    x_ptr,  # in-out: pointer to self tensor
    scalar,  # scalar to compare against
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=tl.zeros([BLOCK_SIZE], dtype=DTYPE))
    other = tl.full([BLOCK_SIZE], scalar, dtype=DTYPE)
    cmp = x != other
    out = tl.cast(cmp, DTYPE)
    tl.store(x_ptr + offsets, out, mask=mask)


@triton.jit
def _not_equal_tensor_kernel(
    x_ptr,  # in-out: pointer to self tensor
    y_ptr,  # pointer to other tensor
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=tl.zeros([BLOCK_SIZE], dtype=DTYPE))
    y = tl.load(y_ptr + offsets, mask=mask, other=tl.zeros([BLOCK_SIZE], dtype=DTYPE))
    cmp = x != y
    out = tl.cast(cmp, DTYPE)
    tl.store(x_ptr + offsets, out, mask=mask)


def not_equal__Scalar(self: torch.Tensor, other):
    # In-place: self <- (self != other) cast into self.dtype as 0/1
    if isinstance(other, torch.Tensor):
        if other.numel() != 1:
            raise TypeError("Scalar variant expects a Python number or 0-dim tensor.")
        other = other.item()

    assert self.is_cuda, "Input tensor must be on CUDA device"
    assert self.is_contiguous(), "Only contiguous tensors are supported"
    n_elements = self.numel()
    if n_elements == 0:
        return self

    DTYPE = _torch_dtype_to_triton(self.dtype)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _not_equal_scalar_kernel[grid](
        self, other, n_elements, BLOCK_SIZE=BLOCK_SIZE, DTYPE=DTYPE
    )
    return self


def not_equal__Tensor(self: torch.Tensor, other: torch.Tensor):
    # In-place: self <- (self != other) cast into self.dtype as 0/1
    assert self.is_cuda and other.is_cuda, "Input tensors must be on CUDA device"
    assert (
        self.is_contiguous() and other.is_contiguous()
    ), "Only contiguous tensors are supported"
    assert self.dtype == other.dtype, "Dtype mismatch is not supported"
    assert (
        self.numel() == other.numel()
    ), "Only tensors with the same number of elements are supported"

    n_elements = self.numel()
    if n_elements == 0:
        return self

    DTYPE = _torch_dtype_to_triton(self.dtype)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _not_equal_tensor_kernel[grid](
        self, other, n_elements, BLOCK_SIZE=BLOCK_SIZE, DTYPE=DTYPE
    )
    return self
