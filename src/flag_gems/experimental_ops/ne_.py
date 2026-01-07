import torch
import triton
import triton.language as tl


@triton.jit
def ne_inplace_kernel(x_ptr, y_ptr, n_elements, y_numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    # broadcast y if needed
    y_offsets = offsets % y_numel
    y = tl.load(y_ptr + y_offsets, mask=mask)

    cmp = x != y
    one = tl.full(x.shape, 1, x.dtype)
    zero = tl.full(x.shape, 0, x.dtype)
    out = tl.where(cmp, one, zero)

    tl.store(x_ptr + offsets, out, mask=mask)


def _launch_ne_inplace(x: torch.Tensor, y_buf: torch.Tensor, y_numel: int):
    assert x.is_cuda and y_buf.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype == y_buf.dtype, "x and y must have the same dtype"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert y_buf.is_contiguous(), "y buffer must be contiguous"
    n_elements = x.numel()
    if n_elements == 0:
        return x
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    ne_inplace_kernel[grid](x, y_buf, n_elements, y_numel, BLOCK_SIZE=BLOCK_SIZE)
    return x


def ne__Scalar(self: torch.Tensor, other):
    # other is a Python scalar or 0-d tensor
    if isinstance(other, torch.Tensor):
        assert other.numel() == 1, "Scalar overload expects a single value"
        other = other.item()
    y_buf = (
        torch.tensor(other, dtype=self.dtype, device=self.device)
        .reshape(1)
        .contiguous()
    )
    return _launch_ne_inplace(self, y_buf, 1)


def ne__Tensor(self: torch.Tensor, other: torch.Tensor):
    assert other.device == self.device, "Tensors must be on the same device"
    # Handle scalar-like tensor
    if other.numel() == 1:
        y_buf = other.to(dtype=self.dtype).reshape(1).contiguous()
        return _launch_ne_inplace(self, y_buf, 1)
    # Broadcast to self's shape
    try:
        other_exp = other.to(dtype=self.dtype).expand_as(self)
    except RuntimeError as e:
        raise RuntimeError(
            f"Incompatible shapes for broadcasting: {self.shape} and {other.shape}"
        ) from e
    assert (
        self.numel() == other_exp.numel()
    ), "Broadcasted tensor must match the number of elements"
    y_buf = other_exp.contiguous().view(-1)
    return _launch_ne_inplace(self, y_buf, y_buf.numel())
