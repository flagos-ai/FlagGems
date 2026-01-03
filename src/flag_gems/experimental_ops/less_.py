import torch
import triton
import triton.language as tl


@triton.jit
def less_inplace_kernel(x_ptr, y_ptr, n_elements, SCALAR: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    if SCALAR:
        y = tl.load(y_ptr)
    else:
        y = tl.load(y_ptr + offsets, mask=mask)

    cmp = x < y
    one = tl.full([BLOCK_SIZE], 1, x.dtype)
    zero = tl.full([BLOCK_SIZE], 0, x.dtype)
    out = tl.where(cmp, one, zero)
    tl.store(x_ptr + offsets, out, mask=mask)


def _launch_less_inplace(x: torch.Tensor, y_tensor: torch.Tensor, scalar: bool):
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert y_tensor.is_cuda, "Other/scalar tensor must be on CUDA device"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert y_tensor.is_contiguous(), "Other/scalar tensor must be contiguous"
    n_elements = x.numel()
    assert scalar or (y_tensor.numel() == n_elements), "Shape mismatch for elementwise comparison"

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    less_inplace_kernel[grid](x, y_tensor, n_elements, SCALAR=scalar, BLOCK_SIZE=1024)
    return x


def less__Scalar(self: torch.Tensor, other):
    x = self
    # Ensure contiguity; if not, operate on a contiguous copy and copy back.
    needs_copy_back = not x.is_contiguous()
    if needs_copy_back:
        x_work = x.contiguous()
    else:
        x_work = x

    # Make scalar tensor on device, cast to x dtype
    y_tensor = torch.tensor([other], dtype=x_work.dtype, device=x_work.device).contiguous()

    _launch_less_inplace(x_work, y_tensor, scalar=True)

    if needs_copy_back:
        self.copy_(x_work)
    return self


def less__Tensor(self: torch.Tensor, other: torch.Tensor):
    x = self
    # Ensure contiguity; if not, operate on a contiguous copy and copy back.
    needs_copy_back = not x.is_contiguous()
    if needs_copy_back:
        x_work = x.contiguous()
    else:
        x_work = x

    # Prepare 'other' on same device/dtype and choose scalar or tensor path
    if other.numel() == 1:
        y_tensor = other.to(dtype=x_work.dtype, device=x_work.device).reshape(1).contiguous()
        scalar = True
    else:
        assert other.numel() == x_work.numel(), "Shape mismatch for elementwise comparison"
        y_tensor = other.to(dtype=x_work.dtype, device=x_work.device).contiguous()
        scalar = False

    _launch_less_inplace(x_work, y_tensor, scalar=scalar)

    if needs_copy_back:
        self.copy_(x_work)
    return self