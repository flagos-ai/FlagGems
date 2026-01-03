import torch
import triton
import triton.language as tl


@triton.jit
def le_inplace_kernel(
    x_ptr,             # *Pointer* to input/output tensor (in-place)
    y_ptr,             # *Pointer* to second tensor (ignored if is_scalar=True)
    n_elements,        # Number of elements
    scalar,            # Scalar value for comparison when is_scalar=True
    IS_SCALAR: tl.constexpr,   # Whether we are comparing against a scalar
    X_IS_BOOL: tl.constexpr,   # Whether x dtype is bool
    BLOCK_SIZE: tl.constexpr,  # Elements per program
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if IS_SCALAR:
        y = tl.full((BLOCK_SIZE,), scalar, x.dtype)
    else:
        y = tl.load(y_ptr + offsets, mask=mask).to(x.dtype)

    cmp_res = x <= y

    if X_IS_BOOL:
        out = cmp_res
    else:
        one = tl.full((BLOCK_SIZE,), 1, x.dtype)
        zero = tl.full((BLOCK_SIZE,), 0, x.dtype)
        out = tl.where(cmp_res, one, zero)

    tl.store(x_ptr + offsets, out, mask=mask)


def _launch_le_inplace(x: torch.Tensor, y: torch.Tensor = None, scalar=None):
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.is_contiguous(), "Only contiguous tensors are supported"

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    x_is_bool = x.dtype == torch.bool

    if scalar is not None:
        # Use scalar path; pass x as y_ptr placeholder (unused)
        le_inplace_kernel[grid](
            x, x, n_elements, scalar,
            IS_SCALAR=True, X_IS_BOOL=x_is_bool, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        assert y is not None, "Either a tensor 'y' or a scalar must be provided"
        assert y.is_cuda and y.device == x.device, "Both tensors must be on the same CUDA device"
        if y.numel() == 1:
            # Treat 0-dim or single element tensor as scalar
            scalar_val = y.item()
            le_inplace_kernel[grid](
                x, x, n_elements, scalar_val,
                IS_SCALAR=True, X_IS_BOOL=x_is_bool, BLOCK_SIZE=BLOCK_SIZE
            )
            return x
        assert y.is_contiguous(), "Only contiguous tensors are supported"
        assert y.numel() == x.numel(), "Tensors must have the same number of elements"
        if y.dtype != x.dtype:
            y = y.to(dtype=x.dtype)
        le_inplace_kernel[grid](
            x, y, n_elements, 0,  # scalar is unused in tensor path
            IS_SCALAR=False, X_IS_BOOL=x_is_bool, BLOCK_SIZE=BLOCK_SIZE
        )
    return x


def le__Scalar(self: torch.Tensor, other):
    """
    In-place self <= other (scalar). Returns self.
    """
    return _launch_le_inplace(self, scalar=other)


def le__Tensor(self: torch.Tensor, other: torch.Tensor):
    """
    In-place self <= other (tensor). Returns self.
    """
    return _launch_le_inplace(self, y=other)