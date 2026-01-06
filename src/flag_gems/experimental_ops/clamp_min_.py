from numbers import Number

import torch
import triton
import triton.language as tl


@triton.jit
def clamp_min_inplace_kernel(
    x_ptr,  # *Pointer* to input tensor (in-place mutated).
    other_ptr,  # *Pointer* to other tensor if using Tensor min, unused otherwise.
    n_elements,  # Number of elements in the tensor.
    min_val,  # Scalar minimum value if not using Tensor min.
    min_is_tensor: tl.constexpr,  # Whether to use tensor-based min.
    BLOCK_SIZE: tl.constexpr,  # Elements per program.
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    if min_is_tensor:
        y = tl.load(other_ptr + offsets, mask=mask)
    else:
        y = min_val

    out = tl.where(x < y, y, x)
    tl.store(x_ptr + offsets, out, mask=mask)


def _launch_clamp_min_inplace(
    x: torch.Tensor, other: torch.Tensor | None, min_val: Number | None
):
    assert x.is_cuda, "Input tensor must be on CUDA device."
    assert x.is_contiguous(), "Input tensor must be contiguous."
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    if other is not None:
        assert (
            other.is_cuda and other.is_contiguous()
        ), "Other tensor must be CUDA and contiguous."
        if other.numel() == 1:
            # Treat single-element tensor as scalar min
            min_scalar = other.item()
            clamp_min_inplace_kernel[grid](
                x, x, n_elements, min_scalar, min_is_tensor=False, BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            assert (
                other.numel() == n_elements
            ), "Other tensor must have the same number of elements as input."
            assert (
                other.dtype == x.dtype
            ), "Dtype of other tensor must match input dtype."
            clamp_min_inplace_kernel[grid](
                x, other, n_elements, 0.0, min_is_tensor=True, BLOCK_SIZE=BLOCK_SIZE
            )
    else:
        assert isinstance(
            min_val, Number
        ), "min must be a Python number when other tensor is not provided."
        clamp_min_inplace_kernel[grid](
            x, x, n_elements, min_val, min_is_tensor=False, BLOCK_SIZE=BLOCK_SIZE
        )
    return x


def clamp_min_(*args, **kwargs):
    # Expected: (Tensor self, Scalar min)
    # If a Tensor is provided for min, delegate to the Tensor variant.
    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("self", kwargs.get("input"))
    if x is None:
        raise ValueError("clamp_min_: expected a tensor as the first argument")

    # Determine min argument
    if len(args) >= 2:
        min_arg = args[1]
    else:
        min_arg = kwargs.get("min", kwargs.get("other"))

    if isinstance(min_arg, torch.Tensor):
        return clamp_min__Tensor(x, min_arg)
    else:
        return _launch_clamp_min_inplace(x, None, min_arg)


def clamp_min__Tensor(*args, **kwargs):
    # Expected: (Tensor self, Tensor min)
    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("self", kwargs.get("input"))
    if x is None:
        raise ValueError("clamp_min__Tensor: expected a tensor as the first argument")

    if len(args) >= 2:
        other = args[1]
    else:
        other = kwargs.get("min", kwargs.get("other"))
    if other is None or not isinstance(other, torch.Tensor):
        raise ValueError("clamp_min__Tensor: expected a Tensor for 'min' argument")

    return _launch_clamp_min_inplace(x, other, None)
