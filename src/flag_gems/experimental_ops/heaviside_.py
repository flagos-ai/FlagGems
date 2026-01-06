import torch
import triton
import triton.language as tl


@triton.jit
def heaviside_(x_ptr, v_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)

    is_zero = x == 0
    is_neg = x < 0
    is_pos = x > 0

    # For NaN handling on floating types: if none of the comparisons are true, use x + x to propagate NaN.
    res = tl.where(is_zero, v, tl.where(is_neg, 0, tl.where(is_pos, 1, x + x)))
    tl.store(x_ptr + offsets, res, mask=mask)


# Keep a handle to the kernel (its __name__ is "heaviside_")
heaviside__kernel = heaviside_


def heaviside_(*args, **kwargs):
    # Parse arguments similar to torch.ops.aten.heaviside_(self, values)
    if len(args) >= 2:
        x, values = args[0], args[1]
    else:
        # Fallback to kwargs if provided
        x = kwargs.get("input", kwargs.get("self", None))
        values = kwargs.get("values", None)
    assert (
        x is not None and values is not None
    ), "heaviside_ requires two arguments: input tensor and values."

    # Ensure CUDA tensors
    assert x.is_cuda, "Input tensor must be on CUDA device."
    assert x.is_contiguous(), "Input tensor must be contiguous."

    # Prepare values tensor (support scalar or tensor), broadcast to input shape and ensure same dtype/device
    if not torch.is_tensor(values):
        v_tensor = torch.as_tensor(values, device=x.device, dtype=x.dtype)
    else:
        v_tensor = values.to(device=x.device, dtype=x.dtype)

    v_tensor = v_tensor.expand_as(x).contiguous()
    assert (
        v_tensor.is_cuda and v_tensor.is_contiguous()
    ), "Values tensor must be CUDA and contiguous after expansion."

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    heaviside__kernel[grid](x, v_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x
