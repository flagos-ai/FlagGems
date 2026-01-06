import torch
import triton
import triton.language as tl


@triton.jit
def _heaviside_kernel(x_ptr, v_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)

    zeros = x - x
    ones = zeros + 1
    is_pos = x > zeros
    is_zero = x == zeros
    out = tl.where(is_zero, v, tl.where(is_pos, ones, zeros))

    tl.store(out_ptr + offsets, out, mask=mask)


def heaviside(input, values):
    # Prepare tensors
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values, device=input.device)
    # Broadcast
    x_b, v_b = torch.broadcast_tensors(input, values)
    # Dtype promotion
    out_dtype = torch.result_type(x_b, v_b)
    x = x_b.to(dtype=out_dtype).contiguous()
    v = v_b.to(dtype=out_dtype).contiguous()

    # Allocate output
    out = torch.empty_like(x)

    # Launch kernel
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _heaviside_kernel[grid](x, v, out, n_elements, BLOCK_SIZE=1024)
    return out


def heaviside_out(input, values, out):
    # Prepare tensors
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values, device=input.device)
    # Broadcast
    x_b, v_b = torch.broadcast_tensors(input, values)
    # Dtype promotion
    expected_dtype = torch.result_type(x_b, v_b)
    expected_shape = x_b.shape
    device = x_b.device
    # Check output tensor
    if out.device != device:
        raise ValueError("out tensor device must match input device")
    if out.dtype != expected_dtype:
        raise ValueError("out tensor dtype must be the result type of input and values")
    if out.shape != expected_shape:
        raise ValueError(
            "out tensor shape must be the broadcasted shape of input and values"
        )

    x = x_b.to(dtype=expected_dtype).contiguous()
    v = v_b.to(dtype=expected_dtype).contiguous()

    # If out is contiguous, write directly; otherwise use a temp and copy
    if out.is_contiguous():
        target = out
    else:
        target = torch.empty_like(out, memory_format=torch.contiguous_format)

    n_elements = target.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _heaviside_kernel[grid](x, v, target, n_elements, BLOCK_SIZE=1024)

    if target is not out:
        out.copy_(target)
    return out
