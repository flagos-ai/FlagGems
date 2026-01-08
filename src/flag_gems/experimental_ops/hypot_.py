import torch
import triton
import triton.language as tl


@triton.jit
def hypot_(
    x_ptr,  # Pointer to first input (will be output if in-place).
    y_ptr,  # Pointer to second input (broadcasted/contiguous).
    out_ptr,  # Pointer to output buffer.
    n_elements,  # Number of elements to process.
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    x32 = x.to(tl.float32)
    y32 = y.to(tl.float32)
    out32 = tl.sqrt(x32 * x32 + y32 * y32)

    out_cast = out32.to(x.dtype)
    tl.store(out_ptr + offsets, out_cast, mask=mask)


_hypot_kernel = hypot_


def hypot_(*args, **kwargs):
    # Extract arguments similar to torch.ops.aten.hypot_(self, other)
    x = None
    other = None
    if len(args) >= 1:
        x = args[0]
    if len(args) >= 2:
        other = args[1]
    if x is None:
        x = kwargs.get("input", kwargs.get("self", None))
    if other is None:
        other = kwargs.get("other", None)

    if x is None or other is None:
        raise TypeError("hypot_ expects two arguments: self and other")

    if not isinstance(x, torch.Tensor):
        raise TypeError("self must be a torch.Tensor")
    if not x.is_cuda:
        raise ValueError("hypot_ Triton kernel only supports CUDA tensors")

    device = x.device

    # Prepare 'other' on the same device and dtype as x (in-place ops keep dtype)
    if isinstance(other, torch.Tensor):
        other_t = other.to(device)
    else:
        other_t = torch.tensor(other, device=device)

    # In-place must keep dtype of x; cast other to x.dtype
    if other_t.dtype != x.dtype:
        other_t = other_t.to(x.dtype)

    # Broadcast other to x's shape
    try:
        other_b = torch.broadcast_to(other_t, x.shape)
    except Exception:
        other_b = torch.broadcast_tensors(other_t, x)[0]

    # Ensure contiguous buffers for kernel
    x_c = x if x.is_contiguous() else x.contiguous()
    other_c = other_b if other_b.is_contiguous() else other_b.contiguous()

    n_elements = x.numel()
    if n_elements == 0:
        return x

    # If x is contiguous, write directly in-place into x; otherwise write to temp and copy back.
    out_buf = x_c if x.is_contiguous() else torch.empty_like(x_c)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _hypot_kernel[grid](x_c, other_c, out_buf, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if not x.is_contiguous():
        x.copy_(out_buf)

    return x
