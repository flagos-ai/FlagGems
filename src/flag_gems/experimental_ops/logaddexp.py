import torch
import triton
import triton.language as tl


@triton.jit
def logaddexp_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    xf32 = x.to(tl.float32)
    yf32 = y.to(tl.float32)

    delta = xf32 - yf32
    adelta = tl.abs(delta)
    m = tl.maximum(xf32, yf32)
    res = m + tl.log(1.0 + tl.exp(-adelta))

    out_ty = out_ptr.dtype.element_ty
    tl.store(out_ptr + offsets, res.to(out_ty), mask=mask)


def _ensure_cuda_tensor(obj, device, dtype):
    if torch.is_tensor(obj):
        return obj.to(device=device, dtype=dtype)
    else:
        return torch.tensor(obj, device=device, dtype=dtype)


def _common_float_dtype(x: torch.Tensor, y: torch.Tensor):
    dt = torch.result_type(x, y)
    if dt not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        dt = torch.get_default_dtype()
    return dt


def _launch_logaddexp_kernel(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor):
    assert x.is_cuda and y.is_cuda and out.is_cuda, "All tensors must be on CUDA device"
    assert (
        x.numel() == y.numel() == out.numel()
    ), "Input and output must have the same number of elements"

    x_flat = x.contiguous().view(-1)
    y_flat = y.contiguous().view(-1)
    out_flat = out.contiguous().view(-1)

    n_elements = out_flat.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    logaddexp_kernel[grid](x_flat, y_flat, out_flat, n_elements, BLOCK_SIZE=1024)

    # If out was non-contiguous, copy results back into original layout
    if not out.is_contiguous():
        out.copy_(out_flat.view_as(out))


def logaddexp(x, y):
    # Determine device
    device = None
    if torch.is_tensor(x) and x.is_cuda:
        device = x.device
    if device is None and torch.is_tensor(y) and y.is_cuda:
        device = y.device
    if device is None:
        raise ValueError("At least one input must be a CUDA tensor")

    # Determine dtype
    x_t = x if torch.is_tensor(x) else torch.tensor(x)
    y_t = y if torch.is_tensor(y) else torch.tensor(y)
    dtype = _common_float_dtype(x_t, y_t)

    # Convert to device and dtype
    x_t = _ensure_cuda_tensor(x, device, dtype)
    y_t = _ensure_cuda_tensor(y, device, dtype)

    # Broadcast
    xb, yb = torch.broadcast_tensors(x_t, y_t)

    # Allocate output
    out = torch.empty_like(xb, dtype=dtype, device=device)

    _launch_logaddexp_kernel(xb, yb, out)
    return out


def logaddexp_out(x, y, out):
    if not torch.is_tensor(out) or not out.is_cuda:
        raise ValueError("out must be a CUDA tensor")

    # Determine computation device and dtype from out
    device = out.device
    out_dtype = out.dtype
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("out dtype must be a floating point type")

    # Prepare inputs
    x_t = _ensure_cuda_tensor(x, device, out_dtype)
    y_t = _ensure_cuda_tensor(y, device, out_dtype)

    # Broadcast inputs
    xb, yb = torch.broadcast_tensors(x_t, y_t)

    # Ensure out shape matches
    if tuple(out.shape) != tuple(xb.shape):
        raise ValueError(
            f"out shape {tuple(out.shape)} does not match broadcasted shape {tuple(xb.shape)}"
        )

    _launch_logaddexp_kernel(xb, yb, out)
    return out
