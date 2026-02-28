import torch
import triton
import triton.language as tl


@triton.jit
def fmin_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = tl.minimum(x, y)
    tl.store(out_ptr + offsets, out, mask=mask)


def _to_tensor(x, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        t = x
        if device is not None and t.device != device:
            t = t.to(device)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return t
    return torch.tensor(x, device=device, dtype=dtype)


def _prepare_inputs(a, b, out=None):
    # Determine target device
    dev = None
    if isinstance(out, torch.Tensor):
        dev = out.device
    else:
        if isinstance(a, torch.Tensor):
            dev = a.device
        if isinstance(b, torch.Tensor):
            dev = b.device if dev is None else dev
    if dev is None:
        dev = torch.device("cuda")
    # Convert to tensors on the target device
    a = _to_tensor(a, device=dev)
    b = _to_tensor(b, device=dev)
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise ValueError(
            "Inputs must be CUDA tensors or convertible to CUDA tensors for Triton kernels."
        )
    # Broadcast
    a_b, b_b = torch.broadcast_tensors(a, b)
    # Determine output dtype
    out_dtype = torch.result_type(a_b, b_b)
    if out_dtype.is_complex:
        raise TypeError("fmin does not support complex dtypes.")
    # Compute dtype for kernel (avoid bool in kernel by using int8)
    compute_dtype = torch.int8 if out_dtype == torch.bool else out_dtype
    a_c = a_b.to(compute_dtype).contiguous()
    b_c = b_b.to(compute_dtype).contiguous()
    return a_c, b_c, out_dtype, compute_dtype


def fmin(a, b):
    a_c, b_c, out_dtype, compute_dtype = _prepare_inputs(a, b, out=None)
    out_shape = a_c.shape  # same as b_c.shape after broadcast
    # Allocate outputs
    if compute_dtype == out_dtype:
        out = torch.empty(out_shape, dtype=out_dtype, device=a_c.device)
        out_c = out
    else:
        out = torch.empty(out_shape, dtype=out_dtype, device=a_c.device)
        out_c = torch.empty(out_shape, dtype=compute_dtype, device=a_c.device)
    n_elements = out_c.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    fmin_kernel[grid](a_c, b_c, out_c, n_elements, BLOCK_SIZE=1024)
    if out_c.dtype != out.dtype:
        out.copy_(out_c.to(out_dtype))
    return out


def fmin_out(a, b, out):
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a Tensor")
    a_c, b_c, out_dtype, compute_dtype = _prepare_inputs(a, b, out=out)
    # Validate out tensor shape/dtype/device
    expected_shape = a_c.shape
    if out.device != a_c.device:
        raise ValueError("out tensor must be on the same device as inputs.")
    if out.dtype != out_dtype:
        raise TypeError(f"out tensor has dtype {out.dtype}, expected {out_dtype}.")
    if tuple(out.shape) != tuple(expected_shape):
        raise ValueError(
            f"out tensor has shape {tuple(out.shape)}, expected {tuple(expected_shape)} after broadcasting."
        )
    # Prepare a contiguous buffer to write into
    if compute_dtype == out_dtype and out.is_contiguous():
        out_c = out
    else:
        # If dtype conversion is needed or out is non-contiguous, use a temporary buffer
        out_c = torch.empty(expected_shape, dtype=compute_dtype, device=out.device)
    n_elements = out_c.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    fmin_kernel[grid](a_c, b_c, out_c, n_elements, BLOCK_SIZE=1024)
    # Move result into out if we used a temporary buffer or dtype differs
    if out_c is not out:
        if out_c.dtype != out.dtype:
            out.copy_(out_c.to(out.dtype))
        else:
            if out.is_contiguous():
                out.copy_(out_c)
            else:
                out.view_as(out.contiguous()).copy_(out_c)
    return out
