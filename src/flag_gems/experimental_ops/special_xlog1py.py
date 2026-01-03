import torch
import triton
import triton.language as tl


@triton.jit
def _xlog1py_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x * tl.log(1.0 + y)
    tl.store(out_ptr + offsets, out, mask=mask)


def _ensure_cuda_tensor(t):
    if not isinstance(t, torch.Tensor):
        raise TypeError("Expected a torch.Tensor")
    if t.device.type != "cuda":
        raise ValueError("Tensors must be on CUDA device")
    return t


def _prepare_inputs(x, y):
    x = _ensure_cuda_tensor(x)
    y = _ensure_cuda_tensor(y)
    xb, yb = torch.broadcast_tensors(x, y)
    dtype_out = torch.result_type(xb, yb)
    xb_fp32 = xb.to(torch.float32).contiguous()
    yb_fp32 = yb.to(torch.float32).contiguous()
    return xb_fp32, yb_fp32, dtype_out


def _launch_xlog1py(x_fp32, y_fp32, out_fp32):
    n_elements = out_fp32.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _xlog1py_kernel[grid](x_fp32, y_fp32, out_fp32, n_elements, BLOCK_SIZE=1024)


def special_xlog1py(x, y):
    xb_fp32, yb_fp32, dtype_out = _prepare_inputs(x, y)
    out_fp32 = torch.empty_like(xb_fp32)
    _launch_xlog1py(xb_fp32, yb_fp32, out_fp32)
    if dtype_out == torch.float32:
        return out_fp32
    else:
        return out_fp32.to(dtype_out)


def special_xlog1py_other_scalar(x, other):
    x = _ensure_cuda_tensor(x)
    other_tensor = torch.as_tensor(other, device=x.device, dtype=x.dtype)
    xb_fp32, yb_fp32, dtype_out = _prepare_inputs(x, other_tensor)
    out_fp32 = torch.empty_like(xb_fp32)
    _launch_xlog1py(xb_fp32, yb_fp32, out_fp32)
    if dtype_out == torch.float32:
        return out_fp32
    else:
        return out_fp32.to(dtype_out)


def special_xlog1py_self_scalar(self, other):
    other = _ensure_cuda_tensor(other)
    self_tensor = torch.as_tensor(self, device=other.device, dtype=other.dtype)
    xb_fp32, yb_fp32, dtype_out = _prepare_inputs(self_tensor, other)
    out_fp32 = torch.empty_like(xb_fp32)
    _launch_xlog1py(xb_fp32, yb_fp32, out_fp32)
    if dtype_out == torch.float32:
        return out_fp32
    else:
        return out_fp32.to(dtype_out)


def special_xlog1py_out(x, y, out):
    out = _ensure_cuda_tensor(out)
    xb_fp32, yb_fp32, dtype_out = _prepare_inputs(_ensure_cuda_tensor(x), _ensure_cuda_tensor(y))
    # Validate output shape
    expected_shape = torch.broadcast_shapes(xb_fp32.shape, yb_fp32.shape)
    if out.shape != expected_shape:
        raise ValueError(f"Out tensor has shape {out.shape}, expected {expected_shape}")
    out_fp32 = torch.empty(expected_shape, device=out.device, dtype=torch.float32)
    _launch_xlog1py(xb_fp32, yb_fp32, out_fp32)
    out.copy_(out_fp32.to(out.dtype))
    return out


def special_xlog1py_self_scalar_out(self, other, out):
    out = _ensure_cuda_tensor(out)
    other = _ensure_cuda_tensor(other)
    self_tensor = torch.as_tensor(self, device=other.device, dtype=other.dtype)
    xb_fp32, yb_fp32, dtype_out = _prepare_inputs(self_tensor, other)
    expected_shape = torch.broadcast_shapes(xb_fp32.shape, yb_fp32.shape)
    if out.shape != expected_shape:
        raise ValueError(f"Out tensor has shape {out.shape}, expected {expected_shape}")
    out_fp32 = torch.empty(expected_shape, device=out.device, dtype=torch.float32)
    _launch_xlog1py(xb_fp32, yb_fp32, out_fp32)
    out.copy_(out_fp32.to(out.dtype))
    return out


def special_xlog1py_other_scalar_out(x, other, out):
    out = _ensure_cuda_tensor(out)
    x = _ensure_cuda_tensor(x)
    other_tensor = torch.as_tensor(other, device=x.device, dtype=x.dtype)
    xb_fp32, yb_fp32, dtype_out = _prepare_inputs(x, other_tensor)
    expected_shape = torch.broadcast_shapes(xb_fp32.shape, yb_fp32.shape)
    if out.shape != expected_shape:
        raise ValueError(f"Out tensor has shape {out.shape}, expected {expected_shape}")
    out_fp32 = torch.empty(expected_shape, device=out.device, dtype=torch.float32)
    _launch_xlog1py(xb_fp32, yb_fp32, out_fp32)
    out.copy_(out_fp32.to(out.dtype))
    return out