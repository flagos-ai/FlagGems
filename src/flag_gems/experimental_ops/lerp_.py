import torch
import triton
import triton.language as tl


@triton.jit
def _lerp_inplace_scalar_kernel(x_ptr, end_ptr, n_elements, weight, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_raw = tl.load(x_ptr + offsets, mask=mask, other=0)
    y_raw = tl.load(end_ptr + offsets, mask=mask, other=0)

    x32 = x_raw.to(tl.float32)
    y32 = y_raw.to(tl.float32)
    out32 = x32 + weight * (y32 - x32)

    out = out32.to(x_raw.dtype)
    tl.store(x_ptr + offsets, out, mask=mask)


@triton.jit
def _lerp_inplace_tensor_kernel(x_ptr, end_ptr, w_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_raw = tl.load(x_ptr + offsets, mask=mask, other=0)
    y_raw = tl.load(end_ptr + offsets, mask=mask, other=0)
    w_raw = tl.load(w_ptr + offsets, mask=mask, other=0)

    x32 = x_raw.to(tl.float32)
    y32 = y_raw.to(tl.float32)
    w32 = w_raw.to(tl.float32)

    out32 = x32 + w32 * (y32 - x32)
    out = out32.to(x_raw.dtype)
    tl.store(x_ptr + offsets, out, mask=mask)


def _ensure_contig_same_dtype_device(t: torch.Tensor, ref: torch.Tensor):
    t = t.to(device=ref.device, dtype=ref.dtype)
    if t.shape != ref.shape or not t.is_contiguous():
        t = t.expand_as(ref).contiguous()
    return t


def _parse_args_scalar(args, kwargs):
    # Accept either (self, end, weight) or args wrapped in a single tuple/list
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        args = tuple(args[0])
    self_t = kwargs.get('self', args[0] if len(args) > 0 else None)
    end = kwargs.get('end', args[1] if len(args) > 1 else None)
    weight = kwargs.get('weight', args[2] if len(args) > 2 else None)
    return self_t, end, weight


def _parse_args_tensor(args, kwargs):
    # Accept either (self, end, weight) or args wrapped in a single tuple/list
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        args = tuple(args[0])
    self_t = kwargs.get('self', args[0] if len(args) > 0 else None)
    end = kwargs.get('end', args[1] if len(args) > 1 else None)
    weight = kwargs.get('weight', args[2] if len(args) > 2 else None)
    return self_t, end, weight


def lerp__Scalar(*args, **kwargs):
    self_t, end, weight = _parse_args_scalar(args, kwargs)
    if not isinstance(self_t, torch.Tensor) or not isinstance(end, torch.Tensor):
        raise TypeError("lerp__Scalar expects tensors for 'self' and 'end'.")
    if not torch.is_floating_point(self_t):
        raise TypeError("lerp__Scalar only supports floating point tensors for in-place operation.")
    if not self_t.is_cuda or not end.is_cuda:
        raise ValueError("lerp__Scalar requires CUDA tensors.")
    if not self_t.is_contiguous():
        raise ValueError("lerp__Scalar requires 'self' to be contiguous for in-place operation.")

    # Handle scalar weight (number or 0-dim tensor)
    if isinstance(weight, torch.Tensor):
        if weight.numel() != 1:
            raise ValueError("Scalar overload received a non-scalar tensor for 'weight'.")
        weight_val = float(weight.item())
    else:
        weight_val = float(weight)

    end_c = _ensure_contig_same_dtype_device(end, self_t)

    n_elements = self_t.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _lerp_inplace_scalar_kernel[grid](
        self_t, end_c, n_elements, float(weight_val), BLOCK_SIZE=1024
    )
    return self_t


def lerp__Tensor(*args, **kwargs):
    self_t, end, weight = _parse_args_tensor(args, kwargs)
    if not isinstance(self_t, torch.Tensor) or not isinstance(end, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("lerp__Tensor expects tensors for 'self', 'end', and 'weight'.")
    if not torch.is_floating_point(self_t):
        raise TypeError("lerp__Tensor only supports floating point tensors for in-place operation.")
    if not self_t.is_cuda or not end.is_cuda or not weight.is_cuda:
        raise ValueError("lerp__Tensor requires CUDA tensors.")
    if not self_t.is_contiguous():
        raise ValueError("lerp__Tensor requires 'self' to be contiguous for in-place operation.")

    end_c = _ensure_contig_same_dtype_device(end, self_t)
    weight_c = _ensure_contig_same_dtype_device(weight, self_t)

    n_elements = self_t.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _lerp_inplace_tensor_kernel[grid](
        self_t, end_c, weight_c, n_elements, BLOCK_SIZE=1024
    )
    return self_t