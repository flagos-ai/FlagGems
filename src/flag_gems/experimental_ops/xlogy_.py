import torch
import triton
import triton.language as tl


@triton.jit
def xlogy_inplace_tensor_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)

    logy = tl.log(y_f32)
    res = x_f32 * logy
    res = tl.where(x_f32 == 0.0, 0.0, res)

    tl.store(x_ptr + offsets, res.to(x.dtype), mask=mask)


@triton.jit
def xlogy_inplace_scalar_kernel(x_ptr, y_scalar, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)

    y_vec = tl.full((BLOCK_SIZE,), y_scalar, tl.float32)
    logy = tl.log(y_vec)

    res = x_f32 * logy
    res = tl.where(x_f32 == 0.0, 0.0, res)

    tl.store(x_ptr + offsets, res.to(x.dtype), mask=mask)


def _ensure_supported_dtype(t: torch.Tensor):
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"Unsupported dtype {t.dtype}. Supported: float16, bfloat16, float32."
        )


def _ensure_cuda_contiguous(t: torch.Tensor, name: str):
    if not t.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor.")
    if not t.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous.")


def xlogy__Tensor(*args, **kwargs):
    # Expecting signature: (self, other)
    if len(args) >= 2:
        x, other = args[0], args[1]
    else:
        x = kwargs.get("self", kwargs.get("input", None))
        other = kwargs.get("other", None)
    if x is None or other is None:
        raise ValueError("xlogy__Tensor expects (self, other) where both are tensors.")

    if not isinstance(other, torch.Tensor):
        raise TypeError(
            "xlogy__Tensor expects 'other' to be a Tensor. Use xlogy__Scalar_Other for scalar 'other'."
        )

    _ensure_cuda_contiguous(x, "self")
    _ensure_supported_dtype(x)
    _ensure_cuda_contiguous(other, "other")
    _ensure_supported_dtype(other)

    n_elements = x.numel()
    if other.numel() == 1:
        # Treat as scalar
        y_scalar = other.to(torch.float32).item()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        xlogy_inplace_scalar_kernel[grid](x, y_scalar, n_elements, BLOCK_SIZE=1024)
    else:
        if x.numel() != other.numel() or x.shape != other.shape:
            raise RuntimeError(
                "For xlogy__Tensor, 'other' must have the same shape as 'self' or be a scalar tensor."
            )
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        xlogy_inplace_tensor_kernel[grid](x, other, n_elements, BLOCK_SIZE=1024)

    return x


def xlogy__Scalar_Other(*args, **kwargs):
    # Expecting signature: (self, other_scalar)
    if len(args) >= 2:
        x, other = args[0], args[1]
    else:
        x = kwargs.get("self", kwargs.get("input", None))
        other = kwargs.get("other", None)
    if x is None:
        raise ValueError("xlogy__Scalar_Other expects 'self' tensor.")
    if other is None or isinstance(other, torch.Tensor):
        raise TypeError(
            "xlogy__Scalar_Other expects 'other' to be a Python scalar (not a Tensor)."
        )

    _ensure_cuda_contiguous(x, "self")
    _ensure_supported_dtype(x)

    # Convert scalar to float for kernel
    y_scalar = float(other)

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    xlogy_inplace_scalar_kernel[grid](x, y_scalar, n_elements, BLOCK_SIZE=1024)
    return x
