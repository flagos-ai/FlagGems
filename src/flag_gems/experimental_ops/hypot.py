import torch
import triton
import triton.language as tl


def _torch_dtype_to_triton(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float64:
        return tl.float64
    raise ValueError(f"Unsupported dtype for Triton conversion: {dtype}")


@triton.jit
def _hypot_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)

    xf = x.to(COMPUTE_DTYPE)
    yf = y.to(COMPUTE_DTYPE)

    ax = tl.abs(xf)
    ay = tl.abs(yf)
    t = tl.maximum(ax, ay)
    m = tl.minimum(ax, ay)
    t_nz = tl.where(t > 0, t, 1).to(COMPUTE_DTYPE)
    r = m / t_nz
    res = tl.where(t > 0, t * tl.sqrt(1 + r * r), m)

    out_val = res.to(OUT_DTYPE)
    tl.store(out_ptr + offsets, out_val, mask=mask)


def _infer_hypot_out_dtype(a: torch.Tensor, b: torch.Tensor) -> torch.dtype:
    if a.is_complex() or b.is_complex():
        raise NotImplementedError(
            "Complex dtypes are not supported for hypot in this implementation."
        )
    if a.is_floating_point() or b.is_floating_point():
        return torch.result_type(a, b)
    # For integral/bool inputs, follow floating promotion behavior
    return torch.get_default_dtype()


def _launch_hypot_kernel(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor):
    assert x.device == y.device == out.device, "All tensors must be on the same device"
    assert out.is_cuda, "Triton kernels require CUDA tensors"
    n_elements = out.numel()
    if n_elements == 0:
        return

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    out_dtype = out.dtype
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"Unsupported output dtype for hypot: {out_dtype}")

    OUT_DTYPE = _torch_dtype_to_triton(out_dtype)
    COMPUTE_DTYPE = tl.float64 if out_dtype == torch.float64 else tl.float32

    _hypot_kernel[grid](
        x,
        y,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        OUT_DTYPE=OUT_DTYPE,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
    )


def hypot(a: torch.Tensor, b: torch.Tensor):
    # Determine output dtype and broadcasted shape
    out_dtype = _infer_hypot_out_dtype(a, b)
    device = a.device
    if b.device != device:
        raise ValueError("Input tensors must be on the same device")
    if device.type != "cuda":
        raise ValueError("This implementation requires CUDA tensors")

    out_shape = torch.broadcast_shapes(a.shape, b.shape)
    out = torch.empty(out_shape, dtype=out_dtype, device=device)

    # Prepare expanded, contiguous inputs
    x = a.expand(out_shape).contiguous()
    y = b.expand(out_shape).contiguous()

    _launch_hypot_kernel(x, y, out)
    return out


def hypot_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor):
    # Validate device and shape
    device = out.device
    if (not out.is_cuda) or a.device != device or b.device != device:
        raise ValueError(
            "All tensors (a, b, out) must be CUDA tensors on the same device"
        )

    # Validate dtype
    if out.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"Unsupported out dtype for hypot_out: {out.dtype}")

    # Validate/broadcast inputs to out shape
    target_shape = out.shape
    x = a.expand(target_shape).contiguous()
    y = b.expand(target_shape).contiguous()

    _launch_hypot_kernel(x, y, out)
    return out
