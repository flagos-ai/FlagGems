import torch
import triton
import triton.language as tl


@triton.jit
def _abs_kernel_real(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    # For both integer and floating types: abs = x if x >= 0 else -x
    y = tl.where(x >= 0, x, -x)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _abs_kernel_complex(rr_ptr, out_ptr, n_complex, BLOCK_SIZE: tl.constexpr):
    # rr_ptr points to interleaved real/imag scalars: [re0, im0, re1, im1, ...]
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # complex element indices
    mask = offsets < n_complex
    base = offsets * 2
    re = tl.load(rr_ptr + base, mask=mask)
    im = tl.load(rr_ptr + base + 1, mask=mask)
    mag = tl.sqrt(re * re + im * im)
    tl.store(out_ptr + offsets, mag, mask=mask)


def _ensure_cuda_tensor(x: torch.Tensor):
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if x.device.type != "cuda":
        raise ValueError("Tensor must be on CUDA device")
    return x


def _complex_abs_out_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    # Optional support if complex32 exists
    if hasattr(torch, "complex32") and dtype == getattr(torch, "complex32"):
        return torch.float16
    raise NotImplementedError(f"Unsupported complex dtype for abs: {dtype}")


def _launch_abs_real(inp: torch.Tensor, out: torch.Tensor):
    n_elements = out.numel()
    if n_elements == 0:
        return
    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _abs_kernel_real[grid](inp, out, n_elements, BLOCK_SIZE=BLOCK)


def _launch_abs_complex(inp: torch.Tensor, out: torch.Tensor):
    # inp is complex contiguous tensor, out is real contiguous with matching shape
    n_complex = inp.numel()
    if n_complex == 0:
        return
    # Create a real view of the interleaved storage
    if inp.dtype == torch.complex64:
        rr = inp.view(torch.float32)
    elif inp.dtype == torch.complex128:
        rr = inp.view(torch.float64)
    elif hasattr(torch, "complex32") and inp.dtype == getattr(torch, "complex32"):
        rr = inp.view(torch.float16)
    else:
        raise NotImplementedError(f"Unsupported complex dtype for abs: {inp.dtype}")
    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(n_complex, meta["BLOCK_SIZE"]),)
    _abs_kernel_complex[grid](rr, out, n_complex, BLOCK_SIZE=BLOCK)


def abs(x: torch.Tensor):
    x = _ensure_cuda_tensor(x)
    if x.is_complex():
        out_dtype = _complex_abs_out_dtype(x.dtype)
        out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        x_c = x.contiguous()
        out_c = out  # already contiguous
        _launch_abs_complex(x_c, out_c)
        return out
    else:
        out = torch.empty_like(x)
        x_c = x.contiguous()
        out_c = out  # contiguous
        _launch_abs_real(x_c, out_c)
        return out


def abs_out(x: torch.Tensor, out: torch.Tensor):
    x = _ensure_cuda_tensor(x)
    out = _ensure_cuda_tensor(out)
    if x.is_complex():
        expected_dtype = _complex_abs_out_dtype(x.dtype)
        if out.dtype != expected_dtype:
            raise TypeError(
                f"abs_out: expected out.dtype={expected_dtype}, got {out.dtype}"
            )
        if out.shape != x.shape:
            raise ValueError(f"abs_out: expected out.shape={x.shape}, got {out.shape}")
        x_c = x.contiguous()
        if out.is_contiguous():
            out_c = out
            _launch_abs_complex(x_c, out_c)
        else:
            tmp = torch.empty_like(out, memory_format=torch.contiguous_format)
            _launch_abs_complex(x_c, tmp)
            out.copy_(tmp)
        return out
    else:
        if out.dtype != x.dtype:
            raise TypeError(f"abs_out: expected out.dtype={x.dtype}, got {out.dtype}")
        if out.shape != x.shape:
            raise ValueError(f"abs_out: expected out.shape={x.shape}, got {out.shape}")
        x_c = x.contiguous()
        if out.is_contiguous():
            out_c = out
            _launch_abs_real(x_c, out_c)
        else:
            tmp = torch.empty_like(out, memory_format=torch.contiguous_format)
            _launch_abs_real(x_c, tmp)
            out.copy_(tmp)
        return out
