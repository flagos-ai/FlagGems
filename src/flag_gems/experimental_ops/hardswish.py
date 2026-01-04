import torch
import triton
import triton.language as tl


@triton.jit
def hardswish_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    lower = x32 <= -3.0
    upper = x32 >= 3.0
    mid = (~lower) & (~upper)

    res32 = tl.zeros_like(x32)
    res32 = tl.where(upper, x32, res32)
    res32 = tl.where(mid, (x32 * (x32 + 3.0)) / 6.0, res32)
    # lower region already zero

    res = res32.to(x.dtype)
    tl.store(out_ptr + offsets, res, mask=mask)


def _launch_hardswish(x: torch.Tensor, out: torch.Tensor):
    n_elements = x.numel()
    if n_elements == 0:
        return
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    hardswish_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)


def _parse_input_tensor(*args, **kwargs) -> torch.Tensor:
    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
        return args[0]
    if 'self' in kwargs and isinstance(kwargs['self'], torch.Tensor):
        return kwargs['self']
    if 'input' in kwargs and isinstance(kwargs['input'], torch.Tensor):
        return kwargs['input']
    raise ValueError("Expected input tensor as the first positional argument or as 'self'/'input' keyword argument.")


def _parse_out_tensor(*args, **kwargs) -> torch.Tensor:
    if len(args) >= 2 and isinstance(args[1], torch.Tensor):
        return args[1]
    if 'out' in kwargs and isinstance(kwargs['out'], torch.Tensor):
        return kwargs['out']
    raise ValueError("Expected 'out' tensor as the second positional argument or as 'out' keyword argument.")


def _ensure_cuda_tensor(t: torch.Tensor, name: str):
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor (got device {t.device})")


_supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}


def _hardswish_impl(x: torch.Tensor, out: torch.Tensor = None):
    _ensure_cuda_tensor(x, "input")
    if out is not None:
        _ensure_cuda_tensor(out, "out")
        if out.shape != x.shape:
            raise ValueError(f"out shape {out.shape} must match input shape {x.shape}")

    x_co = x.contiguous()
    compute_dtype = x_co.dtype if x_co.dtype in _supported_dtypes else torch.float32
    x_work = x_co if x_co.dtype == compute_dtype else x_co.to(compute_dtype)

    if out is None:
        final_out = torch.empty_like(x)  # preserve layout/strides of input
    else:
        final_out = out

    can_write_direct = (
        final_out.is_contiguous()
        and final_out.device == x.device
        and final_out.dtype in _supported_dtypes
        and final_out.dtype == compute_dtype
    )

    if can_write_direct:
        out_work = final_out
        _launch_hardswish(x_work, out_work)
        return final_out
    else:
        out_work = torch.empty(x_work.shape, dtype=compute_dtype, device=x_work.device)
        _launch_hardswish(x_work, out_work)
        final_out.copy_(out_work.to(final_out.dtype))
        return final_out


def hardswish(*args, **kwargs):
    x = _parse_input_tensor(*args, **kwargs)
    return _hardswish_impl(x)


def hardswish_out(*args, **kwargs):
    x = _parse_input_tensor(*args, **kwargs)
    out = _parse_out_tensor(*args, **kwargs)
    _hardswish_impl(x, out)
    return out