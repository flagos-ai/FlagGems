import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _scalar_fill_kernel(out_ptr, value, IS_BOOL: tl.constexpr):
    if IS_BOOL:
        tl.store(out_ptr, value != 0)
    else:
        tl.store(out_ptr, value.to(out_ptr.dtype.element_ty))


@triton.jit
def _scalar_fill_ptr(out_ptr, val_ptr, IS_BOOL: tl.constexpr):
    v = tl.load(val_ptr)
    if IS_BOOL:
        tl.store(out_ptr, v != 0)
    else:
        tl.store(out_ptr, v.to(out_ptr.dtype.element_ty))


def _resolve_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(s, dtype):
    if dtype is not None:
        return dtype
    if isinstance(s, bool):
        return torch.bool
    if isinstance(s, int):
        return torch.int64
    if isinstance(s, torch.Tensor):
        return s.dtype
    return torch.get_default_dtype()


def run(s, *, dtype=None, layout=None, device=None, pin_memory=None):
    dev = _resolve_device(device)
    out_dtype = _resolve_dtype(s, dtype)
    is_bool = out_dtype is torch.bool

    if isinstance(s, torch.Tensor):
        if s.numel() != 1:
            raise ValueError("scalar_tensor expects a scalar value")
        val = s.detach().reshape(())
        if val.device != dev:
            val = val.to(dev, non_blocking=True)
        out = torch.empty(
            (), dtype=out_dtype, layout=layout, device=dev, pin_memory=pin_memory
        )
        _scalar_fill_ptr[(1,)](out, val, IS_BOOL=is_bool, num_warps=1)
        return out

    if isinstance(s, np.generic):
        s = s.item()

    if isinstance(s, bool):
        value = 1.0 if s else 0.0
    elif isinstance(s, (int, float)):
        value = float(s)
    else:
        raise TypeError(f"unsupported scalar type for scalar_tensor: {type(s)!r}")

    out = torch.empty(
        (), dtype=out_dtype, layout=layout, device=dev, pin_memory=pin_memory
    )
    _scalar_fill_kernel[(1,)](out, value, IS_BOOL=is_bool, num_warps=2)
    return out


# Alias for FlagGems import convention
scalar_tensor = run
