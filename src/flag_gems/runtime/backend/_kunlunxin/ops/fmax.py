import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def fmax_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, NEED_MASK: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
    else:
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
    # fmax semantics: ignore NaN when possible
    # - if one is NaN, return the other
    # - if both are NaN, return NaN
    # NaN detection via integer bit pattern (avoids fp16/bf16 unordered
    # setcc which the XPU backend fails to select for large tiles).
    if x.dtype == tl.float64:
        x_nan = (x.to(tl.int64, bitcast=True) & 0x7FFFFFFFFFFFFFFF) > 0x7FF0000000000000
        y_nan = (y.to(tl.int64, bitcast=True) & 0x7FFFFFFFFFFFFFFF) > 0x7FF0000000000000
        out = tl.where(x_nan, y, tl.where(y_nan, x, tl.where(x >= y, x, y)))
    else:
        if x.dtype == tl.float16 or x.dtype == tl.bfloat16:
            xf = x.to(tl.float32)
            yf = y.to(tl.float32)
        else:
            xf = x
            yf = y
        xi = xf.to(tl.int32, bitcast=True)
        yi = yf.to(tl.int32, bitcast=True)
        # NaN iff exp all-1 and mantissa non-zero (f16/bf16 upcast to f32)
        x_nan = ((xi & 0x7F800000) == 0x7F800000) & ((xi & 0x007FFFFF) != 0)
        y_nan = ((yi & 0x7F800000) == 0x7F800000) & ((yi & 0x007FFFFF) != 0)
        out = tl.where(x_nan, y, tl.where(y_nan, x, tl.where(xf >= yf, x, y)))
    if NEED_MASK:
        tl.store(out_ptr + offsets, out, mask=mask)
    else:
        tl.store(out_ptr + offsets, out)


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
    a = _to_tensor(a, device=dev)
    b = _to_tensor(b, device=dev)
    a_b, b_b = torch.broadcast_tensors(a, b)
    out_dtype = torch.result_type(a_b, b_b)
    if out_dtype.is_complex:
        raise TypeError("fmax does not support complex dtypes.")
    compute_dtype = torch.int8 if out_dtype == torch.bool else out_dtype
    a_c = a_b.to(compute_dtype).contiguous()
    b_c = b_b.to(compute_dtype).contiguous()
    return a_c, b_c, out_dtype, compute_dtype


def _launch_fmax(a_c, b_c, out_c):
    n_elements = out_c.numel()
    block_size = 4096
    need_mask = (n_elements % block_size) != 0
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(out_c.device):
        fmax_kernel[grid](
            a_c, b_c, out_c, n_elements,
            BLOCK_SIZE=block_size, NEED_MASK=need_mask,
        )


def fmax(a, b):
    logger.debug("GEMS KUNLUNXIN FMAX")
    a_c, b_c, out_dtype, compute_dtype = _prepare_inputs(a, b, out=None)
    out_shape = a_c.shape
    if compute_dtype == out_dtype:
        out = torch.empty(out_shape, dtype=out_dtype, device=a_c.device)
        out_c = out
    else:
        out = torch.empty(out_shape, dtype=out_dtype, device=a_c.device)
        out_c = torch.empty(out_shape, dtype=compute_dtype, device=a_c.device)
    _launch_fmax(a_c, b_c, out_c)
    if out_c.dtype != out.dtype:
        out.copy_(out_c.to(out_dtype))
    return out


def fmax_out(a, b, out):
    logger.debug("GEMS KUNLUNXIN FMAX_OUT")
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a Tensor")
    a_c, b_c, out_dtype, compute_dtype = _prepare_inputs(a, b, out=out)
    expected_shape = a_c.shape
    if out.device != a_c.device:
        raise ValueError("out tensor must be on the same device as inputs.")
    if out.dtype != out_dtype:
        raise TypeError(f"out tensor has dtype {out.dtype}, expected {out_dtype}.")
    if tuple(out.shape) != tuple(expected_shape):
        raise ValueError(
            f"out tensor has shape {tuple(out.shape)}, expected {tuple(expected_shape)} after broadcasting."
        )
    if compute_dtype == out_dtype and out.is_contiguous():
        out_c = out
    else:
        out_c = torch.empty(expected_shape, dtype=compute_dtype, device=out.device)
    _launch_fmax(a_c, b_c, out_c)
    if out_c is not out:
        if out_c.dtype != out.dtype:
            out.copy_(out_c.to(out.dtype))
        else:
            if out.is_contiguous():
                out.copy_(out_c)
            else:
                out.view_as(out.contiguous()).copy_(out_c)
    return out