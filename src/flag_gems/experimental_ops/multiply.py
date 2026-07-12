from numbers import Number

import torch
import triton
import triton.language as tl


@triton.jit
def _multiply_tt_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * y, mask=mask)


@triton.jit
def _multiply_ts_kernel(x_ptr, scalar, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # scalar will be implicitly cast to x's dtype by Triton during multiplication
    tl.store(out_ptr + offsets, x * scalar, mask=mask)


def _broadcast_shape(a_shape, b_shape):
    return torch.broadcast_shapes(a_shape, b_shape)


def _result_dtype_for(a, b):
    if isinstance(b, torch.Tensor):
        return torch.result_type(a, b)
    else:
        # b is a Python scalar/Number
        return torch.result_type(a, torch.tensor(b))


def _ensure_cuda_device(t):
    if not (isinstance(t, torch.Tensor) and t.is_cuda):
        raise ValueError("Input tensors must be CUDA tensors for Triton kernels.")


def _launch_tt(a_ctg, b_ctg, out_t):
    n_elements = out_t.numel()
    if n_elements == 0:
        return
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _multiply_tt_kernel[grid](a_ctg, b_ctg, out_t, n_elements, BLOCK_SIZE=1024)


def _launch_ts(a_ctg, scalar, out_t):
    n_elements = out_t.numel()
    if n_elements == 0:
        return
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _multiply_ts_kernel[grid](a_ctg, scalar, out_t, n_elements, BLOCK_SIZE=1024)


def _multiply_impl(a, b, out=None):
    if not isinstance(a, torch.Tensor):
        raise TypeError("First argument must be a torch.Tensor")
    _ensure_cuda_device(a)
    device = a.device

    # Determine result dtype and broadcasted shape
    res_dtype = _result_dtype_for(a, b)

    if isinstance(b, torch.Tensor):
        _ensure_cuda_device(b)
        if b.device != device:
            raise ValueError("Both tensors must be on the same CUDA device.")
        out_shape = _broadcast_shape(a.shape, b.shape)
        a_ctg = a.to(res_dtype).expand(out_shape).contiguous()
        b_ctg = b.to(res_dtype).expand(out_shape).contiguous()
        if out is None:
            out_t = torch.empty(out_shape, device=device, dtype=res_dtype)
        else:
            if not isinstance(out, torch.Tensor) or not out.is_cuda:
                raise TypeError("out must be a CUDA torch.Tensor")
            if out.shape != out_shape:
                raise ValueError(
                    f"out shape {out.shape} does not match broadcasted shape {out_shape}"
                )
            if out.dtype != res_dtype:
                raise TypeError(
                    f"out dtype {out.dtype} does not match result dtype {res_dtype}"
                )
            if out.device != device:
                raise ValueError("out must be on the same CUDA device as inputs")
            out_t = out
        _launch_tt(a_ctg, b_ctg, out_t)
        return out_t
    elif isinstance(b, Number):
        # Scalar path
        out_shape = a.shape
        a_ctg = a.to(res_dtype).contiguous()
        if out is None:
            out_t = torch.empty(out_shape, device=device, dtype=res_dtype)
        else:
            if not isinstance(out, torch.Tensor) or not out.is_cuda:
                raise TypeError("out must be a CUDA torch.Tensor")
            if out.shape != out_shape:
                raise ValueError(
                    f"out shape {out.shape} does not match input tensor shape {out_shape}"
                )
            if out.dtype != res_dtype:
                raise TypeError(
                    f"out dtype {out.dtype} does not match result dtype {res_dtype}"
                )
            if out.device != device:
                raise ValueError("out must be on the same CUDA device as inputs")
            out_t = out
        _launch_ts(a_ctg, b, out_t)
        return out_t
    else:
        raise TypeError("Second argument must be a torch.Tensor or a Python scalar.")


def multiply_Tensor(self, other):
    return _multiply_impl(self, other, out=None)


def multiply_Scalar(self, other):
    return _multiply_impl(self, other, out=None)


def multiply_out(self, other, out):
    return _multiply_impl(self, other, out=out)
