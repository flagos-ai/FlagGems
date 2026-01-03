import torch
import triton
import triton.language as tl


def _tl_dtype_from_torch(dtype: torch.dtype):
    # Map common torch dtypes to Triton dtypes
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float64:
        return tl.float64
    if dtype == torch.int8:
        return tl.int8
    if dtype == torch.int16:
        return tl.int16
    if dtype == torch.int32:
        return tl.int32
    if dtype == torch.int64:
        return tl.int64
    if dtype == torch.uint8:
        return tl.uint8
    raise NotImplementedError(f"Unsupported dtype for Triton copy_: {dtype}")


@triton.jit
def _copy_kernel(dst_ptr, src_ptr, n_elements, BLOCK_SIZE: tl.constexpr, DST_DTYPE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    vals = tl.cast(vals, DST_DTYPE)
    tl.store(dst_ptr + offsets, vals, mask=mask)


@triton.jit
def _fill_kernel(dst_ptr, scalar_value, n_elements, BLOCK_SIZE: tl.constexpr, DST_DTYPE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.full((BLOCK_SIZE,), tl.cast(scalar_value, DST_DTYPE), DST_DTYPE)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def _launch_copy_tensor(dst: torch.Tensor, src: torch.Tensor):
    assert dst.is_cuda and src.is_cuda, "Triton copy_ supports CUDA tensors only."
    assert dst.is_contiguous() and src.is_contiguous(), "Only contiguous tensors are supported."
    n_elements = dst.numel()
    assert src.numel() == n_elements, "Source and destination must have the same number of elements."
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    DST_DTYPE = _tl_dtype_from_torch(dst.dtype)
    _copy_kernel[grid](
        dst, src, n_elements,
        BLOCK_SIZE=1024,
        DST_DTYPE=DST_DTYPE,
    )
    return dst


def _launch_fill_scalar(dst: torch.Tensor, scalar):
    assert dst.is_cuda, "Triton copy_ (scalar) supports CUDA tensors only."
    assert dst.is_contiguous(), "Only contiguous tensors are supported."
    n_elements = dst.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    DST_DTYPE = _tl_dtype_from_torch(dst.dtype)
    # Convert scalar to a Python number for kernel argument
    if dst.dtype.is_floating_point:
        scalar_val = float(scalar)
    else:
        scalar_val = int(scalar)
    _fill_kernel[grid](
        dst, scalar_val, n_elements,
        BLOCK_SIZE=1024,
        DST_DTYPE=DST_DTYPE,
    )
    return dst


def copy_(self: torch.Tensor, src, non_blocking: bool = False):
    if isinstance(src, torch.Tensor):
        return _launch_copy_tensor(self, src)
    elif isinstance(src, (int, bool)):
        return _launch_fill_scalar(self, int(src))
    elif isinstance(src, float):
        return _launch_fill_scalar(self, float(src))
    else:
        raise TypeError(f"Unsupported src type for copy_: {type(src)}")


def copy__Tensor(self: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
    return _launch_copy_tensor(self, src)


def copy__int(self: torch.Tensor, src: int, non_blocking: bool = False):
    return _launch_fill_scalar(self, int(src))


def copy__float(self: torch.Tensor, src: float, non_blocking: bool = False):
    return _launch_fill_scalar(self, float(src))