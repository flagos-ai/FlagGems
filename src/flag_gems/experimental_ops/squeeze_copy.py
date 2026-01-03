import torch
import triton
import triton.language as tl
from typing import Iterable, List, Optional


@triton.jit
def _squeeze_copy_kernel(
    in_ptr,                      # data pointer: same dtype as input tensor
    out_ptr,                     # data pointer: same dtype as output tensor
    n_elements,                  # total number of elements in output
    index_strides_ptr,           # int64* of length OUT_NDIM: input strides mapped to output dims (in elements)
    out_shape_ptr,               # int64* of length OUT_NDIM: output sizes
    out_lin_strides_ptr,         # int64* of length OUT_NDIM: linearization strides of output dims
    BLOCK_SIZE: tl.constexpr,    # per-program elements
    OUT_NDIM: tl.constexpr       # number of output dimensions
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Compute input offsets (in elements) from linear output indices
    lin = offs.to(tl.int64)

    in_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    # Unroll across output dimensions
    for i in range(OUT_NDIM):
        s_lin = tl.load(out_lin_strides_ptr + i)  # scalar int64
        sz = tl.load(out_shape_ptr + i)           # scalar int64
        idx_i = (lin // s_lin) % sz               # [BLOCK_SIZE] int64
        in_stride_i = tl.load(index_strides_ptr + i)  # scalar int64
        in_offsets += idx_i * in_stride_i

    x = tl.load(in_ptr + in_offsets, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {dim - (ndim if dim < 0 else 0)})")
    return dim


def _normalize_dims(dims: Iterable[int], ndim: int) -> List[int]:
    normed: List[int] = []
    for d in dims:
        nd = _normalize_dim(int(d), ndim)
        normed.append(nd)
    # Keep original order but remove duplicates while preserving first occurrence
    seen = set()
    unique_ordered = []
    for d in normed:
        if d not in seen:
            seen.add(d)
            unique_ordered.append(d)
    return unique_ordered


def _compute_kept_dims_for_all(x: torch.Tensor) -> List[int]:
    sizes = x.shape
    return [i for i, s in enumerate(sizes) if s != 1]


def _compute_kept_dims_for_dim(x: torch.Tensor, dim: int) -> List[int]:
    ndim = x.dim()
    d = _normalize_dim(dim, ndim)
    # Remove only if size==1 on that dim
    if x.size(d) == 1:
        return [i for i in range(ndim) if i != d]
    else:
        return list(range(ndim))


def _compute_kept_dims_for_dims(x: torch.Tensor, dims: Iterable[int]) -> List[int]:
    ndim = x.dim()
    dims_norm = _normalize_dims(dims, ndim)
    remove_set = {d for d in dims_norm if x.size(d) == 1}
    return [i for i in range(ndim) if i not in remove_set]


def _prepare_indexing_params(x: torch.Tensor, kept_dims: List[int]):
    device = x.device
    dtype = x.dtype

    in_strides_elems = list(x.stride())  # in elements
    out_shape = [x.size(i) for i in kept_dims]
    out_ndim = len(out_shape)

    # Map output dims to input strides
    index_strides = [in_strides_elems[i] for i in kept_dims]

    # Compute linearization strides for output
    out_lin_strides: List[int] = []
    prod = 1
    for i in range(out_ndim - 1, -1, -1):
        out_lin_strides.insert(0, prod)
        prod *= out_shape[i]

    # Tensors for kernel (int64)
    if out_ndim == 0:
        # Provide dummy one-element tensors; kernel won't access them for OUT_NDIM=0
        index_strides_t = torch.empty(1, dtype=torch.int64, device=device)
        out_shape_t = torch.empty(1, dtype=torch.int64, device=device)
        out_lin_strides_t = torch.empty(1, dtype=torch.int64, device=device)
        n_elements = 1
    else:
        index_strides_t = torch.tensor(index_strides, dtype=torch.int64, device=device)
        out_shape_t = torch.tensor(out_shape, dtype=torch.int64, device=device)
        out_lin_strides_t = torch.tensor(out_lin_strides, dtype=torch.int64, device=device)
        n_elements = int(prod)

    return out_shape, out_ndim, index_strides_t, out_shape_t, out_lin_strides_t, n_elements, dtype, device


def _launch_squeeze_copy(x: torch.Tensor, kept_dims: List[int], out: Optional[torch.Tensor] = None) -> torch.Tensor:
    out_shape, out_ndim, index_strides_t, out_shape_t, out_lin_strides_t, n_elements, dtype, device = _prepare_indexing_params(x, kept_dims)

    if out is None:
        out = torch.empty(out_shape, device=device, dtype=dtype)
    else:
        if out.device != device:
            raise ValueError("Output tensor is on a different device than input.")
        if out.dtype != dtype:
            raise ValueError("Output tensor dtype must match input dtype.")
        if tuple(out.shape) != tuple(out_shape):
            raise ValueError(f"Output tensor has incorrect shape. Expected {tuple(out_shape)}, got {tuple(out.shape)}.")
        if not out.is_contiguous():
            raise ValueError("Output tensor must be contiguous.")

    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _squeeze_copy_kernel[grid](
        x, out, n_elements,
        index_strides_t, out_shape_t, out_lin_strides_t,
        BLOCK_SIZE=BLOCK_SIZE,
        OUT_NDIM=out_ndim,
    )
    return out


# Wrappers corresponding to ATen operator interfaces:

def squeeze_copy_out(self: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    kept_dims = _compute_kept_dims_for_all(self)
    return _launch_squeeze_copy(self, kept_dims, out=out)


def squeeze_copy_dim_out(self: torch.Tensor, dim: int, out: torch.Tensor) -> torch.Tensor:
    kept_dims = _compute_kept_dims_for_dim(self, dim)
    return _launch_squeeze_copy(self, kept_dims, out=out)


def squeeze_copy_dims_out(self: torch.Tensor, dims: Iterable[int], out: torch.Tensor) -> torch.Tensor:
    kept_dims = _compute_kept_dims_for_dims(self, dims)
    return _launch_squeeze_copy(self, kept_dims, out=out)


def squeeze_copy(self: torch.Tensor) -> torch.Tensor:
    kept_dims = _compute_kept_dims_for_all(self)
    return _launch_squeeze_copy(self, kept_dims, out=None)


def squeeze_copy_dim(self: torch.Tensor, dim: int) -> torch.Tensor:
    kept_dims = _compute_kept_dims_for_dim(self, dim)
    return _launch_squeeze_copy(self, kept_dims, out=None)


def squeeze_copy_dims(self: torch.Tensor, dims: Iterable[int]) -> torch.Tensor:
    kept_dims = _compute_kept_dims_for_dims(self, dims)
    return _launch_squeeze_copy(self, kept_dims, out=None)