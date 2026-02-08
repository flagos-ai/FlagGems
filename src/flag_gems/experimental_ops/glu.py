import math  # noqa: F401

import torch
import triton
import triton.language as tl


@triton.jit
def glu_kernel(
    x_ptr,  # *Pointer* to input tensor data (flattened, contiguous).
    y_ptr,  # *Pointer* to output tensor data (flattened, contiguous).
    n_out_elements,  # Number of elements in the output tensor.
    inner_size,  # Product of sizes of dims after 'dim' in output shape.
    half_size,  # Size along 'dim' in output shape (i.e., original dim size // 2).
    outer_elems,  # Number of elements per 'outer' slice in the input: (2*half_size)*inner_size.
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_out_elements

    idx = offsets
    s = half_size
    inner = inner_size
    outer_inc = outer_elems

    # Map each output index to the corresponding input indices.
    # For contiguous tensors:
    # - output shape: [..., s, ...]; n_out = outer * s * inner
    # - input shape:  [..., 2*s, ...]
    # Let:
    #   o = idx // (s * inner)
    #   r = idx %  (s * inner)
    #   d = r // inner
    #   i = r % inner
    # Then:
    #   x_left_index  = o * (2*s*inner) + d * inner + i
    #   x_right_index = x_left_index + s * inner
    denom = s * inner
    o = idx // denom
    r = idx % denom
    d = r // inner
    i = r % inner

    x_index = o * outer_inc + d * inner + i
    gate_index = x_index + s * inner

    x_val = tl.load(x_ptr + x_index, mask=mask, other=0.0)
    g_val = tl.load(x_ptr + gate_index, mask=mask, other=0.0)

    x_f = x_val.to(tl.float32)
    g_f = g_val.to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-g_f))
    y = x_f * gate
    y_cast = y.to(x_val.dtype)

    tl.store(y_ptr + idx, y_cast, mask=mask)


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if not (0 <= dim < ndim):
        actual_dim = dim - ndim if dim >= ndim else dim
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {actual_dim})"
        )
    return dim


def _check_dtype_supported(dtype: torch.dtype):
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"Unsupported dtype {dtype}. Supported dtypes are: float16, bfloat16, float32."
        )


def _glu_launch(x: torch.Tensor, dim: int, out: torch.Tensor = None) -> torch.Tensor:
    if not x.is_cuda:
        raise AssertionError("Input tensor must be on CUDA device.")
    x = x.contiguous()
    _check_dtype_supported(x.dtype)

    ndim = x.dim()
    dim = _normalize_dim(dim, ndim)
    size_dim = x.size(dim)
    if size_dim % 2 != 0:
        raise RuntimeError(
            f"glu: dimension {dim} size must be even, but got {size_dim}."
        )

    half = size_dim // 2

    # Compute output shape
    out_shape = list(x.shape)
    out_shape[dim] = half

    # Prepare output
    if out is None:
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    else:
        if not out.is_cuda:
            raise AssertionError("Output tensor must be on CUDA device.")
        if tuple(out.shape) != tuple(out_shape):
            raise RuntimeError(
                f"glu_out: provided out has wrong shape. Expected {tuple(out_shape)}, got {tuple(out.shape)}."
            )
        if out.dtype != x.dtype:
            raise RuntimeError(
                f"glu_out: dtype mismatch. out.dtype={out.dtype}, expected {x.dtype}."
            )
        if not out.is_contiguous():
            raise RuntimeError("glu_out: output tensor must be contiguous.")
    out = out.contiguous()

    # Compute mapping parameters for contiguous layout
    # inner_size = product of dimensions after 'dim' in the output shape
    inner_size = 1
    for k in range(dim + 1, ndim):
        inner_size *= out_shape[k]

    n_out = out.numel()
    outer_elems = (2 * half) * inner_size  # elements per 'outer' slice in input

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_out, meta["BLOCK_SIZE"]),)  # noqa: E731

    glu_kernel[grid](
        x,
        out,
        n_out,
        inner_size,
        half,
        outer_elems,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return _glu_launch(input, dim, out=None)


def glu_out(
    input: torch.Tensor, dim: int = -1, out: torch.Tensor = None
) -> torch.Tensor:
    if out is None:
        raise RuntimeError("glu_out: 'out' tensor must be provided.")
    return _glu_launch(input, dim, out=out)
