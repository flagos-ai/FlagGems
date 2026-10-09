"""Kunlunxin unfold_copy override.

unfold_copy(input, dimension, size, step) is the copy variant of
`input.unfold(dimension, size, step)`: it materializes the sliding-window
view into a fresh contiguous tensor.

Fast path (contiguous input): a single flat kernel over the output
decomposes each output linear index back into (outer, window, inner, k)
and gathers the matching input element. All test/benchmark shapes (2D
dim=1 and 3D dim=1/dim=2) are covered with one code path; measured 1.4x
- 2.6x over torch on KL3 for the benchmark matrix, vs 0.4x-1.1x for the
previous as_strided-view + generic strided-copy recipe (whose block-DMA
fast path never fires: the inner run is `size` <= 8 < 16 and the 4-D
generic path re-enters the slow pointwise-dynamic codegen).

Fallback (non-contiguous / exotic inputs): build the as_strided view
with Torch-compatible semantics/errors, then use the vendor as_strided
copy machinery (block-DMA fast paths + generic strided Triton kernels),
exactly as before.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._kunlunxin.ops.as_strided_copy import (
    _can_use_byte_triton,
    _can_use_triton,
    _launch_as_strided_copy,
    _launch_byte_as_strided_copy,
    _try_fast_copy,
)
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

_BLOCK = 1024


@libentry()
@triton.jit
def _unfold_copy_kernel(
    inp, out, outer, D, inner, L, size, step, numel, BLOCK: tl.constexpr
):
    pid = ext.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    offs = offs.to(tl.int64)
    k = offs % size
    t = offs // size
    c = t % inner
    t = t // inner
    w = t % L
    o = t // L
    in_off = (o * D + w * step + k) * inner + c
    vals = tl.load(inp + in_off, mask=mask, other=0)
    tl.store(out + offs, vals, mask=mask)


def _make_unfold_view(input, dimension, size, step):
    """Build the unfold view with Torch-compatible semantics/errors."""
    ndim = input.ndim
    if ndim == 0:
        dim_size = 1
        dimension = 0
    else:
        orig_dim = dimension
        if dimension < 0:
            dimension += ndim
        if dimension < 0 or dimension >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-ndim}, {ndim - 1}], but got {orig_dim})"
            )
        dim_size = input.shape[dimension]

    if size > dim_size:
        raise RuntimeError(
            f"maximum size for tensor at dimension {dimension} is {dim_size} "
            f"but size is {size}"
        )

    n_windows = (dim_size - size) // step + 1

    if ndim == 0:
        new_shape = [n_windows, size]
        new_strides = [step, 1]
    else:
        new_shape = list(input.shape)
        new_shape[dimension] = n_windows
        new_shape.append(size)
        old_strides = list(input.stride())
        dim_stride = old_strides[dimension]
        new_strides = list(old_strides)
        new_strides[dimension] = step * dim_stride
        new_strides.append(dim_stride)

    return input.as_strided(new_shape, new_strides, input.storage_offset())


def unfold_copy(input, dimension, size, step):
    logger.debug("GEMS_KUNLUNXIN UNFOLD_COPY")
    size = int(size)
    step = int(step)

    ndim = input.ndim
    if ndim == 0:
        dim = 0
        dim_size = 1
        if dimension not in (0, -1):
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[-1, 0], but got {dimension})"
            )
        if size > 1:
            raise RuntimeError(
                f"maximum size for tensor at dimension 0 is 1 but size is {size}"
            )
    else:
        orig_dim = dimension
        dim = dimension
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-ndim}, {ndim - 1}], but got {orig_dim})"
            )
        dim_size = input.shape[dim]

        if size > dim_size:
            raise RuntimeError(
                f"maximum size for tensor at dimension {dim} is {dim_size} "
                f"but size is {size}"
            )

    if step <= 0:
        raise RuntimeError(f"step is {step} but must be > 0")

    n_windows = (dim_size - size) // step + 1
    if ndim == 0:
        out_shape = (n_windows * size,)
        outer = 1
        inner = 1
    else:
        out_shape = input.shape[:dim] + (n_windows,) + input.shape[dim + 1 :] + (size,)
        outer = math.prod(input.shape[:dim])
        inner = math.prod(input.shape[dim + 1 :])
    out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    numel = out.numel()
    if numel == 0:
        return out

    if input.is_contiguous():
        grid = (triton.cdiv(numel, _BLOCK),)
        _unfold_copy_kernel[grid](
            input,
            out,
            outer,
            dim_size,
            inner,
            n_windows,
            size,
            step,
            numel,
            BLOCK=_BLOCK,
        )
        return out

    view = _make_unfold_view(input, dimension, size, step)
    if _try_fast_copy(view, out):
        return out
    if _can_use_triton(view, out):
        return _launch_as_strided_copy(view, out)
    if _can_use_byte_triton(view, out):
        return _launch_byte_as_strided_copy(view, out)
    raise NotImplementedError(
        "Kunlunxin unfold_copy does not support this stride layout."
    )
