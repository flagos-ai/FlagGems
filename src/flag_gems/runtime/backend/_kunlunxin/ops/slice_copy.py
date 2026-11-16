# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kunlunxin (XPU3) override for ``aten::slice_copy`` / ``aten::slice_copy.out``.

The generic kernel is a one-element-per-lane gather: every output element
decomposes the flat index into ``(outer, slice_pos, inner)`` with 1-3 runtime
``int64`` integer divisions and then forms a lane-varying address.  On XPU3 the
soft-divide + masked-memory path costs ~0.01-0.8x versus ATen for the large
``dim == 0`` cases (a (1024, 65536) dim-0 slice takes 4.3 ms vs 0.038 ms).

A slice only touches one axis, so for contiguous inputs every output "row"
(the trailing axes after ``dim``) is itself a **contiguous run** of the input:

    src_offset(row, i) = start*inner
                       + (row // slice_len) * (dim_size*inner)
                       + (row %  slice_len) * (step*inner)
                       + i,          i in [0, inner)

When ``inner`` is wide enough this becomes a plain 2-D block DMA (one program
per ``(row, chunk)``) with **scalar** bases, exactly the access shape XPU3 likes.
Narrow ``inner`` (trailing-dim / column slices) stay a bounded-rank affine
gather, and non-contiguous inputs use the same gather so any stride/step works.

Addresses in the run kernel are clamped to ``inner - 1`` for masked-off lanes,
so no lane ever forms an address outside its source run.  Only allocation
(``torch.empty``/``empty_strided``) and metadata views are used; the movement is
a real XPU Triton kernel (no ``aten._copy_from``).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.slice_copy import (
    _BYTEWISE_DTYPES,
    _dtype_name,
    _resolve_slice_bounds,
    _slice_view,
)

from .flatten_dense_tensors import _TL_DTYPE, _copy_strided

logger = logging.getLogger(__name__)

# Below this run width the 2-D run kernel is launch-bound and a flat/affine
# gather is at least as good.
_MIN_RUN = 64
# Small slices are launch-bound; the plain affine gather has less fixed cost
# than the 2-D run kernel for a handful of short rows.
_RUN_MIN_NUMEL = 1 << 17


@triton.jit
def _slice_run_kernel(
    src_ptr,
    dst_ptr,
    start,
    dim_size,
    slice_len,
    inner,
    step,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Copy one contiguous run (a row of the slice) per program chunk.

    ``row`` indexes the flattened ``(outer, slice_pos)`` space of the output.
    The source base is a scalar (computed once per program); only the trailing
    ``inner`` axis is lane-varying, so the load/store are contiguous block DMA.
    """
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)

    outer = row // slice_len
    pos = row - outer * slice_len
    src_base = (
        start * inner + outer * (dim_size * inner) + pos * (step * inner)
    ).to(tl.int64)

    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    # Clamp the address so even masked-off lanes stay inside their run.
    safe = tl.minimum(offs, inner - 1)
    dst_base = row * inner
    pp = src_ptr.to(tl.pointer_type(DT))
    op = dst_ptr.to(tl.pointer_type(DT))
    if NEED_MASK:
        mask = offs < inner
        x = tl.load(pp + src_base + safe, mask=mask)
        tl.store(op + dst_base + safe, x, mask=mask)
    else:
        x = tl.load(pp + src_base + offs)
        tl.store(op + dst_base + offs, x)


def _run_launch(src, dst_flat, start, dim_size, slice_len, inner, step, dt):
    rows = (dst_flat.numel() // inner) if inner else 0
    block = min(triton.next_power_of_2(inner), 8192)
    block = max(block, 256)
    grid = (rows, triton.cdiv(inner, block))
    _slice_run_kernel[grid](
        src,
        dst_flat,
        start,
        dim_size,
        slice_len,
        inner,
        step,
        BLOCK=block,
        DT=dt,
        NEED_MASK=(inner % block != 0),
        num_warps=8,
    )


def _gather_launch(src_view, dst_flat, dt):
    _copy_strided(dst_flat, src_view, 0, dt)


def _shape_at(inp, dim, slice_len):
    shape = list(inp.shape)
    shape[dim] = slice_len
    return shape


def _slice_copy_impl(inp, dim, start, end, step, out):
    ndim = inp.ndim
    if ndim == 0:
        raise IndexError("slice() cannot be applied to a 0-dim tensor.")
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})"
        )
    if step <= 0:
        raise RuntimeError("slice step must be positive")

    dim = dim % ndim
    dim_size = inp.size(dim)
    start, end = _resolve_slice_bounds(dim_size, start, end)
    slice_len = max(0, (end - start + step - 1) // step)
    target_shape = _shape_at(inp, dim, slice_len)

    if out is None:
        out = torch.empty(target_shape, dtype=inp.dtype, device=inp.device)
        direct = True
    else:
        # ATen resizes before reporting dtype/device mismatches.
        if list(out.shape) != target_shape:
            out.resize_(target_shape)
        if out.dtype != inp.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {_dtype_name(inp.dtype)}, "
                f"but got {_dtype_name(out.dtype)} instead"
            )
        if out.device != inp.device:
            raise RuntimeError(
                f"Expected out tensor to have device {inp.device}, "
                f"but got {out.device} instead"
            )
        direct = out.is_contiguous()

    if out.numel() == 0:
        return out

    # Lazy conjugate/negative views carry un-resolved bytes: materialise them so
    # the copy yields the same resolved values as ATen.
    work = inp.resolve_conj().resolve_neg()

    inner = 1
    for d in range(dim + 1, work.ndim):
        inner *= work.size(d)

    dst = out if direct else torch.empty_like(out, memory_format=torch.contiguous_format)

    if work.dtype in _BYTEWISE_DTYPES:
        # bool / float8 have no Triton scalar type here: bit-copy through uint8.
        u8 = work.contiguous().view(torch.uint8)
        view = _slice_view(u8, dim, start, slice_len, step)
        _gather_launch(view, dst.view(torch.uint8).view(-1), _TL_DTYPE[torch.uint8])
    elif work.is_complex():
        # Triton has no complex scalar type: copy the real view (inner doubles).
        real = torch.view_as_real(work)
        dst_real = torch.view_as_real(dst)
        dt = _TL_DTYPE[real.dtype]
        if work.is_contiguous() and inner * 2 >= _MIN_RUN:
            _run_launch(
                real,
                dst_real.view(-1),
                start,
                dim_size,
                slice_len,
                inner * 2,
                step,
                dt,
            )
        else:
            view = _slice_view(real, dim, start, slice_len, step)
            _gather_launch(view, dst_real.view(-1), dt)
    else:
        dt = _TL_DTYPE.get(work.dtype)
        if dt is None:
            raise RuntimeError(f"slice_copy: unsupported dtype {work.dtype}")
        if (
            work.is_contiguous()
            and inner >= _MIN_RUN
            and dst.numel() >= _RUN_MIN_NUMEL
        ):
            _run_launch(work, dst.view(-1), start, dim_size, slice_len, inner, step, dt)
        else:
            view = _slice_view(work, dim, start, slice_len, step)
            _gather_launch(view, dst.view(-1), dt)

    if not direct:
        out.copy_(dst.view(out.shape))
    return out


def slice_copy(self, dim=0, start=None, end=None, step=1):
    logger.debug("GEMS_KUNLUNXIN SLICE_COPY")
    return _slice_copy_impl(self, dim, start, end, step, None)


def slice_copy_out(self, dim=0, start=None, end=None, step=1, *, out):
    logger.debug("GEMS_KUNLUNXIN SLICE_COPY_OUT")
    return _slice_copy_impl(self, dim, start, end, step, out)
