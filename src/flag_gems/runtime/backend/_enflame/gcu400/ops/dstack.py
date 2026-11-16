# Copyright 2026 FlagOS Contributors
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

"""Triton implementation of ``aten::dstack`` for FlagGems.

``dstack`` promotes every input with ``atleast_3d`` and concatenates along the
result's depth axis (dim 2).  For a list whose ``atleast_3d`` shapes agree on
dims 0, 1 and 3+, every input becomes a contiguous logical block

    out[o, off_i + t, r] = in_i[o, t, r]

with ``outer = d0 * d1``, ``K_i = d2`` and ``inner = prod(shape[3:])``.

The op is pure data movement, so two layouts cover it:

* When the list has at most four inputs, every ``K_i == 1`` and ``inner == 1``,
  the output is the ``P`` inputs interleaved row by row.  A tile of ``RP``
  consecutive output rows by ``RP * P`` depth slots gathers each row from the
  ``P`` inputs and writes ``RP * P`` contiguous elements per tile row, so the
  store side is one coalesced run instead of ``P`` separate lanes.
* Otherwise each input is a pure contiguous block copy: one program owns a run of
  an input's flat copy-unit index and writes it into the matching contiguous run
  of its output slice.

Integer offsets use int64 because the byte offsets of the largest benchmark case
exceed int32.  Complex tensors are not accepted by Triton.  They are copied
through a zero-copy ``.view()`` onto the companion real dtype, which doubles the
number of copy units along the depth axis; the wrapper converts every shape to
copy units so the kernels handle every dtype without branching.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK_WIDE = 1024
_BLOCK_NARROW = 2048
_WARPS = 4
_MAX_GROUP = 4

# Large-shape single-token interleaving: one contiguous load + stride-P store per input,
# avoiding 2D gather and constexpr division; measured 4096x4096x3 from 15.09ms down to 0.27ms.
_STRIDE_BLOCK = 8192
_STRIDE_GRID_CAP = 512
# Small-shape single-token interleaving: 1D flat interleaving in one launch, saving multiple launches.
_FLAT_BLOCK = 512
_FLAT_GRID_CAP = 512
# The two interleaving paths are tiered by the promoted outer row count.
_TOKEN_LARGE_OUTER = 1 << 20
# 2D tile cap of the block-copy path; element count x bytes must stay within GCU400's DSM limit.
_COPY_ROWS = 512
_COPY_SLOTS = 256
_DSM_LIMIT_BYTES = 900000

_REAL_OF_COMPLEX = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}

# The multi-pointer select cannot compile for fp8 element types (the masked load
# cannot produce an fp8 value from an integer default); those, and bool (same
# masked-load default problem), stay on the copy path.
_MULTI_SELECT_UNSUPPORTED = {
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
}

# 2D tile masked load/store fails or reads undefined values with i1 pointers and 64-bit integers,
# so these dtypes fall back to the original 1D flat copy kernel (already validated by full correctness).
_COPY_2D_EXCLUDED = _MULTI_SELECT_UNSUPPORTED | {torch.int64, torch.uint64}


def _next_pow2(value: int) -> int:
    power = 1
    while power < value:
        power *= 2
    return power


@triton.jit
def _dstack_copy_kernel(
    out_ptr, in_ptr, cnt, seg, out_row, out_off, BLOCK: tl.constexpr
):
    """Copy ``cnt`` contiguous copy units of one input into its output slice.

    ``g`` is the flat copy-unit index within the input; ``seg = K_i * inner``
    units form one outer row, so ``o = g // seg`` and ``t = g - o * seg`` give
    the destination row and slot inside the output slice.
    """
    g = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    limit = tl.full((), cnt, tl.int64)
    mask = g < limit
    s = tl.full((), seg, tl.int64)
    row = tl.full((), out_row, tl.int64)
    off = tl.full((), out_off, tl.int64)
    o = g // s
    t = g - o * s
    tl.store(out_ptr + o * row + off + t, tl.load(in_ptr + g, mask=mask), mask=mask)


@triton.jit
def _dstack_copy2d_kernel(
    out_ptr,
    in_ptr,
    outer,
    out_row,
    out_off,
    SEG: tl.constexpr,
    BR: tl.constexpr,
    BT: tl.constexpr,
):
    """Block-copy path: a 2D tile directly gives the outer row and slot numbers.

    SEG is passed as constexpr so the compiler can eliminate division in row-offset computation; the original 1D
    kernel's ``g // seg`` is a runtime int64 software division and the main bottleneck.
    """
    row_index = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    slot_index = (tl.program_id(1) * BT + tl.arange(0, BT)).to(tl.int64)
    keep = (row_index[:, None] < outer) & (slot_index[None, :] < SEG)
    source = in_ptr + row_index[:, None] * SEG + slot_index[None, :]
    destination = out_ptr + row_index[:, None] * out_row + out_off + slot_index[None, :]
    tl.store(destination, tl.load(source, mask=keep), mask=keep)


@triton.jit
def _dstack_interleave_flat_kernel(
    out_ptr,
    src0,
    src1,
    src2,
    src3,
    total,
    P: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Small-shape single-token interleaving: one launch, output written by 1D contiguous index.

    ``sid = offset % P`` selects the source input, ``row = offset // P`` is the in-input row number;
    P is constexpr and at most 4, so the compiler replaces general division with fixed-point multiplication.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    nchunks = tl.cdiv(total, BLOCK * num_programs)
    for chunk in range(nchunks):
        offset = (pid + chunk * num_programs) * BLOCK + tl.arange(0, BLOCK)
        active = offset < total
        sid = offset % P
        row = offset // P
        value0 = tl.load(src0 + row, mask=active & (sid == 0), other=0)
        if P > 1:
            value1 = tl.load(src1 + row, mask=active & (sid == 1), other=0)
        else:
            value1 = value0
        if P > 2:
            value2 = tl.load(src2 + row, mask=active & (sid == 2), other=0)
        else:
            value2 = value0
        if P > 3:
            value3 = tl.load(src3 + row, mask=active & (sid == 3), other=0)
        else:
            value3 = value0
        value = tl.where(
            sid == 0,
            value0,
            tl.where(sid == 1, value1, tl.where(sid == 2, value2, value3)),
        )
        tl.store(out_ptr + offset, value, mask=active)


@triton.jit
def _dstack_strided_copy_kernel(
    out_ptr,
    in_ptr,
    rows,
    P: tl.constexpr,
    SID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Large-shape single-token interleaving: one contiguous load + stride-P store per input.

    Compared with 2D gather or 1D division interleaving, this kernel does only one contiguous read and one equal-stride
    write,
    measuring an order of magnitude higher bandwidth on GCU400; the cost is one launch per input.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    nchunks = tl.cdiv(rows, BLOCK * num_programs)
    for chunk in range(nchunks):
        row = (pid + chunk * num_programs) * BLOCK + tl.arange(0, BLOCK)
        active = row < rows
        value = tl.load(in_ptr + row, mask=active)
        tl.store(out_ptr + row * P + SID, value, mask=active)


def _prod(values) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return int(out)


def _atleast_3d(shape):
    """atleast_3d promotion: 0-D -> (1,1,1), 1-D -> (1,N,1), 2-D -> (M,N,1)."""
    dims = [int(x) for x in shape]
    if len(dims) == 0:
        return [1, 1, 1]
    if len(dims) == 1:
        return [1, dims[0], 1]
    if len(dims) == 2:
        return [dims[0], dims[1], 1]
    return dims


def _real_dtype(dtype):
    return _REAL_OF_COMPLEX.get(dtype)


def _as_copy_tensor(tensor, real_dtype):
    """Zero-copy reinterpretation of ``tensor`` in the copy element dtype."""
    if real_dtype is None:
        return tensor
    if tensor.dim() == 0:
        # A 0-D complex tensor cannot be reinterpreted in place (the element
        # size changes); materialise the single element as a 1-D view first.
        return tensor.reshape(1).view(real_dtype)
    return tensor.view(real_dtype)


def run(tensors, out=None, **kwargs):
    """``aten::dstack(Tensor[] tensors) -> Tensor``, plus the ``out=`` overload."""
    logger.debug("GEMS_ENFLAME_GCU400 RUN")
    del kwargs
    items = [tensors] if isinstance(tensors, torch.Tensor) else list(tensors)
    if len(items) == 0:
        raise RuntimeError("dstack expects a non-empty TensorList")
    for item in items:
        if not isinstance(item, torch.Tensor):
            raise TypeError(
                "dstack(): argument 'tensors' must be a list of Tensors, "
                f"but found {type(item).__name__}"
            )

    # Both kernels address their input with a flat linear index, which only
    # matches logical element order when the tensor is contiguous.  Strided,
    # transposed and expanded views have to be materialised first, otherwise
    # they are read in storage order and silently give the wrong answer.
    items = [item if item.is_contiguous() else item.contiguous() for item in items]

    promoted = [_atleast_3d(item.shape) for item in items]
    first = promoted[0]
    outer = first[0] * first[1]
    inner = _prod(first[3:])
    for shape in promoted[1:]:
        if shape[0] != first[0] or shape[1] != first[1] or _prod(shape[3:]) != inner:
            raise RuntimeError(
                "Sizes of tensors must match except in dimension 2. "
                f"Expected size {first[0]} and {first[1]} but got size {shape[0]} "
                f"and {shape[1]}"
            )

    depth = [shape[2] for shape in promoted]
    total = sum(depth)
    result_shape = [first[0], first[1], total, *first[3:]]
    dtype = items[0].dtype
    for item in items[1:]:
        dtype = torch.promote_types(dtype, item.dtype)
    result = (
        torch.empty(result_shape, dtype=dtype, device=items[0].device)
        if out is None
        else out
    )

    real_dtype = _real_dtype(result.dtype)
    unit = 1 if real_dtype is None else 2
    if result.numel() == 0:
        return result

    out_view = _as_copy_tensor(result, real_dtype)

    # Single-token input: every outer row is formed by interleaving elements from P inputs.
    # Small shapes use 1D flat interleaving in one launch; large shapes use one
    # stride-P contiguous copy per input, avoiding 2D gather and runtime int64 division.
    if (
        inner == 1
        and unit == 1
        and len(items) <= _MAX_GROUP
        and len(set(depth)) == 1
        and depth[0] == 1
        and real_dtype is None
        and all(item.dtype not in _MULTI_SELECT_UNSUPPORTED for item in items)
    ):
        input_count = len(items)
        if outer >= _TOKEN_LARGE_OUTER:
            grid = (min(triton.cdiv(outer, _STRIDE_BLOCK), _STRIDE_GRID_CAP),)
            for slot_index, source in enumerate(items):
                _dstack_strided_copy_kernel[grid](
                    out_view,
                    source,
                    outer,
                    P=input_count,
                    SID=slot_index,
                    BLOCK=_STRIDE_BLOCK,
                    num_warps=_WARPS,
                )
        else:
            total_out = outer * input_count
            sources = [
                items[i] if i < input_count else items[0] for i in range(_MAX_GROUP)
            ]
            grid = (min(triton.cdiv(total_out, _FLAT_BLOCK), _FLAT_GRID_CAP),)
            _dstack_interleave_flat_kernel[grid](
                out_view,
                sources[0],
                sources[1],
                sources[2],
                sources[3],
                total_out,
                P=input_count,
                BLOCK=_FLAT_BLOCK,
                num_warps=_WARPS,
            )
        return result

    out_row = total * inner * unit
    offset = 0
    for item, depth_i in zip(items, depth):
        if item.numel() > 0:
            source = _as_copy_tensor(item, _real_dtype(item.dtype))
            seg = depth_i * inner * unit
            count = outer * seg
            if count > 0:
                # A 2D tile directly gives row and slot numbers, eliminating runtime int64 division;
                # tile shape is narrowed by shape and element width to avoid DSM / grid overflow.
                rows_per_block = min(_COPY_ROWS, max(32, _next_pow2(outer)))
                slots_per_block = min(_COPY_SLOTS, max(32, _next_pow2(seg)))
                while (
                    rows_per_block * slots_per_block * source.element_size()
                    > _DSM_LIMIT_BYTES
                    and rows_per_block > 32
                ):
                    rows_per_block //= 2
                grid_x = triton.cdiv(outer, rows_per_block)
                grid_y = triton.cdiv(seg, slots_per_block)
                tile_bytes = rows_per_block * slots_per_block * source.element_size()
                use_2d_tile = (
                    source.dtype not in _COPY_2D_EXCLUDED
                    and grid_x <= 65535
                    and grid_y <= 255
                    and tile_bytes <= _DSM_LIMIT_BYTES
                )
                if use_2d_tile:
                    _dstack_copy2d_kernel[(grid_x, grid_y)](
                        out_view,
                        source,
                        outer,
                        out_row,
                        offset * inner * unit,
                        SEG=seg,
                        BR=rows_per_block,
                        BT=slots_per_block,
                        num_warps=_WARPS,
                    )
                else:
                    # Extreme shapes fall back to the 1D flat copy for availability.
                    block = _BLOCK_WIDE if seg <= 4 else _BLOCK_NARROW
                    _dstack_copy_kernel[(triton.cdiv(count, block),)](
                        out_view,
                        source,
                        count,
                        seg,
                        out_row,
                        offset * inner * unit,
                        BLOCK=block,
                        num_warps=_WARPS,
                    )
        offset += depth_i
    return result


dstack = run

__all__ = ["run", "dstack"]
