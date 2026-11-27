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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_MAX_TRITON_ELEMENTS = torch.iinfo(torch.int32).max
# Grid-stride tile sizes: tuned on gcu400 (do_bench median, cold-L2).  The
# tl.range loop costs ~0.4us per iteration, so tiles are sized to keep every
# program at ~1-3 iterations while staying within one 48-program wave.
_BLOCK_SIZE = 8192
_STRIDED_BLOCK = 4096
_DIVMOD_BLOCK = 2048
_DIVMOD3D_BLOCK = 4096
# GCU400 (Libra) schedules one wave of 48 programs per kernel; a larger grid
# pays the full wave cost again per batch of 48 programs (measured: a no-work
# kernel at grid=1024 costs 52us vs 0.43us at grid=1). All kernels below use a
# grid-stride loop with this cap instead of one program per tile.
_GRID_CAP = 48
# 2D tile shapes: word-widened wide-row tiles for row-major unit-stride views
# (BLOCK_N <= 256 words; measured fastest at [32, 256] on a 512x512 row slice),
# square tiles otherwise (transpose-like views measured fastest at 64x64).
_TILE_ELEMS = 8192
# Flat copies move int32 words when the byte size allows it: 4 bytes per lane
# instead of 1 (bool/int8) or 2 (fp16/bf16/int16), and int64 tensors never
# materialize int64 lanes in the kernel.
_FLAT_BLOCK_MAX = 8192
_MAX_FLAT_ELEMS = _MAX_TRITON_ELEMENTS - _FLAT_BLOCK_MAX
# i32 indexing headroom: offsets reach (num_tiles - 1) * BLOCK + BLOCK.
_MAX_I32_OFFSET = _MAX_TRITON_ELEMENTS - 16384


@triton.jit
def _as_strided_copy_flat_kernel(
    src,
    dst,
    n_elements,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(n_elements, BLOCK)
    for tile_id in tl.range(pid, num_tiles, GRID):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        values = tl.load(src + offsets, mask=mask, other=0)
        tl.store(dst + offsets, values, mask=mask)


@triton.jit
def _as_strided_copy_1d_kernel(
    input,
    out,
    input_stride_0,
    out_stride_0,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID: tl.constexpr,
    WIDE_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in tl.range(pid, num_tiles, GRID):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        if WIDE_INDEX:
            offsets = offsets.to(tl.int64)
        values = tl.load(input + offsets * input_stride_0, mask=mask)
        tl.store(out + offsets * out_stride_0, values, mask=mask)


@triton.jit
def _as_strided_copy_2d_kernel(
    input,
    out,
    input_stride_0,
    input_stride_1,
    out_stride_0,
    out_stride_1,
    dim_0,
    dim_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_m = offsets_m.to(tl.int64)[:, None]
    offsets_n = offsets_n.to(tl.int64)[None, :]
    mask = (offsets_m < dim_0) & (offsets_n < dim_1)
    input_offsets = offsets_m * input_stride_0 + offsets_n * input_stride_1
    out_offsets = offsets_m * out_stride_0 + offsets_n * out_stride_1
    values = tl.load(input + input_offsets, mask=mask)
    tl.store(out + out_offsets, values, mask=mask)


@triton.jit
def _as_strided_copy_2d_divmod_kernel(
    input,
    out,
    input_stride_0,
    input_stride_1,
    out_stride_0,
    out_stride_1,
    dim_1,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID: tl.constexpr,
    WIDE_INDEX: tl.constexpr,
):
    # Flat 1D-style decomposition of the 2D index space.  The 2D-affine tile
    # kernel above mis-lowers gathers without a unit-stride axis on gcu400
    # (silent wrong values) and aborts on zero-stride views; the div/mod form
    # is exact for every stride pattern (verified against ATen on a stride
    # matrix covering both-strides->2, unit-stride, non-contiguous out).
    # i32 indexing: i64 div/mod is 2.5-4x slower; WIDE_INDEX is only used for
    # tensors whose byte reach exceeds int32.
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in tl.range(pid, num_tiles, GRID):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        if WIDE_INDEX:
            offsets = offsets.to(tl.int64)
        index_1 = offsets % dim_1
        index_0 = offsets // dim_1
        input_offsets = index_0 * input_stride_0 + index_1 * input_stride_1
        out_offsets = index_0 * out_stride_0 + index_1 * out_stride_1
        values = tl.load(input + input_offsets, mask=mask)
        tl.store(out + out_offsets, values, mask=mask)


@triton.jit
def _as_strided_copy_3d_kernel(
    input,
    out,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    dim_1,
    dim_2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID: tl.constexpr,
    WIDE_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in tl.range(pid, num_tiles, GRID):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        if WIDE_INDEX:
            offsets = offsets.to(tl.int64)
        index_2 = offsets % dim_2
        tmp = offsets // dim_2
        index_1 = tmp % dim_1
        index_0 = tmp // dim_1
        input_offsets = (
            index_0 * input_stride_0
            + index_1 * input_stride_1
            + index_2 * input_stride_2
        )
        out_offsets = (
            index_0 * out_stride_0 + index_1 * out_stride_1 + index_2 * out_stride_2
        )
        values = tl.load(input + input_offsets, mask=mask)
        tl.store(out + out_offsets, values, mask=mask)


def _is_float8(dtype: torch.dtype) -> bool:
    return str(dtype).startswith("torch.float8_")


def _has_lazy_metadata(tensor: torch.Tensor) -> bool:
    is_neg = getattr(tensor, "is_neg", lambda: False)
    return tensor.is_conj() or is_neg()


def _make_as_strided_view(
    input: torch.Tensor,
    size,
    stride,
    storage_offset,
) -> torch.Tensor:
    # Reuse PyTorch's view construction to match its validation and None-offset semantics.
    if storage_offset is None:
        return torch.as_strided(input, size, stride)
    return torch.as_strided(input, size, stride, storage_offset)


def _native_copy_(out: torch.Tensor, src: torch.Tensor):
    return torch.ops.aten.copy_.default.redispatch(_FALLBACK_KEYSET, out, src, False)


def _fallback_as_strided_copy(input, size, stride, storage_offset=None):
    view = _make_as_strided_view(input, size, stride, storage_offset)
    out = torch.empty(tuple(size), dtype=input.dtype, device=input.device)
    if out.numel() != 0:
        # Call native copy_ directly so unsupported CUDA dtypes do not re-enter
        # FlagGems copy kernels through the composite as_strided_copy fallback.
        _native_copy_(out, view)
    return out


def _fallback_as_strided_copy_out(input, size, stride, storage_offset=None, *, out):
    view = _make_as_strided_view(input, size, stride, storage_offset)
    if (
        torch._C._is_alias_of(input, out)
        or has_internal_overlapping(out) != MemOverlap.No
    ):
        temp = torch.empty(tuple(size), dtype=input.dtype, device=input.device)
        if temp.numel() != 0:
            _native_copy_(temp, view)
        view = temp
    _native_copy_(out, view)
    return out


def _can_use_triton(input: torch.Tensor, out: torch.Tensor) -> bool:
    if input.layout != torch.strided or out.layout != torch.strided:
        return False
    if input.device != out.device or input.dtype != out.dtype:
        return False
    if input.is_quantized or out.is_quantized:
        return False
    if input.is_complex() or _is_float8(input.dtype):
        return False
    if out.numel() > _MAX_TRITON_ELEMENTS:
        return False
    return True


def _can_use_byte_triton(input: torch.Tensor, out: torch.Tensor) -> bool:
    if input.layout != torch.strided or out.layout != torch.strided:
        return False
    if input.device != out.device or input.dtype != out.dtype:
        return False
    if not _is_float8(input.dtype):
        return False
    if input.element_size() != 1 or out.element_size() != 1:
        return False
    if _has_lazy_metadata(input) or _has_lazy_metadata(out):
        return False
    if out.numel() > _MAX_TRITON_ELEMENTS:
        return False
    return True


def _launch_flat_copy(src, dst, n_elements):
    if n_elements <= _FLAT_BLOCK_MAX:
        block = max(1, triton.next_power_of_2(n_elements))
        grid = (1,)
    else:
        block = _FLAT_BLOCK_MAX
        grid = (min(_GRID_CAP, triton.cdiv(n_elements, block)),)
    _as_strided_copy_flat_kernel[grid](
        src,
        dst,
        n_elements,
        BLOCK=block,
        GRID=_GRID_CAP,
        num_warps=2 if block <= 2048 else 4,
    )


def _try_flat_as_strided_copy(view: torch.Tensor, out: torch.Tensor) -> bool:
    """Whole-tensor flat copy for contiguous view/out pairs.

    Plain offset loads/stores only: unlike the DTE (block-pointer) channel the
    pointwise kernels use, offset traffic cannot overrun the allocation on the
    last (masked) tile, which on gcu400 faults the MMU when the allocation ends
    on a page boundary.  Data is reinterpreted as int32 words whenever the byte
    size and offsets allow it, so every dtype moves 4 bytes per lane.
    """
    if not (view.is_contiguous() and out.is_contiguous()):
        return False
    esize = view.element_size()
    numel = view.numel()
    if (numel * esize) % 4 == 0 and numel * esize // 4 <= _MAX_FLAT_ELEMS:
        if (view.storage_offset() * esize) % 4 == 0 and (
            out.storage_offset() * esize
        ) % 4 == 0:
            try:
                src = view.reshape(-1).view(torch.int32)
                dst = out.reshape(-1).view(torch.int32)
            except (RuntimeError, TypeError, NotImplementedError):
                return False
            _launch_flat_copy(src, dst, numel * esize // 4)
            return True
    if esize == 1 and numel <= _MAX_FLAT_ELEMS:
        _launch_flat_copy(view.reshape(-1), out.reshape(-1), numel)
        return True
    return False


def _has_broadcast_stride(view: torch.Tensor) -> bool:
    # A size>1 axis with stride 0 makes the kernel-side gather abort on gcu400;
    # such views (and anything the flat/dim paths cannot express) go to the
    # native fallback, which reads overlapping sources correctly.
    return any(s > 1 and st == 0 for s, st in zip(view.shape, view.stride()))


def _needs_wide_index(view: torch.Tensor, out: torch.Tensor) -> bool:
    # i32 lanes address up to 2^31 elements; larger strides/storage offsets
    # (possible even for views within _MAX_TRITON_ELEMENTS output) need i64.
    for tensor in (view, out):
        offset = tensor.storage_offset()
        for size, stride in zip(tensor.shape, tensor.stride()):
            if size > 1:
                offset += (size - 1) * stride
        if offset > _MAX_I32_OFFSET:
            return True
    return False


def _2d_tile_plan(view: torch.Tensor, out: torch.Tensor):
    """Return (src, dst, rows, cols, BLOCK_M, BLOCK_N) for the exact 2D tile
    kernel, or None.

    The 2D tile kernel is only exact when loads and stores each keep a
    coalescable axis: a size>1 unit-stride axis in the view and a contiguous
    out.  Tiles also stay within one scheduler wave.  Row-unit views are
    widened through int32 word views first (4 bytes per lane; measured
    -16~-18% on fp16/bool row slices).
    """
    if not out.is_contiguous():
        return None
    dim_0, dim_1 = view.shape
    strides = view.stride()
    if dim_1 > 1 and strides[1] == 1:
        src, dst = view, out
        cols = dim_1
        try:
            src = view.view(torch.int32)
            dst = out.view(torch.int32)
            cols = src.shape[1]
        except (RuntimeError, TypeError, NotImplementedError):
            pass
        block_n = min(triton.next_power_of_2(cols), 256)
        block_m = max(1, min(32, _TILE_ELEMS // block_n))
        if triton.cdiv(dim_0, block_m) * triton.cdiv(cols, block_n) > _GRID_CAP:
            return None
        return src, dst, dim_0, cols, block_m, block_n
    if dim_0 > 1 and strides[0] == 1:
        block_m, block_n = 64, 64
        if triton.cdiv(dim_0, block_m) * triton.cdiv(dim_1, block_n) > _GRID_CAP:
            return None
        return view, out, dim_0, dim_1, block_m, block_n
    return None


def _launch_as_strided_copy(view: torch.Tensor, out: torch.Tensor):
    """Launch the copy kernel; returns None when the caller must fall back."""
    if _has_broadcast_stride(view):
        return None
    if _try_flat_as_strided_copy(view, out):
        return out
    dim = view.dim()
    wide = _needs_wide_index(view, out)
    if dim == 1:
        n_elements = view.numel()
        block = _STRIDED_BLOCK if view.element_size() >= 4 else _BLOCK_SIZE
        grid = (min(_GRID_CAP, triton.cdiv(n_elements, block)),)
        _as_strided_copy_1d_kernel[grid](
            view,
            out,
            view.stride(0),
            out.stride(0),
            n_elements,
            BLOCK_SIZE=block,
            GRID=_GRID_CAP,
            WIDE_INDEX=wide,
        )
    elif dim == 2:
        dim_1 = view.shape[1]
        tile = _2d_tile_plan(view, out)
        if tile is not None:
            src, dst, rows, cols, block_m, block_n = tile
            grid = (triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))
            # 4-byte element tiles run best at 2 warps; word-widened tiles
            # (1/2-byte elements) keep the default 4 (measured on gcu400).
            num_warps = 2 if view.element_size() == 4 else 4
            _as_strided_copy_2d_kernel[grid](
                src,
                dst,
                src.stride(0),
                src.stride(1),
                dst.stride(0),
                dst.stride(1),
                rows,
                cols,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=num_warps,
            )
        else:
            n_elements = view.numel()
            grid = (min(_GRID_CAP, triton.cdiv(n_elements, _DIVMOD_BLOCK)),)
            _as_strided_copy_2d_divmod_kernel[grid](
                view,
                out,
                view.stride(0),
                view.stride(1),
                out.stride(0),
                out.stride(1),
                dim_1,
                n_elements,
                BLOCK_SIZE=_DIVMOD_BLOCK,
                GRID=_GRID_CAP,
                WIDE_INDEX=wide,
            )
    elif dim == 3:
        n_elements = view.numel()
        grid = (min(_GRID_CAP, triton.cdiv(n_elements, _DIVMOD3D_BLOCK)),)
        _as_strided_copy_3d_kernel[grid](
            view,
            out,
            view.stride(0),
            view.stride(1),
            view.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            view.shape[1],
            view.shape[2],
            n_elements,
            BLOCK_SIZE=_DIVMOD3D_BLOCK,
            GRID=_GRID_CAP,
            WIDE_INDEX=wide,
        )
    else:
        # >=4D strided views keep the native path: the pointwise DTE channel
        # used by the generic implementation has a documented MMU tail-overrun
        # fault mode on gcu400 and no validated accuracy here.
        return None
    return out


def _launch_byte_as_strided_copy(view: torch.Tensor, out: torch.Tensor):
    # Copy one-byte dtypes through uint8 views to avoid Triton fp8 scalar codegen.
    # The dtype-view API requires at least one logical dimension on some builds.
    byte_view = (
        view.reshape(1).view(torch.uint8) if view.dim() == 0 else view.view(torch.uint8)
    )
    byte_out = (
        out.reshape(1).view(torch.uint8) if out.dim() == 0 else out.view(torch.uint8)
    )
    if _launch_as_strided_copy(byte_view, byte_out) is None:
        return None
    return out


def as_strided_copy(input, size, stride, storage_offset=None):
    logger.debug("GEMS_ENFLAME AS_STRIDED_COPY")
    if input.device.type not in ("cuda", "gcu"):
        view = _make_as_strided_view(input, size, stride, storage_offset)
        return view.clone(memory_format=torch.contiguous_format)

    out = torch.empty(size, dtype=input.dtype, device=input.device)
    if out.numel() == 0:
        _make_as_strided_view(input, size, stride, storage_offset)
        return out

    view = _make_as_strided_view(input, size, stride, storage_offset)
    if _can_use_triton(view, out):
        result = _launch_as_strided_copy(view, out)
        if result is not None:
            return result
    if _can_use_byte_triton(view, out):
        result = _launch_byte_as_strided_copy(view, out)
        if result is not None:
            return result
    return _fallback_as_strided_copy(input, size, stride, storage_offset)


def as_strided_copy_out(input, size, stride, storage_offset=None, *, out):
    logger.debug("GEMS_ENFLAME AS_STRIDED_COPY_OUT")
    if out.dtype != input.dtype:
        # Match PyTorch's strict out-dtype contract without measuring native fallback.
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, but got {out.dtype} instead"
        )

    target_size = tuple(size)
    if tuple(out.shape) != target_size:
        out.resize_(target_size)

    if out.numel() == 0:
        _make_as_strided_view(input, size, stride, storage_offset)
        return out

    if input.device.type not in ("cuda", "gcu"):
        view = _make_as_strided_view(input, size, stride, storage_offset)
        if (
            torch._C._is_alias_of(input, out)
            or has_internal_overlapping(out) != MemOverlap.No
        ):
            view = view.clone(memory_format=torch.contiguous_format)
        out.copy_(view)
        return out

    if (
        torch._C._is_alias_of(input, out)
        or has_internal_overlapping(out) != MemOverlap.No
    ):
        return _fallback_as_strided_copy_out(
            input, size, stride, storage_offset, out=out
        )

    view = _make_as_strided_view(input, size, stride, storage_offset)
    if _can_use_triton(view, out):
        result = _launch_as_strided_copy(view, out)
        if result is not None:
            return result
    if _can_use_byte_triton(view, out):
        result = _launch_byte_as_strided_copy(view, out)
        if result is not None:
            return result
    return _fallback_as_strided_copy_out(input, size, stride, storage_offset, out=out)
