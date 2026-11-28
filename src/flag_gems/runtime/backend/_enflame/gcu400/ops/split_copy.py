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

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# A fixed 48-program grid (4 DIEs x 6 SIPs on GCU400) keeps every SIP busy and
# never trips the hardware grid.x limit (65535): each program walks a
# grid-strided range of tiles instead of owning exactly one tile.
_GRID_SIZE = 48
_BLOCK_SIZE = 8192
# The linear arm (source and destination coincide element for element) is a pure
# 1R1W stream, so it follows the 1R1W recipe: a wide 64K-element tile at 48 CTAs.
# Measured 2026-11-21 with an identical grid-stride body, native baseline in
# brackets: (2048x8192) fp16 split 512 48.4us(B8192) -> 26.7us [33.8us];
# (8192x8192) fp16 split 2048 112.5us -> 90.8us [99.6us];
# (8x8192x8192) fp16 split 2048 888us -> 694us [701us]; same shape fp32
# 1445us -> 1386us [1401us].
_LINEAR_BLOCK_SIZE = 65536
_MIN_BLOCK_SIZE = 256
_NUM_WARPS = 4
_MAX_RANK = 8
# The transposing copy keeps its tile inside a fixed byte budget: a wide tile
# amortises the transpose, but an over-wide one (a 512x128 tile over 4-byte
# elements) exceeds the register budget of the device.
_TRANSPOSE_TILE_BYTES = 128 * 1024
_TRANSPOSE_BLOCK_M = 256
_TRANSPOSE_NUM_WARPS = 8
_INT32_LIMIT = 2**31

# Tile walk modes of ``_split_copy_rows_kernel``. ``tl.constexpr`` keeps them
# readable inside the kernel, which may only reference constexpr globals.
_MODE_LINEAR = tl.constexpr(0)  # source and destination are both flat
_MODE_ROW = tl.constexpr(1)  # one row segment per tile
_MODE_ROW_TILE = tl.constexpr(2)  # several short rows grouped into one tile


@triton.jit
def _split_copy_rows_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    n_rows,
    outer,
    src_outer_stride,
    src_base,
    row_len,
    MODE: tl.constexpr,
    ROWS_PER_TILE: tl.constexpr,
    OUTER_SHIFT: tl.constexpr,
    SINGLE_COLUMN: tl.constexpr,
    EVEN: tl.constexpr,
    WIDE_INDEX: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy rows whose ``[dim, trailing]`` part is contiguous in both tensors.

    Each output chunk is a contiguous ``[outer, row_len]`` matrix with
    ``row_len = chunk_len * inner``; chunks sit back to back in the output
    buffer, so the destination offset of a row is ``row * row_len``. The same row
    starts at ``src_base + (row % outer) * src_outer_stride + (row / outer) *
    row_len`` in the source: the leading dimensions collapse into one index whose
    unit step is ``src_outer_stride``, and stepping to the next chunk adds one
    whole row.

    ``MODE`` picks the tile walk:

    * ``_MODE_LINEAR``: ``src_outer_stride == row_len``, so source and
      destination coincide element for element and the copy is a flat linear
      copy (the common case for a contiguous input).
    * ``_MODE_ROW``: one tile covers one row segment; the row index is derived
      once per tile from the tile index.
    * ``_MODE_ROW_TILE``: ``ROWS_PER_TILE`` short rows are grouped into a single
      tile so the tile still spans ``BLOCK_SIZE`` elements, while the index
      arithmetic stays a ``ROWS_PER_TILE``-wide vector instead of per element.

    ``EVEN`` marks that every tile is full, so no mask is needed at all;
    ``WIDE_INDEX`` marks that the flat offsets exceed the int32 range and must be
    widened before the address arithmetic.
    """
    pid = tl.program_id(axis=0)
    if MODE == _MODE_LINEAR:
        n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    else:
        n_block_columns = tl.cdiv(row_len, BLOCK_SIZE // ROWS_PER_TILE)
        n_tiles = tl.cdiv(n_rows, ROWS_PER_TILE) * n_block_columns

    for tile in tl.range(pid, n_tiles, GRID_SIZE):
        if MODE == _MODE_LINEAR:
            offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            if WIDE_INDEX:
                offsets = offsets.to(tl.int64)
            if EVEN:
                values = tl.load(src_ptr + src_base + offsets)
                tl.store(dst_ptr + offsets, values)
            else:
                mask = offsets < n_elements
                values = tl.load(src_ptr + src_base + offsets, mask=mask)
                tl.store(dst_ptr + offsets, values, mask=mask)
        elif MODE == _MODE_ROW:
            if SINGLE_COLUMN:
                row = tile
                column_block = 0
            else:
                row = tile // n_block_columns
                column_block = tile % n_block_columns
            if WIDE_INDEX:
                row = row.to(tl.int64)

            columns = column_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = columns < row_len
            if WIDE_INDEX:
                columns = columns.to(tl.int64)

            if OUTER_SHIFT >= 0:
                chunk_index = row >> OUTER_SHIFT
                outer_index = row & (outer - 1)
            else:
                chunk_index = row // outer
                outer_index = row % outer
            src_offsets = (
                src_base
                + outer_index * src_outer_stride
                + chunk_index * row_len
                + columns
            )
            dst_offsets = row * row_len + columns
            if EVEN:
                values = tl.load(src_ptr + src_offsets)
                tl.store(dst_ptr + dst_offsets, values)
            else:
                values = tl.load(src_ptr + src_offsets, mask=mask)
                tl.store(dst_ptr + dst_offsets, values, mask=mask)
        else:
            if SINGLE_COLUMN:
                row_tile = tile
                column_block = 0
            else:
                row_tile = tile // n_block_columns
                column_block = tile % n_block_columns

            rows = row_tile * ROWS_PER_TILE + tl.arange(0, ROWS_PER_TILE)
            columns = column_block * (BLOCK_SIZE // ROWS_PER_TILE) + tl.arange(
                0, BLOCK_SIZE // ROWS_PER_TILE
            )
            if WIDE_INDEX:
                rows = rows.to(tl.int64)
                columns = columns.to(tl.int64)

            if OUTER_SHIFT >= 0:
                chunk_index = rows >> OUTER_SHIFT
                outer_index = rows & (outer - 1)
            else:
                chunk_index = rows // outer
                outer_index = rows % outer
            row_bases = (
                src_base + outer_index * src_outer_stride + chunk_index * row_len
            )

            src_offsets = row_bases[:, None] + columns[None, :]
            dst_offsets = rows[:, None] * row_len + columns[None, :]
            if EVEN:
                values = tl.load(src_ptr + src_offsets)
                tl.store(dst_ptr + dst_offsets, values)
            else:
                mask = (rows < n_rows)[:, None] & (columns < row_len)[None, :]
                values = tl.load(src_ptr + src_offsets, mask=mask)
                tl.store(dst_ptr + dst_offsets, values, mask=mask)


@triton.jit
def _split_copy_transpose_kernel(
    src_ptr,
    dst_ptr,
    chunk_len,
    inner,
    src_row_stride,
    src_base,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """Transposing copy for a 2-D input whose fast axis is ``dim``.

    For ``base.t()`` style inputs (``stride(0) == 1``, split along dim 0) the
    chunk is ``dst[m, n] = src[src_base + m + n * src_row_stride]``: neither side
    is contiguous with the other, so a plain gather walks one tensor with a
    page-sized stride and loses most of the available bandwidth. Each tile
    therefore loads a ``BLOCK_N x BLOCK_M`` block whose *last* axis (``m``) is
    contiguous in the source, transposes it in registers with ``tl.trans``, and
    stores the ``BLOCK_M x BLOCK_N`` block whose last axis (``n``) is contiguous
    in the destination, keeping both the load and the store coalesced.
    """
    pid = tl.program_id(axis=0)
    n_column_blocks = tl.cdiv(inner, BLOCK_N)
    n_tiles = tl.cdiv(chunk_len, BLOCK_M) * n_column_blocks
    for tile in tl.range(pid, n_tiles, GRID_SIZE):
        row_block = tile // n_column_blocks
        column_block = tile % n_column_blocks
        rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        columns = column_block * BLOCK_N + tl.arange(0, BLOCK_N)

        # Source tile transposed: shape (BLOCK_N, BLOCK_M), last axis contiguous.
        src_offsets = (
            src_ptr + src_base + rows[None, :] + columns[:, None] * src_row_stride
        )
        dst_offsets = dst_ptr + rows[:, None] * inner + columns[None, :]
        if EVEN:
            tl.store(dst_offsets, tl.trans(tl.load(src_offsets)))
        else:
            row_mask = rows < chunk_len
            column_mask = columns < inner
            values = tl.load(src_offsets, mask=column_mask[:, None] & row_mask[None, :])
            tl.store(
                dst_offsets,
                tl.trans(values),
                mask=row_mask[:, None] & column_mask[None, :],
            )


@triton.jit
def _split_copy_gather_kernel(
    src_ptr,
    dst_ptr,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    n_elements,
    src_base,
    RANK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather one chunk out of an arbitrary strided input.

    The output chunk is contiguous, so its flat offset is the output index. Each
    output index is decomposed row-major over the chunk shape, whose ``dim``
    extent is the chunk length and whose remaining extents equal the input's;
    every recovered coordinate is scaled by the matching *input* stride (the
    chunked one included), and the wrapper folds ``chunk_start * stride(dim)``
    into ``src_base``.
    """
    pid = tl.program_id(axis=0)
    src_strides = [s0, s1, s2, s3, s4, s5, s6, s7]
    chunk_dims = [d0, d1, d2, d3, d4, d5, d6, d7]
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)

    for tile in tl.range(pid, n_tiles, GRID_SIZE):
        offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        offsets = offsets.to(tl.int64)

        rem = offsets
        src_offsets = tl.full([], src_base, tl.int64)
        for j in tl.static_range(RANK - 1, -1, -1):
            idx = rem % chunk_dims[j]
            rem = rem // chunk_dims[j]
            src_offsets += idx * src_strides[j]

        values = tl.load(src_ptr + src_offsets, mask=mask)
        tl.store(dst_ptr + offsets, values, mask=mask)


def _is_collapsible(shape, strides, lo, hi):
    """True when dimensions ``[lo, hi)`` form one unit-step contiguous run.

    A run is collapsible when each dimension's stride equals the extent that
    follows it times that dimension's stride, i.e.
    ``stride[i] == shape[i + 1] * stride[i + 1]``.
    """
    for i in range(lo, hi - 1):
        if strides[i] != shape[i + 1] * strides[i + 1]:
            return False
    return True


def _shift_of(value):
    """log2 of ``value`` when it is a power of two, else -1 (use div/mod)."""
    return value.bit_length() - 1 if value & (value - 1) == 0 else -1


def _launch_rows_copy(dst, src, n_chunks, outer, row_len, src_outer_stride, src_base):
    """Copy ``n_chunks`` uniform row-blocks of ``[outer, row_len]`` elements."""
    wide_index = (
        n_chunks * outer * row_len + src_base + src_outer_stride >= _INT32_LIMIT
    )
    if src_outer_stride == row_len:
        mode, rows_per_tile, block_size = _MODE_LINEAR, 1, _LINEAR_BLOCK_SIZE
        single_column, even = (
            True,
            n_chunks * outer * row_len % _LINEAR_BLOCK_SIZE == 0,
        )
    else:
        # Keep one tile at BLOCK_SIZE elements: short rows get grouped so the
        # tile stays wide, long rows are split into column blocks.
        columns_per_tile = min(
            _BLOCK_SIZE, max(_MIN_BLOCK_SIZE, triton.next_power_of_2(row_len))
        )
        rows_per_tile = max(1, _BLOCK_SIZE // columns_per_tile)
        block_size = columns_per_tile * rows_per_tile
        if rows_per_tile == 1:
            mode = _MODE_ROW
            single_column = row_len <= block_size
            even = row_len % block_size == 0
        else:
            mode = _MODE_ROW_TILE
            single_column = row_len <= columns_per_tile
            even = (
                row_len % columns_per_tile == 0
                and n_chunks * outer % rows_per_tile == 0
            )
    _split_copy_rows_kernel[(_GRID_SIZE,)](
        src,
        dst,
        n_chunks * outer * row_len,
        n_chunks * outer,
        outer,
        src_outer_stride,
        src_base,
        row_len,
        MODE=mode,
        ROWS_PER_TILE=rows_per_tile,
        OUTER_SHIFT=_shift_of(outer),
        SINGLE_COLUMN=single_column,
        EVEN=even,
        WIDE_INDEX=wide_index,
        GRID_SIZE=_GRID_SIZE,
        BLOCK_SIZE=block_size,
        num_warps=_NUM_WARPS,
    )


def split_copy(self: torch.Tensor, split_size: int, dim: int = 0):
    """Split ``self`` into chunks of ``split_size`` along ``dim`` and copy each.

    This mirrors ``aten::split_copy``: the out-of-place counterpart of
    ``torch.split``. The last chunk is smaller when ``dim``'s size is not
    divisible by ``split_size``; a ``split_size`` larger than the dimension
    yields a single whole-dimension chunk. Negative ``dim`` is normalized like
    PyTorch. Every output chunk is contiguous, freshly allocated, and never
    aliases the input. 0-dim inputs, out-of-range ``dim`` and non-positive
    ``split_size`` (unless the dimension itself is empty) are rejected with the
    same error messages PyTorch raises.
    """
    logger.debug("GEMS_ENFLAME SPLIT_COPY")

    if self.dim() == 0:
        raise RuntimeError("split expects at least a 1-dimensional tensor")
    if split_size < 0:
        raise RuntimeError(
            f"split expects split_size be non-negative, but got split_size={split_size}"
        )

    dim_size = self.size(dim)
    if split_size == 0 and dim_size != 0:
        raise RuntimeError(
            "split_size can only be 0 if dimension size is 0, but got dimension "
            f"size of {dim_size}"
        )
    dim = dim % self.dim()

    rank = self.dim()
    shape = [int(s) for s in self.shape]
    src_strides = [int(s) for s in self.stride()]

    # ``inner`` is the size of the trailing block that moves with one step along
    # ``dim``; ``outer`` counts the leading slices that are copied independently.
    inner = 1
    for i in range(dim + 1, rank):
        inner *= shape[i]
    outer = 1
    for i in range(dim):
        outer *= shape[i]

    # Fast path: the whole [dim, trailing] block is a unit-step contiguous run in
    # the source and the leading dimensions are a regular row-major nesting. Each
    # chunk is then ``outer`` contiguous runs of ``chunk_len * inner`` elements,
    # so no per-element decomposition is needed at all.
    block_contiguous = src_strides[dim] == inner and _is_collapsible(
        shape, src_strides, dim, rank
    )
    rows_are_flat = block_contiguous and _is_collapsible(shape, src_strides, 0, dim)
    # A transposed 2-D view (``stride(0) == 1``, split along dim 0) has no
    # contiguous axis in common with the output, so only a tiled transpose keeps
    # the source reads coalesced.
    is_2d_transposed = (
        rank == 2
        and dim == 0
        and inner > 1
        and src_strides[0] == 1
        and src_strides[1] != 1
    )

    # An empty dimension still splits into a single empty chunk (torch keeps one
    # output whose ``dim`` extent is 0).
    n_chunks = max(1, -(-dim_size // split_size)) if split_size else 1
    chunk_lens = [min(split_size, dim_size - c * split_size) for c in range(n_chunks)]

    if not rows_are_flat and not is_2d_transposed and rank > _MAX_RANK:
        raise RuntimeError(
            f"split_copy supports at most {_MAX_RANK} dimensions, got {rank}"
        )

    # One flat buffer holds every chunk back to back, which matches the
    # contiguous per-chunk layout the operator promises, so a single allocation
    # (and for uniform chunks a single launch) covers the whole operator.
    flat = torch.empty(self.numel(), dtype=self.dtype, device=self.device)
    result = []
    offset = 0
    for chunk_len in chunk_lens:
        out_shape = list(shape)
        out_shape[dim] = chunk_len
        chunk_elements = outer * chunk_len * inner
        result.append(flat.narrow(0, offset, chunk_elements).view(out_shape))
        offset += chunk_elements

    if self.numel() == 0:
        return result

    src_outer_stride = src_strides[dim - 1] if dim > 0 else inner
    with torch_device_fn.device(self.device):
        if rows_are_flat:
            # Uniform chunks keep the row formula valid across the whole flat
            # buffer, so they need a single launch; a ragged tail chunk (at most
            # one, always last) gets its own launch.
            chunk_len = chunk_lens[0]
            n_uniform = n_chunks if dim_size % split_size == 0 else n_chunks - 1
            if n_uniform > 0:
                row_len = chunk_len * inner
                _launch_rows_copy(
                    flat.narrow(0, 0, n_uniform * outer * row_len),
                    self,
                    n_uniform,
                    outer,
                    row_len,
                    # The chunk-to-chunk step of a single uniform chunk is not
                    # needed, so the flat-copy shortcut is always valid for dim 0.
                    src_outer_stride if dim > 0 else row_len,
                    0,
                )
            if n_uniform < n_chunks:
                tail_len = chunk_lens[-1]
                tail_row_len = tail_len * inner
                _launch_rows_copy(
                    flat.narrow(
                        0,
                        n_uniform * outer * chunk_len * inner,
                        tail_len * outer * inner,
                    ),
                    self,
                    1,
                    outer,
                    tail_row_len,
                    src_outer_stride if dim > 0 else tail_row_len,
                    n_uniform * chunk_len * src_strides[dim],
                )
        elif is_2d_transposed:
            transpose_block_n = max(
                32,
                min(
                    _TRANSPOSE_BLOCK_M,
                    _TRANSPOSE_TILE_BYTES // self.element_size() // _TRANSPOSE_BLOCK_M,
                ),
            )
            chunk_start = 0
            for chunk_len, out in zip(chunk_lens, result):
                _split_copy_transpose_kernel[(_GRID_SIZE,)](
                    self,
                    out,
                    chunk_len,
                    inner,
                    src_strides[1],
                    chunk_start * src_strides[0],
                    BLOCK_M=_TRANSPOSE_BLOCK_M,
                    BLOCK_N=transpose_block_n,
                    EVEN=chunk_len % _TRANSPOSE_BLOCK_M == 0
                    and inner % transpose_block_n == 0,
                    GRID_SIZE=_GRID_SIZE,
                    num_warps=_TRANSPOSE_NUM_WARPS,
                )
                chunk_start += chunk_len
        else:
            # The chunk shape mirrors the input shape with the chunked dimension
            # clamped to the chunk length; both arrays are padded to 8 entries
            # with zero-stride / size-1 slots so the kernel always receives its
            # full fixed argument sets. Order matters: strides first, dims
            # second, matching the kernel signature.
            dim_stride = src_strides[dim]
            chunk_start = 0
            for chunk_len, out in zip(chunk_lens, result):
                chunk_dims = list(shape)
                chunk_dims[dim] = chunk_len
                padded_strides = src_strides + [0] * (_MAX_RANK - rank)
                padded_dims = chunk_dims + [1] * (_MAX_RANK - rank)
                # ``src_base`` only carries the chunk's first position along
                # ``dim``: Triton receives a data pointer that already includes
                # the storage offset, so it must not be added again here.
                _split_copy_gather_kernel[(_GRID_SIZE,)](
                    self,
                    out,
                    *padded_strides,
                    *padded_dims,
                    out.numel(),
                    chunk_start * dim_stride,
                    RANK=rank,
                    GRID_SIZE=_GRID_SIZE,
                    BLOCK_SIZE=_BLOCK_SIZE,
                    num_warps=_NUM_WARPS,
                )
                chunk_start += chunk_len

    return result
