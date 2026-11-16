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

"""Kunlunxin (XPU) specialization of ``aten::fliplr``.

``torch.fliplr`` flips a tensor along dimension 1 (left/right).  On the XPU
stack ``torch.fliplr`` materializes a *contiguous* result (it is a flip + copy,
not a view), so the generic ``torch.empty_like(self)`` with the default
``preserve_format`` produces the wrong strides for transposed / non-contiguous
inputs: the generic kernel values are correct, but the returned tensor keeps the
input's non-contiguous strides and ``result.stride() != torch.fliplr(ref).stride()``
(the CPU reference ``.to("cpu")`` also makes the reference contiguous).

The generic implementation is also extremely slow on this backend: it performs a
per-element reversed gather along the width axis, and a stride ``-1`` gather has
no vectorised path on TritonXPU (D-017), so it runs ~10x slower than the vendor
flip.

The contiguous input is laid out as ``(outer, width, inner)`` with
``outer = shape[0]``, ``width = shape[1]``, ``inner = prod(shape[2:])``, so
``fliplr`` only reorders whole ``inner``-element *contiguous runs*: the run at
column ``j`` of row ``o`` lands at column ``W-1-j`` of the same row.  Three
paths implement that, in order of preference:

1. ``inner >= 2``: the reordering is a 2-D DMA.  One ``tle.dsa`` copy reads the
   source run with the destination's row stride and unit-stride columns, and a
   second one writes it back; both sides stay affine, so this runs at the SDNN
   DMA engine's block-DMA bandwidth (measured 2.7x the vendor ``fliplr`` on
   ``(64, 512, 512)`` fp16 and 5.0x on ``(16, 128, 64, 64)``).  This is the flat
   row-gather recipe from ``techniques/xpu3-flat-rowkeep-gather-copy.md`` with
   the per-element index gather replaced by a DMA row selector.
2. ``inner >= 1024`` (and divisible bythe tile) when TLE is unavailable: the
   per-run reversal is still one block-DMA copy per chunk with the *chunk* index
   reversed once per program.
3. otherwise (``inner == 1``, i.e. a plain 2-D input, or TLE unavailable): the
   reversal is along the innermost axis and must fall back to the reversed
   gather, which is the structural wall on this backend -- there is no
   vectorised negative-stride load, and neither ``tl.flip`` (uni_sram
   exhaustion) nor a negative DMA stride (does not compile) is expressible.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.tle_copy import tle_dma_available

try:  # `tle.dsa` (SDNN / DMA engine) exists only on the xpu3 tile-language build
    import triton.experimental.tle.language as tle

    _HAS_TLE = hasattr(tle, "dsa") and all(
        hasattr(tle.dsa, a) for a in ("alloc", "copy", "to_tensor", "UNI_SRAM")
    )
except ImportError:  # triton without the XPU tile-language extension
    tle = None
    _HAS_TLE = False

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def fliplr_contiguous_kernel(
    input,
    output,
    n_elements,
    width,
    inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    offsets = offsets.to(tl.int64)
    column = (offsets // inner_size) % width
    input_offsets = offsets + (width - 1 - 2 * column) * inner_size
    values = tl.load(input + input_offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


@libentry()
@triton.jit
def fliplr_block_kernel(
    input,
    output,
    width,
    inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program copies one contiguous BLOCK_SIZE chunk that lies entirely
    # inside a single (row, column) block of ``inner_size`` elements.  The column
    # block index is reversed once per program, so both load and store stay
    # affine (block DMA) and only the block *order* is reversed.
    pid = tl.program_id(0)
    chunks_per_unit = inner_size // BLOCK_SIZE
    unit = pid // chunks_per_unit
    chunk = pid % chunks_per_unit
    j = unit % width
    row = unit // width
    src_unit = row * width + (width - 1 - j)
    offsets = unit * inner_size + chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    src_offsets = src_unit * inner_size + chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input + src_offsets)
    tl.store(output + offsets, values)


@libentry()
@triton.jit
def fliplr_strided_kernel(
    input,
    output,
    n_elements,
    SHAPE: tl.constexpr,
    INPUT_STRIDES: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    linear = offsets.to(tl.int64)
    input_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for dim in tl.static_range(NDIM - 1, -1, -1):
        index = linear % SHAPE[dim]
        linear //= SHAPE[dim]
        input_index = SHAPE[dim] - 1 - index if dim == 1 else index
        input_offsets += input_index * INPUT_STRIDES[dim]
    values = tl.load(input + input_offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


# uni_sram budget per program on the dsa path (see utils/tle_copy.py).
_DSA_SRAM_BYTES = 32768
# A tile is one DMA descriptor pair.  Below this many elements per program the
# per-descriptor cost dominates and the reversed gather kernel is cheaper, so
# the dsa path is only taken for tiny shapes when the program count is small.
_DSA_MIN_TILE = 512
_DSA_MIN_TILE_FEW_PROGRAMS = 4
_DSA_FEW_PROGRAMS = 1024
# The DMA row count is a descriptor loop count; nothing above this has been
# measured better and a tall tile wastes uni_sram (a whole tile per program).
_DSA_MAX_ROWS = 128

# `tle.dsa` moves 1-, 2- and 4-byte elements; the 4-byte transfer has to be
# spelled f32 (int32 faults and int64 is rejected, see `_DSA_MOVE_DTYPE` in
# utils/tle_copy.py).  8-byte elements are excluded on purpose: they move as two
# 4-byte words, and reversing at that width would split each element in half.
_DSA_MOVE = {
    1: (torch.int8, tl.int8),
    2: (torch.int16, tl.int16),
    4: (torch.float32, tl.float32),
}
# `s_row`/the plane offset is a 32-bit kernel argument.
_DSA_MAX_NUMEL = 1 << 31


@triton.jit(
    # Only the tile geometry changes the code; the extents and the row stride
    # change per shape, and specializing on them would recompile per shape for
    # no gain (same reasoning as `_tle_dsa_row_copy_kernel`).
    do_not_specialize=["outer", "width", "inner", "s_row"]
)
def fliplr_dsa_kernel(
    in_ptr,
    out_ptr,
    outer,
    width,
    inner,
    s_row,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    DT: tl.constexpr,
):
    """Move one mirrored run per program with one 2-D DMA in and one out.

    For the run of output column ``j`` the source is the run of column
    ``width - 1 - j``: `outer` rows strided by `s_row` (the whole
    ``width * inner`` plane) with `inner` contiguous elements per row.  The
    destination has the same geometry inside the output plane, so the two
    descriptors stay affine on both sides and the order *inside* a run is never
    touched -- which is exactly what a stride ``-1`` gather cannot do on this
    backend.  Tails go through `sizes`, never a mask (a masked dsa copy loses
    its extent), and every index is clamped by `sizes`, so no lane can describe
    an address outside either allocation.
    """
    pid = tl.program_id(0)
    j = pid % width
    o0 = (pid // width) * ROWS
    c0 = tl.program_id(1) * COLS
    r = tl.arange(0, ROWS)
    c = tl.arange(0, COLS)
    row_tail = tl.minimum(outer - o0, ROWS)
    col_tail = tl.minimum(inner - c0, COLS)
    buf = tle.dsa.alloc([ROWS, COLS], DT, tle.dsa.UNI_SRAM)
    src = (
        in_ptr + (width - 1 - j) * inner + (o0 + r)[:, None] * s_row + (c0 + c)[None, :]
    )
    tle.dsa.copy(src, buf, sizes=[row_tail, col_tail])
    dst = out_ptr + j * inner + (o0 + r)[:, None] * s_row + (c0 + c)[None, :]
    tle.dsa.copy(buf, dst, sizes=[row_tail, col_tail])


def _dsa_plan(outer: int, inner: int, element_size: int):
    """(ROWS, COLS) for the uni_sram tile: cover `outer` if it fits, else widen.

    The best measured point is the widest tile the 32 KB uni_sram budget allows
    (a 16 K element fp16 tile moves a 4 M element tensor in 94 us on
    ``(64, 512, 512)``, four times faster than a 4 K element tile), with as many
    rows as `outer` needs so the row loop count stays low.
    """
    max_elems = max(_DSA_SRAM_BYTES // element_size, 1)
    rows = min(triton.next_power_of_2(min(outer, _DSA_MAX_ROWS)), max_elems)
    cols = min(triton.next_power_of_2(inner), max(max_elems // rows, 1))
    while rows > 1 and rows * cols > max_elems:
        rows //= 2
    while cols > 1 and rows * cols > max_elems:
        cols //= 2
    return max(rows, 1), max(cols, 1)


def _fliplr_dsa(src, out, outer: int, width: int, inner: int) -> bool:
    """Try the DMA path; False when it cannot express this launch."""
    move = _DSA_MOVE.get(src.element_size())
    if move is None:
        return False
    rows, cols = _dsa_plan(outer, inner, src.element_size())
    row_blocks = -(-outer // rows)  # ceil, avoids the triton.cdiv host cost
    col_blocks = -(-inner // cols)
    tile = rows * cols
    n_prog = row_blocks * width * col_blocks
    if tile < _DSA_MIN_TILE_FEW_PROGRAMS or (
        tile < _DSA_MIN_TILE and n_prog > _DSA_FEW_PROGRAMS
    ):
        return False
    move_dtype, tl_dtype = move
    try:
        # A bit reinterpret, not a conversion: the move never reads a value.
        src_v = src.view(move_dtype)
        out_v = out.view(move_dtype)
    except RuntimeError:  # not reinterpretable (odd last extent / layout)
        return False
    grid = (row_blocks * width, col_blocks)
    with torch_device_fn.device(src.device):
        fliplr_dsa_kernel[grid](
            src_v,
            out_v,
            outer,
            width,
            inner,
            width * inner,
            ROWS=rows,
            COLS=cols,
            DT=tl_dtype,
            is_sdnn=True,
        )
    return True


# Candidate block sizes for the block-reverse path, largest first.  Only large
# blocks (>= 1024 lanes) are kept: the block-reverse only pays off when each
# program moves a wide contiguous block (block DMA).  Smaller tiles (512/256/
# 128/64) produce many more programs with a worse tile width and end up slower
# than the plain reversed gather (measured on the (64, 512, 512) inner==512
# case: block 512 is ~1.2x slower than the gather for fp16/bf16), so inner
# sizes below 1024 fall back to the gather kernel instead.
_BLOCK_SIZES = (4096, 2048, 1024)


def _pick_block_size(inner_size: int):
    for block_size in _BLOCK_SIZES:
        if inner_size % block_size == 0:
            return block_size
    return None


def fliplr(self: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN FLIPLR")

    if self.ndim < 2:
        raise RuntimeError("Input must be >= 2-d.")

    # torch.fliplr (and the CPU reference via ``.to("cpu")``) yields a contiguous
    # result, so allocate a contiguous output regardless of input layout.
    output = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    n_elements = self.numel()
    if n_elements == 0:
        return output

    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(self.device):
        if self.is_contiguous():
            width = self.shape[1]
            inner_size = math.prod(self.shape[2:])
            # A run is reordered as a whole, so any `inner_size >= 2` is a pure
            # 2-D DMA (see the module docstring).  `inner_size == 1` reverses
            # the innermost axis and has no vectorised expression here.
            if (
                inner_size > 1
                and n_elements < _DSA_MAX_NUMEL
                and _HAS_TLE
                and tle_dma_available()
                and _fliplr_dsa(self, output, self.shape[0], width, inner_size)
            ):
                return output
            block_size = _pick_block_size(inner_size)
            if block_size is not None and inner_size > 1:
                grid = (n_elements // block_size,)
                fliplr_block_kernel[grid](
                    self,
                    output,
                    width,
                    inner_size,
                    BLOCK_SIZE=block_size,
                )
            else:
                fliplr_contiguous_kernel[grid](
                    self,
                    output,
                    n_elements,
                    width,
                    inner_size,
                    BLOCK_SIZE=1024,
                )
        else:
            fliplr_strided_kernel[grid](
                self,
                output,
                n_elements,
                SHAPE=tuple(self.shape),
                INPUT_STRIDES=tuple(self.stride()),
                NDIM=self.ndim,
                BLOCK_SIZE=1024,
            )

    return output
