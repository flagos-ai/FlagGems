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

"""tle implementation of the pure-copy operators.

Two kernels cover the copies tle can express:

* `_tle_tile_copy_kernel` -- both sides contiguous, same dtype: one flat TMA
  tile GM -> LM -> GM on the `tle.gpu` cluster path. This is the throughput path
  (1.5-1.7x the pointwise kernel up to ~16M elements on KL3, level with
  `aten::copy_` above that), and with `_tile_launch` reusing the compiled kernel
  it is also what small copies ride on, where the host side of a launch is the
  whole cost.
* `_tle_dsa_row_copy_kernel` -- everything else, on the `tle.dsa` SDNN path. It
  moves rows of a contiguous run through a uni_sram buffer, with independent
  src/dst row strides: the shape the DMA engine actually has, and the same one
  the hand-written `memcpy_2d_sdnn.xpu` uses (`dma_cfg_2d(loop, dst_stride,
  src_stride)`). Broadcast rows (src row stride 0) and overlapping reads come
  for free, and `CONVERT` casts on the tensor side between the two transfers, so
  a dtype-changing `copy_` stays on the same kernel. It replaced a `tle.gpu`
  element-wise gather, which moved the same bytes ~200x slower than the fallback
  did (4096x4096 f32 strided window: 32.8ms, against 157us for `aten::copy_` and
  114us here).

Anything outside that envelope makes `tle_copy` return False and the caller
keeps its own fallback. Measured on KL3, the dsa path is out when:

* the copy needs a dtype `tle.dsa` cannot handle. A pure move goes as a 1-, 2- or
  4-byte bit pattern (`_DSA_MOVE_DTYPE`), with an 8-byte element split into two
  4-byte ones, which needs `stride(-1) == 1`; a converting copy needs the real
  element types on both sides (`_DSA_BUF_DTYPE` / `_DSA_VAL_DTYPE`) and cannot
  widen an integer into a float, which the SDNN cast turns into zeros
* the innermost dimension is not unit-stride on *both* sides. A stride inside
  the row would need a third DMA level; asking for it moves the wrong elements
  without any diagnostic. A real transpose lands here, as does a destination
  like `y[::2]`
* the layout still needs more than `MAX_COPY_RANK` dimensions after collapsing
"""

import logging
import math
import os

import torch
import triton
import triton.language as tl
from triton.runtime import driver

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

try:
    # `.language` carries both surfaces: `tle.gpu` for the cluster/LM path and
    # `tle.dsa` for the SDNN one.
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False

# tritonxpu-tle-core-tiling hands each of the 64 cores whole rows of the tile.
CORE_NUM = 64
# LM per core for the tile path. triton/docs/xpu3/tle_user_guide.md quotes ~2KB
# as the budget and test/tle/test_tle_copy_1d.py sizes its block that way, but
# 4KB still compiles and is measurably better on large tensors (fp16 2**28:
# 630us -> 577us, i.e. level with aten::copy_); 8KB fails to allocate.
LM_BYTES_PER_CORE = 4096
# uni_sram per program on the dsa path, and the bytes per row inside that tile.
# The row is what one DMA descriptor moves, so it wants to be a few KB; the tile
# is the on-chip buffer, which the multi-buffer allocator replicates.
DSA_TILE_BYTES = 32768
DSA_ROW_BYTES = 4096
# Square tile for the on-chip transpose. 64x64 was the only size that worked:
# 128x128 and 32x256 both land around 120-140ms for a 2048x2048 f16 transpose
# against 66us at 64x64.
DSA_TRANS_TILE = 64
# Dimensions the copy addresses: two in the tile, the rest in the grid.
MAX_COPY_RANK = 5

# dtypes verified on KL3. torch.bool is absent on purpose: i1 is packed in LM and
# comes back corrupted, so bool tensors move through an int8 view (_byte_view).
_TL_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
}

# What the SDNN DMA moves. A pure move only cares about the width, and the
# widths that work are 1, 2 and 4 bytes -- with the 4-byte transfer spelled f32,
# because int32 compiles and then faults with an illegal access and int64 is
# rejected as "Unsupported dma dst data type". Both do so for a plain
# `tl.load`/`tl.store` kernel launched with `is_sdnn=True`, so they are backend
# gaps rather than something this kernel could spell differently; viewing the
# bits as f32 sidesteps them, since a copy never does arithmetic on the value.
_DSA_MOVE_DTYPE = {1: torch.int8, 2: torch.int16, 4: torch.float32}

# A converting copy does look at the element type: the buffer holds real `SRC`
# values and the cast produces real `DST` ones. bf16 is missing from the buffer
# side because a bf16 uni_sram buffer fails to build ("element types do not
# match" on `bufferization.to_tensor`); as a cast *target* it is fine, since that
# is a store rather than a buffer.
_DSA_BUF_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.int8: tl.int8,
    torch.int16: tl.int16,
}
_DSA_VAL_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.int8: tl.int8,
    torch.int16: tl.int16,
}


@triton.jit
def _tle_tile_copy_kernel(src_desc, dst_desc, BLOCK: tl.constexpr, DTYPE: tl.constexpr):
    """Flat TMA tile: the descriptor lowering clamps the tail block itself."""
    pid = tl.program_id(0)
    buf = tle.gpu.alloc([BLOCK], dtype=DTYPE, layout=None, scope=tle.gpu.lmem)
    tle.gpu.copy(src_desc, buf, [BLOCK], [pid * BLOCK])
    tle.gpu.copy(buf, dst_desc, [BLOCK], [pid * BLOCK])


@triton.jit(
    # None of these change the code, only the data it moves, so specializing on
    # them buys nothing and costs a kernel per class. That matters because the
    # callers allocate a fresh `out` per call and the XPU allocator hands back
    # pointers from varying divisibility classes: with specialization on, a
    # repeated call keeps reloading a kernel. Measured over one sweep of six
    # strided copies, 482s of wall clock against 12s, and the two worst shapes
    # went from 14.7ms and 25.0ms per call to 334us and 199us.
    do_not_specialize=[
        "src_ptr",
        "dst_ptr",
        "rows",
        "cols",
        "s_row",
        "t_row",
        "s_col",
        "row_blocks",
        "d2",
        "d3",
        "d4",
        "s2",
        "s3",
        "s4",
        "t2",
        "t3",
        "t4",
    ]
)
def _tle_dsa_row_copy_kernel(
    src_ptr,
    dst_ptr,
    rows,
    cols,
    s_row,
    t_row,
    s_col,
    row_blocks,
    d2,
    d3,
    d4,
    s2,
    s3,
    s4,
    t2,
    t3,
    t4,
    OUTER_RANK: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    COL_STRIDED: tl.constexpr,
    SRC_DTYPE: tl.constexpr,
    DST_DTYPE: tl.constexpr,
    CONVERT: tl.constexpr,
    TO_BOOL: tl.constexpr,
):
    """Rows of a run through a uni_sram buffer.

    `cols` is the innermost dimension and `rows` the next one out, with
    `s_row` / `t_row` its src and dst strides -- one 2D DMA in and one out, which
    is what the engine does natively. Everything further out (`d2..d4` with
    strides `s2..s4` / `t2..t4`) is a scalar base offset picked from the program
    id, so it costs nothing in the transfer.

    The destination run is always contiguous. The source one is too unless
    COL_STRIDED, which reads it with `s_col` -- that is a per-element transfer,
    so the caller pairs it with `ROWS == 1`: a 2D tile whose rows also carry a
    column stride needs a third DMA level and moves the wrong elements instead.

    Tails use `sizes` rather than a mask: a masked load hides its extent in a
    view of the staging buffer that the dsa rewrite drops, and the compiler
    rejects that. On the way out the same `sizes` becomes the store mask.

    With CONVERT the value is cast on the tensor side between the two transfers,
    which is how a dtype-changing `copy_` is served; TO_BOOL reproduces
    PyTorch's "nonzero becomes True" rule.
    """
    pid = tl.program_id(0)
    row0 = (pid % row_blocks) * ROWS
    outer = pid // row_blocks
    col0 = tl.program_id(1) * COLS

    src_base = 0
    dst_base = 0
    if OUTER_RANK > 0:
        i = outer % d2
        outer = outer // d2
        src_base += i * s2
        dst_base += i * t2
    if OUTER_RANK > 1:
        i = outer % d3
        outer = outer // d3
        src_base += i * s3
        dst_base += i * t3
    if OUTER_RANK > 2:
        i = outer % d4
        src_base += i * s4
        dst_base += i * t4

    r = row0 + tl.arange(0, ROWS)
    c = col0 + tl.arange(0, COLS)
    row_tail = tl.minimum(rows - row0, ROWS)
    col_tail = tl.minimum(cols - col0, COLS)

    buf = tle.dsa.alloc([ROWS, COLS], SRC_DTYPE, tle.dsa.UNI_SRAM)
    if COL_STRIDED:
        src = src_ptr + src_base + r[:, None] * s_row + c[None, :] * s_col
    else:
        src = src_ptr + src_base + r[:, None] * s_row + c[None, :]
    tle.dsa.copy(src, buf, sizes=[row_tail, col_tail])

    dst = dst_ptr + dst_base + r[:, None] * t_row + c[None, :]
    if CONVERT:
        val = tle.dsa.to_tensor(buf)
        if TO_BOOL:
            val = (val != 0).to(DST_DTYPE)
        else:
            val = val.to(DST_DTYPE)
        tle.dsa.copy(val, dst, sizes=[row_tail, col_tail])
    else:
        tle.dsa.copy(buf, dst, sizes=[row_tail, col_tail])


@triton.jit(
    do_not_specialize=[
        "src_ptr",
        "dst_ptr",
        "rows",
        "cols",
        "s_col",
        "t_row",
        "col_blocks",
        "d2",
        "d3",
        "d4",
        "s2",
        "s3",
        "s4",
        "t2",
        "t3",
        "t4",
    ]
)
def _tle_dsa_trans_copy_kernel(
    src_ptr,
    dst_ptr,
    rows,
    cols,
    s_col,
    t_row,
    col_blocks,
    d2,
    d3,
    d4,
    s2,
    s3,
    s4,
    t2,
    t3,
    t4,
    OUTER_RANK: tl.constexpr,
    TILE: tl.constexpr,
    SRC_DTYPE: tl.constexpr,
    DST_DTYPE: tl.constexpr,
    CONVERT: tl.constexpr,
    TO_BOOL: tl.constexpr,
):
    """A square tile transposed on chip, for layouts whose two sides disagree.

    When the destination's contiguous run is the source's strided one and vice
    versa -- a transpose, or `unfold` along an inner axis -- neither side can be
    read and written as the same run. Reading one of them per element is what the
    row kernel would do, and it is an order of magnitude off: `transpose_021_sdnn_bsp`
    in the XDNN kernel library never does that. It moves a tile in with a 2D DMA
    (contiguous rows, row stride only), transposes it on chip with the DS
    shuffle, and writes it out with another 2D DMA. This is that, with `tl.trans`
    standing in for `ds_shuffle_coa_1d`.

    So `rows` is the extent the source is contiguous along, `cols` the extent the
    destination is, and both DMAs stay contiguous inside a row. TILE is square
    because the transposed tile has to fit the same buffer.
    """
    pid = tl.program_id(0)
    col0 = (pid % col_blocks) * TILE
    outer = pid // col_blocks
    row0 = tl.program_id(1) * TILE

    src_base = 0
    dst_base = 0
    if OUTER_RANK > 0:
        i = outer % d2
        outer = outer // d2
        src_base += i * s2
        dst_base += i * t2
    if OUTER_RANK > 1:
        i = outer % d3
        outer = outer // d3
        src_base += i * s3
        dst_base += i * t3
    if OUTER_RANK > 2:
        i = outer % d4
        src_base += i * s4
        dst_base += i * t4

    c = col0 + tl.arange(0, TILE)
    r = row0 + tl.arange(0, TILE)
    col_tail = tl.minimum(cols - col0, TILE)
    row_tail = tl.minimum(rows - row0, TILE)

    buf = tle.dsa.alloc([TILE, TILE], SRC_DTYPE, tle.dsa.UNI_SRAM)
    src = src_ptr + src_base + c[:, None] * s_col + r[None, :]
    tle.dsa.copy(src, buf, sizes=[col_tail, row_tail])

    val = tl.trans(tle.dsa.to_tensor(buf))
    if CONVERT:
        if TO_BOOL:
            val = (val != 0).to(DST_DTYPE)
        else:
            val = val.to(DST_DTYPE)
    dst = dst_ptr + dst_base + r[:, None] * t_row + c[None, :]
    tle.dsa.copy(val, dst, sizes=[row_tail, col_tail])


def tle_dma_available():
    """tle.gpu exists only on the xpu3 (KL3) cluster pipeline."""
    if not _HAS_TLE:
        return False
    if os.environ.get("TRITON_ENABLE_XCN_BACKEND"):
        return False
    return os.environ.get("TRITON_XPU_ARCH", "3") == "3"


def _byte_view(t: torch.Tensor) -> torch.Tensor:
    """bool has no LM representation (i1 is packed), so move it as int8."""
    return t.view(torch.int8) if t.dtype == torch.bool else t


def _collapse(shape, src_stride, dst_stride):
    """Fold the copy into as few dimensions as the two stride sets allow.

    Returns (dims, src_strides, dst_strides) innermost-first and padded to
    MAX_COPY_RANK, plus the real rank; None if it does not fit.
    """
    dims = [
        (int(n), int(ss), int(ds))
        for n, ss, ds in zip(shape, src_stride, dst_stride)
        if n != 1
    ]
    if not dims:
        dims = [(1, 0, 0)]
    # Innermost = fastest varying in the destination, so the scatter stays as
    # close to sequential as the layout allows.
    dims.sort(key=lambda dim: abs(dim[2]))
    merged = []
    for n, ss, ds in dims:
        if merged:
            pn, pss, pds = merged[-1]
            if ss == pss * pn and ds == pds * pn:
                merged[-1] = (pn * n, pss, pds)
                continue
        merged.append((n, ss, ds))
    rank = len(merged)
    if rank > MAX_COPY_RANK:
        return None
    pad = [(1, 0, 0)] * (MAX_COPY_RANK - rank)
    merged += pad
    return (
        [m[0] for m in merged],
        [m[1] for m in merged],
        [m[2] for m in merged],
        rank,
    )


def _tile_launch(src: torch.Tensor, dst: torch.Tensor, numel: int, dtype, block: int):
    """Launch the tile kernel, going through `JITFunction.run` only once.

    A small copy is host-bound: a 16KB move takes ~8us on device while `fn[grid]`
    spends ~20us re-deriving what cannot change once the kernel is compiled. The
    first call binds the kernel with the driver's launcher cache, later ones hand it
    the kernel ABI directly: each descriptor already split into base pointer,
    `.shape` (i32) and `.strides` (i64).

    The kernel's only shape-dependent argument is that `.shape`, passed at launch,
    so one binding serves every size that shares a grid -- and every copy below
    64K-256K elements (dtype-dependent) has `grid == 1`.
    """
    grid = triton.cdiv(numel, block)
    launchers = getattr(driver.active, "flat_launchers", None)
    if launchers is None:
        # triton without the launcher cache: correct, just slower every call.
        _tle_tile_copy_kernel[(grid,)](
            TensorDescriptor.from_tensor(src.view(-1), [block]),
            TensorDescriptor.from_tensor(dst.view(-1), [block]),
            block,
            _TL_DTYPE[dtype],
        )
        return

    key = (dtype, grid)
    launch, stream = launchers.acquire(_tle_tile_copy_kernel, key)
    if launch is None:
        kernel = _tle_tile_copy_kernel[(grid,)](
            TensorDescriptor.from_tensor(src.view(-1), [block]),
            TensorDescriptor.from_tensor(dst.view(-1), [block]),
            block,
            _TL_DTYPE[dtype],
        )
        launchers.bind(_tle_tile_copy_kernel, key, kernel, (grid,))
        return

    launch(stream, src.data_ptr(), numel, 1, dst.data_ptr(), numel, 1)


def _dsa_tile(rows: int, cols: int, element_size: int):
    """(ROWS, COLS) for the on-chip tile: full rows first, then as many as fit."""
    cols_block = min(
        triton.next_power_of_2(cols), max(DSA_ROW_BYTES // element_size, 1)
    )
    rows_block = min(
        triton.next_power_of_2(rows),
        max(DSA_TILE_BYTES // (cols_block * element_size), 1),
    )
    return rows_block, cols_block


def tle_copy(src: torch.Tensor, dst: torch.Tensor) -> bool:
    """Copy `src` into `dst` with tle; False if tle cannot express it."""
    if not tle_dma_available():
        return False
    if src.device != dst.device or src.numel() != dst.numel() or src.numel() == 0:
        return False

    src_v, dst_v = _byte_view(src), _byte_view(dst)
    numel = src.numel()
    convert = src.dtype != dst.dtype

    if not convert and src_v.is_contiguous() and dst_v.is_contiguous():
        tile_ty = _TL_DTYPE.get(src_v.dtype)
        if tile_ty is None:
            return False
        block = LM_BYTES_PER_CORE // src_v.element_size() * CORE_NUM
        _tile_launch(src_v, dst_v, numel, src_v.dtype, block)
        return True

    if convert:
        src_ty = _DSA_BUF_DTYPE.get(src_v.dtype)
        dst_ty = _DSA_VAL_DTYPE.get(dst_v.dtype)
        if src_ty is None or dst_ty is None:
            return False
        # int -> float comes back all zeros, and does so for a plain
        # `tl.load(...).to(tl.float32)` under `is_sdnn=True` as well, so it is the
        # SDNN cast rather than this kernel. float -> int and int -> int are fine.
        if not src_v.dtype.is_floating_point and dst_v.dtype.is_floating_point:
            return False
    else:
        # Same dtype on both sides, so the transfer is just bits: move them as
        # whatever width the DMA has.
        move = _DSA_MOVE_DTYPE.get(src_v.element_size())
        if move is None:
            # 8 bytes has no transfer width, so move each element as two 4-byte
            # ones. `view` does exactly that -- doubling the last extent and
            # every stride -- and refuses unless `stride(-1) == 1`, which is the
            # same run the row path needs contiguous anyway.
            try:
                src_v = src_v.view(torch.float32)
                dst_v = dst_v.view(torch.float32)
            except RuntimeError:
                return False
            move = torch.float32
        src_v = src_v if src_v.dtype == move else src_v.view(move)
        dst_v = dst_v if dst_v.dtype == move else dst_v.view(move)
        src_ty = dst_ty = _DSA_BUF_DTYPE[move]

    if src_v.shape != dst_v.shape:
        return False
    collapsed = _collapse(src_v.shape, src_v.stride(), dst_v.stride())
    if collapsed is None:
        return False
    dims, src_strides, dst_strides, rank = collapsed

    cols = dims[0]
    s_col = src_strides[0]
    # The destination run is what the DMA writes back to back, so it has to be
    # contiguous; there is no level left for a stride on that side and the
    # transfer would go out packed without a diagnostic. An extent of 1 never
    # indexes the stride at all.
    if cols > 1 and dst_strides[0] != 1:
        return False
    # A strided source run is read per element, which the engine does do -- but
    # only on its own: combining it with a row stride would need that missing
    # third level, so the tile keeps a single row and the rows ride the grid.
    # Stride 0 is not one of the strides it does: an innermost run broadcast from
    # a single element comes back as consecutive elements instead, so that layout
    # stays with the caller.
    if cols > 1 and s_col == 0:
        return False
    rows, s_row, t_row = (
        (dims[1], src_strides[1], dst_strides[1]) if rank > 1 else (1, 0, 0)
    )
    outer_rank = max(rank - 2, 0)
    outer_numel = math.prod(dims[2:])

    # The two sides disagree about which run is contiguous: transpose the tile on
    # chip rather than reading one side per element.
    if cols > 1 and s_col != 1 and rows > 1 and s_row == 1:
        col_blocks = triton.cdiv(cols, DSA_TRANS_TILE)
        grid = (col_blocks * outer_numel, triton.cdiv(rows, DSA_TRANS_TILE))
        _tle_dsa_trans_copy_kernel[grid](
            src_v,
            dst_v,
            rows,
            cols,
            s_col,
            t_row,
            col_blocks,
            *dims[2:],
            *src_strides[2:],
            *dst_strides[2:],
            outer_rank,
            DSA_TRANS_TILE,
            src_ty,
            dst_ty,
            convert,
            dst.dtype == torch.bool,
            is_sdnn=True,
            num_stages=2,
        )
        logger.debug(
            "GEMS_KUNLUNXIN TLE_COPY dsa transpose rows=%d cols=%d tile=%d rank=%d "
            "convert=%s",
            rows,
            cols,
            DSA_TRANS_TILE,
            rank,
            convert,
        )
        return True

    # A strided source run is otherwise read per element, which the engine does
    # do -- but only on its own: combining it with a row stride would need that
    # missing third level, so the tile keeps a single row and the rows ride the
    # grid.
    col_strided = cols > 1 and s_col != 1

    rows_block, cols_block = _dsa_tile(rows, cols, src_v.element_size())
    if col_strided:
        rows_block = 1
    row_blocks = triton.cdiv(rows, rows_block)
    grid = (row_blocks * outer_numel, triton.cdiv(cols, cols_block))
    _tle_dsa_row_copy_kernel[grid](
        src_v,
        dst_v,
        rows,
        cols,
        s_row,
        t_row,
        s_col,
        row_blocks,
        *dims[2:],
        *src_strides[2:],
        *dst_strides[2:],
        outer_rank,
        rows_block,
        cols_block,
        col_strided,
        src_ty,
        dst_ty,
        convert,
        dst.dtype == torch.bool,
        is_sdnn=True,
        num_stages=2,
    )
    logger.debug(
        "GEMS_KUNLUNXIN TLE_COPY dsa rows=%d cols=%d tile=%dx%d rank=%d convert=%s "
        "col_strided=%s",
        rows,
        cols,
        rows_block,
        cols_block,
        rank,
        convert,
        col_strided,
    )
    return True
