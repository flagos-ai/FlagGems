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
"""Kunlunxin (XPU) override for ``aten._pad_circular``.

The generic ``flag_gems.ops._pad_circular`` kernels (1D/2D/3D) fail on XPU:

* The 2D kernel (2D ``(BLOCK_H, BLOCK_W)`` tile, ``rows[:, None] * Wo`` index
  arithmetic and a 2D load/store mask) cannot be converted to the XPU
  local-memory model: ``TritonXPULegalize`` aborts with
  ``'arith.muli' op requires the same type for all operands and results``
  (``loc("src_base")``), i.e. the same 2D-tile / uni_sram wall as
  ``linalg_eig`` -- even ``n=2`` fails.
* The 1D/3D kernels wrap with a two-step ``tl.where`` select
  (``sw = w - plw`` then ``+W`` / ``-W``). On XPU the data-dependent select is
  miscompiled: within each 8-lane group only the first two elements are
  correct and the rest read garbage, so every padded position is wrong (the
  test compares with ``assert_close(atol=0, rtol=0)``).

Correctness fix (retained below): compute the circular source index
*arithmetically* instead of by select. ``aten._pad_circular`` crops first and
then wraps the *cropped* array, so with ``xc = x[s0 : s0 + nc]`` (``s0`` is
the left crop, ``nc`` the cropped extent) the output is
``[xc[nc-sl:], xc, xc[:sr]]`` and the source index is

    sw = s0 + (w + nc - sl) % nc

which is non-negative for ``w >= 0`` and ``sl <= nc`` (validated on the host,
same contract as aten), so the modulo is exact, never touches the
negative-index or data-dependent-select paths, and always yields ``sw`` in
``[0, nc)`` -- the load is in-bounds even for masked-off lanes.

Performance (measured on KL3, 2026-11-20): the ``% nc`` in the address makes it
non-affine, so the whole load lowers to a *discrete gather*
(``triton_xpu.gm2lm`` with ``fixedStride = INT_MIN``) at ~5 GB/s: 0.02-0.21x
of ``aten`` on the benchmark matrix. A strided-row copy cannot be fixed by
tile tuning either -- the cluster path bills ~0.05 us per *program*, so a
row-per-program kernel needs >= 16K elements per program to be competitive,
which a row-strided copy cannot express affinely. The fast path therefore
moves every segment with the DMA engine (``tle.dsa`` 2D descriptor), where one
descriptor covers thousands of rows:

* interior (width-interior of all rows)                  -- 1 copy
* D/H wrap bands copied from the output itself (full rows/planes), in launch
  order after the interior                              -- 2 per outer padded dim
* width wrap strips copied from the output (or from ``x`` for 1D pads and
  right-crop)                                            -- 2 copies

i.e. 3 (1D) / 5 (2D) / 7 (3D) launches, each with a >64 KB uni_sram tile
(4x the ``tle_copy`` default, measured 1.2-1.3x on these shapes) and a cached
flat launcher (the plain ``kernel[grid](...)`` path costs ~48 us of host time
per call here, which would dominate every small shape).

The generic kernels below are kept as the fallback for builds without the tle
DMA path and for dtype views the DMA cannot express; both paths work on the
cropped view, so 8/16-byte dtypes (int64, complex) are moved as 4-byte words.

ATen dispatch reaches this vendor function through the
``SpecOpRegistrar``/``_FULL_CONFIG`` mapping; ``flag_gems._pad_circular`` is
therefore this function on XPU and tests must call it via
``flag_gems._pad_circular``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from ..utils.tle_copy import tle_dma_available

try:  # tle.dsa is the DMA engine; absent on plain triton builds
    import triton.experimental.tle.language as tle
    from triton.runtime import driver

    _HAS_DSA = hasattr(tle, "dsa") and all(
        hasattr(tle.dsa, a) for a in ("UNI_SRAM", "alloc", "copy", "to_tensor")
    )
except ImportError:  # pragma: no cover - depends on the triton build
    tle = None
    driver = None
    _HAS_DSA = False

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# generic (fallback) kernels: correct on XPU, but the per-lane `% nc` turns the
# load into a discrete gather, so they are only used when the DMA path is out.
# They index the *original* (contiguous) tensor: `W`/`H`/`D` are the original
# extents used as strides, `nc`/`s0`/`sl` describe the cropped source range.
# --------------------------------------------------------------------------
@triton.jit
def _pad_circular_1d_kernel(
    x_ptr,
    out_ptr,
    W,
    nc,
    Wo,
    s0,
    sl,
    BLOCK_W: tl.constexpr,
):
    b = tl.program_id(0)
    wblk = tl.program_id(1)
    w = wblk * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = w < Wo
    # circular wrap of the last dim over the cropped block; non-negative dividend
    sw = s0 + (w + nc - sl) % nc
    src = b * W + sw
    vals = tl.load(x_ptr + src, mask=mask)
    tl.store(out_ptr + b * Wo + w, vals, mask=mask)


@triton.jit
def _pad_circular_2d_kernel(
    x_ptr,
    out_ptr,
    H,
    W,
    Hc,
    nc,
    Ho,
    Wo,
    s0h,
    slh,
    s0w,
    slw,
    BLOCK_W: tl.constexpr,
):
    # rows enumerate the collapsed (batch, Ho) planes; one program per output
    # row, width vectorized (scalar-hoist pattern that legalizes on XPU).
    row = tl.program_id(0)
    wblk = tl.program_id(1)

    b = row // Ho
    h = row % Ho

    sh = s0h + (h + Hc - slh) % Hc
    src_base = (b * H + sh) * W

    w = wblk * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = w < Wo
    sw = s0w + (w + nc - slw) % nc
    src = src_base + sw
    vals = tl.load(x_ptr + src, mask=mask)
    tl.store(out_ptr + row * Wo + w, vals, mask=mask)


@triton.jit
def _pad_circular_3d_kernel(
    x_ptr,
    out_ptr,
    D,
    H,
    W,
    Dc,
    Hc,
    nc,
    Do,
    Ho,
    Wo,
    s0d,
    sld,
    s0h,
    slh,
    s0w,
    slw,
    BLOCK_W: tl.constexpr,
):
    # rows enumerate the collapsed (batch, Do, Ho) planes; one program per
    # output row, width vectorized (scalar-hoist pattern that legalizes on
    # XPU). Depth/height wrap is scalar per program, width is vectorized.
    row = tl.program_id(0)
    wblk = tl.program_id(1)

    b = row // (Do * Ho)
    rem = row % (Do * Ho)
    d = rem // Ho
    h = rem % Ho

    sd = s0d + (d + Dc - sld) % Dc
    sh = s0h + (h + Hc - slh) % Hc
    src_base = ((b * D + sd) * H + sh) * W

    w = wblk * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = w < Wo
    sw = s0w + (w + nc - slw) % nc
    src = src_base + sw
    vals = tl.load(x_ptr + src, mask=mask)
    tl.store(out_ptr + row * Wo + w, vals, mask=mask)


# --------------------------------------------------------------------------
# DMA fast path: every output segment is a strided 2D copy, moved by the DMA
# engine through a uni_sram tile.
# --------------------------------------------------------------------------
@triton.jit(
    do_not_specialize=[
        "src_ptr",
        "dst_ptr",
        "src_off",
        "dst_off",
        "rows",
        "cols",
        "s_row",
        "t_row",
        "row_blocks",
        "d2",
        "s2",
        "t2",
        "d3",
        "s3",
        "t3",
    ]
)
def _pad_circular_dma_kernel(
    src_ptr,
    dst_ptr,
    src_off,
    dst_off,
    rows,
    cols,
    s_row,
    t_row,
    row_blocks,
    d2,
    s2,
    t2,
    d3,
    s3,
    t3,
    OUTER_RANK: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    DT: tl.constexpr,
):
    """``dst[dst_off + i*t_row + c] = src[src_off + i*s_row + c]``.

    ``rows`` runs are copied per outer index (``d2``/``d3`` strides), ``cols``
    contiguous elements per run. One 2D DMA in, one out per program; the tail
    block is bounded by ``sizes`` (a masked load hides its extent from the dsa
    rewrite, which then rejects the kernel).
    """
    pid = tl.program_id(0)
    row0 = (pid % row_blocks) * ROWS
    outer = pid // row_blocks
    col0 = tl.program_id(1) * COLS

    src_base = src_off
    dst_base = dst_off
    if OUTER_RANK > 0:
        i = outer % d2
        outer = outer // d2
        src_base += i * s2
        dst_base += i * t2
    if OUTER_RANK > 1:
        i = outer % d3
        src_base += i * s3
        dst_base += i * t3

    r = row0 + tl.arange(0, ROWS)
    c = col0 + tl.arange(0, COLS)
    row_tail = tl.minimum(rows - row0, ROWS)
    col_tail = tl.minimum(cols - col0, COLS)

    buf = tle.dsa.alloc([ROWS, COLS], DT, tle.dsa.UNI_SRAM)
    src = src_ptr + src_base + r[:, None] * s_row + c[None, :]
    tle.dsa.copy(src, buf, sizes=[row_tail, col_tail])
    dst = dst_ptr + dst_base + r[:, None] * t_row + c[None, :]
    tle.dsa.copy(buf, dst, sizes=[row_tail, col_tail])


@triton.jit(
    do_not_specialize=[
        "src_ptr",
        "dst_ptr",
        "src_off_l",
        "dst_off_l",
        "src_off_r",
        "dst_off_r",
        "rows",
        "s_row",
        "t_row",
    ]
)
def _pad_circular_strips_kernel(
    src_ptr,
    dst_ptr,
    src_off_l,
    dst_off_l,
    src_off_r,
    dst_off_r,
    rows,
    s_row,
    t_row,
    SL: tl.constexpr,
    SR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Both width-wrap strips of every row, one lane per row.

    A strip is only a few elements wide, so a 2D DMA would issue one
    ``cols``-element run per row: measured ~2.2 us per 1000 one-element rows
    against 0.23 ns/element for a plain gather/scatter, i.e. 5-10x worse on the
    3D shapes. The static column loop keeps the address affine per iteration;
    only the three row-width dtypes ride in ``SL``/``SR``.
    """
    pid = tl.program_id(0)
    r = pid * BLOCK + tl.arange(0, BLOCK)
    m = r < rows
    sb = r * s_row
    db = r * t_row
    for j in tl.static_range(SL):
        v = tl.load(src_ptr + src_off_l + sb + j, mask=m)
        tl.store(dst_ptr + dst_off_l + db + j, v, mask=m)
    for j in tl.static_range(SR):
        v = tl.load(src_ptr + src_off_r + sb + j, mask=m)
        tl.store(dst_ptr + dst_off_r + db + j, v, mask=m)


# What the DMA moves: a pure move only cares about the width. 4 bytes is
# spelled float32 because int32 faults in the engine (see tle_copy).
_MOVE_DTYPE = {1: torch.int8, 2: torch.int16, 4: torch.float32}
_TL_DTYPE = {
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.float32: tl.float32,
}

# Tile budget per program. 32 KB (the tle_copy default) leaves 20-30% on the
# table for every shape in the benchmark matrix; 256 KB stops paying.
_TILE_BYTES = 128 * 1024
# Cache: (device, dtype, ROWS, COLS, rank, grid) -> flat launcher.
_DMA_PLANS = {}
_STRIP_PLANS = {}
# A one-element-wide strip costs the DMA one 2-byte run per row (~2.2 us per
# 1000 rows); a per-row gather kernel is cheaper there. Wider strips measured
# *slower* as a gather than as a DMA, so the gate stays at one column a side.
_STRIP_MAX_ELEMS = 150000
_STRIP_MAX_COLS = 2


def _npo2(v):
    v = int(v)
    if v <= 1:
        return 1
    return 1 << (v - 1).bit_length()


def _pick_tile(rows, cols, esize):
    # tl.arange widens a sub-64 extent silently (and the widened lanes break
    # the `sizes`-bounded DMA), so both tile dims stay >= 64; `sizes` caps the
    # real transfer, the extra lanes are never accessed.
    cols_b = max(min(_npo2(cols), max(8192 // esize, 64)), 64)
    row_budget = max(_TILE_BYTES // (cols_b * esize), 1)
    rows_b = max(min(_npo2(rows), _npo2(row_budget)), 64)
    return rows_b, cols_b


def _move_dtype(t):
    """(view tensor, view/source width ratio) for a pure element move."""
    esize = t.element_size()
    if esize in _MOVE_DTYPE:
        mv = _MOVE_DTYPE[esize]
        return (t if t.dtype == mv else t.view(mv)), 1
    # 8/16-byte elements move as two/four 4-byte words
    return (t if t.dtype == torch.float32 else t.view(torch.float32)), esize // 4


def _dma_copy(src, dst, dst_off, cols, rows, s_row, t_row, outer, src_off=0):
    """Launch one strided copy; ``outer`` is ((extent, s_stride, t_stride), ...)
    innermost-first, every quantity in elements of the (view) dtype."""
    if rows <= 0 or cols <= 0:
        return
    dev = dst.device.index
    rows_b, cols_b = _pick_tile(rows, cols, src.element_size())
    row_blocks = (rows + rows_b - 1) // rows_b
    outer_numel = 1
    for extent, _, _ in outer:
        outer_numel *= extent
    grid = (row_blocks * outer_numel, (cols + cols_b - 1) // cols_b)

    rank = len(outer)
    dims = list(outer) + [(1, 0, 0)] * (2 - rank)
    key = (dev, src.dtype, rows_b, cols_b, rank, grid)
    launch = _DMA_PLANS.get(key)
    args = (
        src.data_ptr(),
        dst.data_ptr(),
        src_off,
        dst_off,
        rows,
        cols,
        s_row,
        t_row,
        row_blocks,
        dims[0][0],
        dims[0][1],
        dims[0][2],
        dims[1][0],
        dims[1][1],
        dims[1][2],
    )
    if launch is None:
        with torch_device_fn.device(dst.device):
            kernel = _pad_circular_dma_kernel[grid](
                src,
                dst,
                src_off,
                dst_off,
                rows,
                cols,
                s_row,
                t_row,
                row_blocks,
                *dims[0],
                *dims[1],
                OUTER_RANK=rank,
                ROWS=rows_b,
                COLS=cols_b,
                DT=_TL_DTYPE[src.dtype],
                is_sdnn=True,
                num_stages=2,
            )
        launch = driver.active.make_flat_launcher(kernel, grid)
        _DMA_PLANS[key] = launch
        return
    launch(driver.active.get_current_stream(dev), *args)


def _strip_copy(
    src, dst, src_off_l, dst_off_l, src_off_r, dst_off_r, rows, s_row, t_row, sl, sr
):
    """Both width strips in one launch; ``sl``/``sr`` are the (small) widths."""
    dev = dst.device.index
    block = 1024
    grid = ((rows + block - 1) // block,)
    key = (dev, src.dtype, sl, sr, block, grid)
    args = (
        src.data_ptr(),
        dst.data_ptr(),
        src_off_l,
        dst_off_l,
        src_off_r,
        dst_off_r,
        rows,
        s_row,
        t_row,
    )
    launch = _STRIP_PLANS.get(key)
    if launch is None:
        with torch_device_fn.device(dst.device):
            kernel = _pad_circular_strips_kernel[grid](
                src,
                dst,
                src_off_l,
                dst_off_l,
                src_off_r,
                dst_off_r,
                rows,
                s_row,
                t_row,
                SL=sl,
                SR=sr,
                BLOCK=block,
            )
        launch = driver.active.make_flat_launcher(kernel, grid)
        _STRIP_PLANS[key] = launch
        return
    launch(driver.active.get_current_stream(dev), *args)


def _dsa_pad_circular(x, out, segs, ndim, npad):
    """Fill ``out`` with DMA copies; returns False if this dtype/shape is out
    of the DMA path (the caller then runs the generic kernels)."""
    if not (_HAS_DSA and tle_dma_available()):
        return False
    try:
        src_v, ratio = _move_dtype(x)
        dst_v, _ = _move_dtype(out)
    except RuntimeError:  # dtype view needs a stride-1 last dim
        return False

    n = [x.shape[-1 - i] for i in range(npad)]
    nc = [s[1] for s in segs]
    sl = [s[2] for s in segs]
    sr = [s[3] for s in segs]
    s0 = [s[0] for s in segs]
    no = [out.shape[-1 - i] for i in range(npad)]

    # Only the innermost dim is scaled by the dtype view; the others keep
    # their extents while their strides scale with ``ratio``. The strides are
    # the *original* extents (the source is the un-sliced contiguous tensor;
    # crops only shift the copy offsets), so the innermost stride is ``n0v``.
    n0v, no0v = n[0] * ratio, no[0] * ratio
    nc0v = nc[0] * ratio
    sl0v, sr0v = sl[0] * ratio, sr[0] * ratio
    s00v = s0[0] * ratio
    n1 = n[1] if npad >= 2 else 1
    n2 = n[2] if npad >= 3 else 1
    no1 = no[1] if npad >= 2 else 1
    no2 = no[2] if npad >= 3 else 1
    row_in, row_out = n0v, no0v
    plane_in, plane_out = n0v * n1, no0v * no1
    s0_row = s0[1] * row_in if npad >= 2 else 0
    s0_plane = s0[2] * plane_in if npad >= 3 else 0

    lead = 1
    for s in x.shape[:-npad]:
        lead *= s
    lead_out = 1
    for s in out.shape[:-npad]:
        lead_out *= s
    if lead != lead_out:
        return False

    # ---- interior: the cropped block -> out's interior block
    if npad == 1:
        _dma_copy(src_v, dst_v, sl0v, nc0v, lead, row_in, row_out, (), src_off=s00v)
    else:
        dst_off = sl[1] * row_out + sl0v
        src_off = s0_row + s00v
        if npad == 2:
            _dma_copy(
                src_v,
                dst_v,
                dst_off,
                nc0v,
                nc[1],
                row_in,
                row_out,
                ((lead, plane_in, plane_out),),
                src_off=src_off,
            )
        else:
            dst_off += sl[2] * plane_out
            src_off += s0_plane
            _dma_copy(
                src_v,
                dst_v,
                dst_off,
                nc0v,
                nc[1],
                row_in,
                row_out,
                (
                    (nc[2], plane_in, plane_out),
                    (lead, n2 * plane_in, no2 * plane_out),
                ),
                src_off=src_off,
            )

    # ---- depth wrap: whole output planes copied from the interior planes
    lead_out_stride = no2 * plane_out
    if npad == 3:
        if sl[2] > 0:
            _dma_copy(
                dst_v,
                dst_v,
                0,
                no0v,
                sl[2] * no1,
                row_out,
                row_out,
                ((lead, lead_out_stride, lead_out_stride),),
                src_off=nc[2] * plane_out,
            )
        if sr[2] > 0:
            _dma_copy(
                dst_v,
                dst_v,
                (sl[2] + nc[2]) * plane_out,
                no0v,
                sr[2] * no1,
                row_out,
                row_out,
                ((lead, lead_out_stride, lead_out_stride),),
                src_off=sl[2] * plane_out,
            )

    # ---- height wrap: full output rows copied from the interior rows
    if npad >= 2:
        planes_out = lead * no2
        if sl[1] > 0:
            _dma_copy(
                dst_v,
                dst_v,
                0,
                no0v,
                sl[1],
                row_out,
                row_out,
                ((planes_out, plane_out, plane_out),),
                src_off=nc[1] * row_out,
            )
        if sr[1] > 0:
            _dma_copy(
                dst_v,
                dst_v,
                (sl[1] + nc[1]) * row_out,
                no0v,
                sr[1],
                row_out,
                row_out,
                ((planes_out, plane_out, plane_out),),
                src_off=sl[1] * row_out,
            )

    # ---- width wrap
    if npad == 1:
        rows_strip = lead
        s_row = row_in
        src_off_l, src_off_r = s00v + nc0v - sl0v, s00v
    else:
        rows_strip = lead * no2 * no1
        s_row = row_out
        src_off_l, src_off_r = nc0v, sl0v
    if sl[0] > 0 or sr[0] > 0:
        if (
            sl[0] + sr[0] <= _STRIP_MAX_COLS
            and rows_strip * (sl[0] + sr[0]) <= _STRIP_MAX_ELEMS
        ):
            _strip_copy(
                dst_v if npad >= 2 else src_v,
                dst_v,
                src_off_l,
                0,
                src_off_r,
                no0v - sr0v,
                rows_strip,
                s_row,
                row_out,
                sl0v,
                sr0v,
            )
        else:
            if sl[0] > 0:
                _dma_copy(
                    src_v if npad == 1 else dst_v,
                    dst_v,
                    0,
                    sl0v,
                    rows_strip,
                    s_row,
                    row_out,
                    (),
                    src_off=src_off_l,
                )
            if sr[0] > 0:
                _dma_copy(
                    src_v if npad == 1 else dst_v,
                    dst_v,
                    no0v - sr0v,
                    sr0v,
                    rows_strip,
                    s_row,
                    row_out,
                    (),
                    src_off=src_off_r,
                )
    return True


def _generic_pad_circular(x, out, segs, ndim, npad):
    """Per-row gather fallback over the original (contiguous) tensor."""
    src, ratio = _move_dtype(x)
    dst, _ = _move_dtype(out)
    n = [x.shape[-1 - i] for i in range(npad)]
    nc = [s[1] for s in segs]
    sl = [s[2] for s in segs]
    no = [out.shape[-1 - i] for i in range(npad)]
    n0v, nc0v, no0v = n[0] * ratio, nc[0] * ratio, no[0] * ratio

    with torch_device_fn.device(x.device):
        if npad == 1:
            W, Wo = n0v, no0v
            B = 1
            for s in src.shape[:-1]:
                B *= s
            BLOCK_W = 1024
            grid = (B, triton.cdiv(Wo, BLOCK_W))
            _pad_circular_1d_kernel[grid](
                src,
                dst,
                W,
                nc0v,
                Wo,
                segs[0][0] * ratio,
                sl[0] * ratio,
                BLOCK_W=BLOCK_W,
            )
            return
        if npad == 2:
            H, W = n[1], n0v
            Hc, Wc = nc[1], nc0v
            Ho, Wo = no[1], no0v
            B = 1
            for s in src.shape[:-2]:
                B *= s
            R = B * Ho
            BLOCK_W = min(1024, triton.next_power_of_2(max(Wo, 1)))
            grid = (R, triton.cdiv(Wo, BLOCK_W))
            _pad_circular_2d_kernel[grid](
                src,
                dst,
                H,
                W,
                Hc,
                Wc,
                Ho,
                Wo,
                segs[1][0],
                sl[1],
                segs[0][0] * ratio,
                sl[0] * ratio,
                BLOCK_W=BLOCK_W,
            )
            return
        D, H, W = n[2], n[1], n0v
        Dc, Hc, Wc = nc[2], nc[1], nc0v
        Do, Ho, Wo = no[2], no[1], no0v
        B = 1
        for s in src.shape[:-3]:
            B *= s
        R = B * Do * Ho
        BLOCK_W = min(1024, triton.next_power_of_2(max(Wo, 1)))
        grid = (R, triton.cdiv(Wo, BLOCK_W))
        _pad_circular_3d_kernel[grid](
            src,
            dst,
            D,
            H,
            W,
            Dc,
            Hc,
            Wc,
            Do,
            Ho,
            Wo,
            segs[2][0],
            sl[2],
            segs[1][0],
            sl[1],
            segs[0][0] * ratio,
            sl[0] * ratio,
            BLOCK_W=BLOCK_W,
        )


def _segs(pad, in_shape, ndim, npad):
    """Per padded dim, innermost first: (crop_left, cropped_extent, left, right).

    ``aten._pad_circular`` crops first and then wraps the *cropped* array, so
    the output is ``[xc[nc-sl:], xc, xc[:sr]]`` with ``xc = x[s0 : s0 + nc]``.
    Wrap validation is deferred to the caller so that empty outputs (a zero
    dim anywhere) can short-circuit first, like aten.
    """
    segs = []
    out_shape = list(in_shape)
    for i in range(npad):
        d = ndim - 1 - i
        n = in_shape[d]
        pl, pr = int(pad[2 * i]), int(pad[2 * i + 1])
        new_size = n + pl + pr
        if new_size < 0:
            raise ValueError(
                "negative padding removed the whole dimension "
                f"{d}: input {n} pad ({pl}, {pr})"
            )
        s0, e0 = max(0, -pl), max(0, -pr)
        sl, sr = max(0, pl), max(0, pr)
        out_shape[d] = sl + (n - s0 - e0) + sr
        segs.append((s0, n - s0 - e0, sl, sr))
    return segs, out_shape


def _pad_circular(x: torch.Tensor, pad):
    logger.debug("GEMS_KUNLUNXIN _PAD_CIRCULAR")

    ndim = x.dim()
    if len(pad) % 2 != 0:
        raise ValueError("padding length must be even")
    npad = len(pad) // 2
    if npad < 1 or npad > 3:
        raise ValueError(
            "flag_gems _pad_circular supports padding the last 1, 2 or 3 dimensions"
        )
    if ndim <= npad:
        raise ValueError("input must have at least one non-padded (batch) dimension")

    x = x.contiguous()
    segs, out_shape = _segs(pad, list(x.shape), ndim, npad)

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    total = 1
    for s in out_shape:
        total *= s
    if total == 0:
        return out
    for s0, nc, sl, sr in segs:
        if nc <= 0 or sl > nc or sr > nc:
            # aten is undefined here and raises for most shapes; reject explicitly
            raise ValueError("Padding value causes wrapping around more than once.")

    try:
        if _dsa_pad_circular(x, out, segs, ndim, npad):
            return out
    except Exception:  # noqa: BLE001 - tle build without a usable dma path
        logger.debug("GEMS_KUNLUNXIN _PAD_CIRCULAR dma path unavailable", exc_info=True)

    _generic_pad_circular(x, out, segs, ndim, npad)
    return out
