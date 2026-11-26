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
"""Kunlunxin(XPU) override for ``aten._pad_enum``.

The generic ``flag_gems.ops._pad_enum`` implementation walks every *output*
element and recovers its multi-dimensional coordinate with per-element
``//``/``%`` (divmod) against the output strides, then gathers the input with a
data-dependent (discrete) offset vector.  On XPU both halves of that are slow:
the divmod chain is long and the gather degenerates to discrete memory access
(measured ~20-30 GB/s on the benchmark shapes).

Constant padding with non-negative pads has a much more regular structure: the
output is ``value`` everywhere except a strided copy of the input, and every
output row along the last (contiguous) dimension is

    [ value ... | input-row (contiguous) | value ... ]

So it can be produced by (a) filling the output with ``value`` and (b) copying
each contiguous input row into its place.  Both passes are pure contiguous
memory traffic with **prefix masks only** -- which matters a lot on this
backend:

* A masked memory op whose mask is a single ``offs < n`` comparison is lowered
  to one contiguous DMA of ``n - offs0`` elements.
* Any mask built from *two or more* comparisons (e.g. ``(jl >= 0) & (jl < n)``)
  is not lowered that way: measured bandwidth collapses from ~600 GB/s to
  ~6.6 GB/s, and for a load the "length" rewrite can even produce wrong data.

Hence the copy kernel only ever masks with ``offs < in_last`` and derives the
row base offsets from the row index (one divmod per *row*, not per element).

The copy is still the weak spot.  This backend only lowers a memory op to a
contiguous DMA chunk when the chunk base is *provably* a multiple of 8 elements
(``arange`` plus a non-zero residue falls back to the scalar path: measured
~140 GB/s vs ~620 GB/s for an aligned store, and striding alone is harmless as
long as the base stays provably aligned).  The padded output row start
(``row * out_last + pad_last``) generally is not 8-aligned, and input/output row
phases differ, so the store cannot be aligned on both sides.  aten's native pad
reaches HBM peak (~1.9 TB/s) on the same shapes.  The fast path is therefore
used only where the fill+copy structure clearly wins over the generic kernel
(long rows, or very few rows to copy); everything else keeps the generic kernel.

All other modes (reflect / replicate / circular) and constant padding with a
negative pad (which crops and cannot be expressed as a row copy) always use the
verbatim generic kernel, kept below so this override is self-contained and
correct for every mode.
"""

import logging
import os
import struct

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


try:  # hand-written-payload injection point (XPU cluster path)
    import triton.experimental.tle as tle

    _HAS_TLE = True
except Exception:  # pragma: no cover - toolchain without tle.raw
    tle = None
    _HAS_TLE = False


# ---------------------------------------------------------------------------
# Hand-written `.xpu` payload (cluster path; flag_gems' own kernel, not a call
# into xdnn/native).
#
# The plain-Triton fast path is stuck on a backend wall: a `tl.store` is only
# lowered to a contiguous DMA when its base is *provably* a multiple of 8
# elements, and the padded output row start `row * out_last + pad_last` never
# is, so the interior copy degenerates to the scalar path (~140 GB/s vs
# ~620 GB/s for an aligned store).  A hand-written payload stages the row
# through LM and writes it back with a plain byte-range `LM2GM`, which accepts
# *any* global offset -- so the padded output row leaves in one contiguous
# store.
#
# Work item = (output row, LM column chunk of `CB` elements):
#   * the LM chunk is pre-filled with the pad value, but only over the gaps
#     (the region the input DMA does not cover) -- scalar element stores only,
#     deliberately no SIMD: `__local__` is 32-byte aligned while a 64-byte
#     vector store needs `__simd__`, and a misaligned one wedges the card,
#   * the overlapping slice of the input row is DMA'd into its (arbitrary)
#     slot inside the chunk,
#   * the whole chunk leaves with a single `LM2GM` to `row * out_last + c0`.
#
# The output-row -> input-row mapping is an odometer over the outer dims; dims
# beyond `ndim-1` carry shape 1 / pad 0 so the loops are fully static (no
# runtime trip count).  Grid is one cluster row (`grid=(12,)` == 12 clusters x
# 64 cores == 768 cores); a single cluster would only expose 64 cores.
# ---------------------------------------------------------------------------

_PAYLOAD_OBJ = os.path.join(
    os.path.dirname(__file__), "..", "payload", "obj", "padenum.o"
)

_XPU_GRID = 12  # clusters on this device (12 x 64 = 768 cores)
_XPU_DNS = [
    "VBITS",
    "ROUT",
    "OUTL",
    "INL",
    "PL",
    "OD0",
    "OD1",
    "OD2",
    "OD3",
    "ID0",
    "ID1",
    "ID2",
    "ID3",
    "PL0",
    "PL1",
    "PL2",
    "PL3",
]

_XPU_KERNELS = {}
if _HAS_TLE:
    try:

        @tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
        def pe_f32(
            x,
            y,
            vbits,
            rout,
            outl,
            inl,
            pl,
            od0,
            od1,
            od2,
            od3,
            id0,
            id1,
            id2,
            id3,
            pl0,
            pl1,
            pl2,
            pl3,
        ): ...

        @tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
        def pe_f16(
            x,
            y,
            vbits,
            rout,
            outl,
            inl,
            pl,
            od0,
            od1,
            od2,
            od3,
            id0,
            id1,
            id2,
            id3,
            pl0,
            pl1,
            pl2,
            pl3,
        ): ...

        @tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
        def pe_bf16(
            x,
            y,
            vbits,
            rout,
            outl,
            inl,
            pl,
            od0,
            od1,
            od2,
            od3,
            id0,
            id1,
            id2,
            id3,
            pl0,
            pl1,
            pl2,
            pl3,
        ): ...

        @triton.jit(do_not_specialize=_XPU_DNS)
        def _pe_raw_f32(
            X,
            Y,
            VBITS,
            ROUT,
            OUTL,
            INL,
            PL,
            OD0,
            OD1,
            OD2,
            OD3,
            ID0,
            ID1,
            ID2,
            ID3,
            PL0,
            PL1,
            PL2,
            PL3,
        ):
            tle.raw.call(
                pe_f32,
                (
                    X,
                    Y,
                    VBITS,
                    ROUT,
                    OUTL,
                    INL,
                    PL,
                    OD0,
                    OD1,
                    OD2,
                    OD3,
                    ID0,
                    ID1,
                    ID2,
                    ID3,
                    PL0,
                    PL1,
                    PL2,
                    PL3,
                ),
            )

        @triton.jit(do_not_specialize=_XPU_DNS)
        def _pe_raw_f16(
            X,
            Y,
            VBITS,
            ROUT,
            OUTL,
            INL,
            PL,
            OD0,
            OD1,
            OD2,
            OD3,
            ID0,
            ID1,
            ID2,
            ID3,
            PL0,
            PL1,
            PL2,
            PL3,
        ):
            tle.raw.call(
                pe_f16,
                (
                    X,
                    Y,
                    VBITS,
                    ROUT,
                    OUTL,
                    INL,
                    PL,
                    OD0,
                    OD1,
                    OD2,
                    OD3,
                    ID0,
                    ID1,
                    ID2,
                    ID3,
                    PL0,
                    PL1,
                    PL2,
                    PL3,
                ),
            )

        @triton.jit(do_not_specialize=_XPU_DNS)
        def _pe_raw_bf16(
            X,
            Y,
            VBITS,
            ROUT,
            OUTL,
            INL,
            PL,
            OD0,
            OD1,
            OD2,
            OD3,
            ID0,
            ID1,
            ID2,
            ID3,
            PL0,
            PL1,
            PL2,
            PL3,
        ):
            tle.raw.call(
                pe_bf16,
                (
                    X,
                    Y,
                    VBITS,
                    ROUT,
                    OUTL,
                    INL,
                    PL,
                    OD0,
                    OD1,
                    OD2,
                    OD3,
                    ID0,
                    ID1,
                    ID2,
                    ID3,
                    PL0,
                    PL1,
                    PL2,
                    PL3,
                ),
            )

        _XPU_KERNELS = {
            torch.float32: _pe_raw_f32,
            torch.float16: _pe_raw_f16,
            torch.bfloat16: _pe_raw_bf16,
        }

    except Exception:  # pragma: no cover - toolchain without the tle.raw object channel
        _XPU_KERNELS = {}
# aten padding_mode enum -> readable name (see aten/src/ATen/native/PadNd.cpp)
_MODE_REFLECT = 0
_MODE_REPLICATE = 1
_MODE_CIRCULAR = 2
_MODE_CONSTANT = 3

# Fill pass tile and copy pass tile bounds (fixed & bounded -> stable codegen).
_FILL_BLOCK = 8192
_COPY_BLOCK_MIN = 256
_COPY_BLOCK_MAX = 8192
# The constant fast path pays ~150ns per copied row (one DMA chunk pair) plus
# one fill pass, while the generic path pays a per-element divmod + discrete
# gather.  It therefore wins when either the rows are long (one chunk amortises
# well) or there are very few rows to copy; tiny outputs are left to the
# generic kernel where the extra launch would dominate.
_MIN_IN_LAST = 512
_MAX_ROWS_WHEN_SHORT = 256
_MIN_NUMEL_FAST = 32768
# Below this the extra `.xpu` launch is not worth it; the generic kernel is
# already competitive on tiny outputs.
_MIN_NUMEL_XPU = 8192
# The payload pays a fixed per-(row, chunk) cost (~1.6us on this part), so a row
# whose output span is only a handful of elements is pure latency.  Rows below
# this many bytes go to the generic kernel instead (measured: `(10000, 1)` with
# `out_last == 7` regressed 17.4us -> 22.0us at 28 bytes/row).
_MIN_ROW_BYTES_XPU = 128


@triton.jit
def _pad_enum_general_kernel(
    x_ptr,
    out_ptr,
    numel_out,
    value,
    in_shape: tl.constexpr,
    out_shape: tl.constexpr,
    out_stride: tl.constexpr,
    in_stride: tl.constexpr,
    pad_left: tl.constexpr,
    NDIM: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Verbatim mirror of ``flag_gems.ops._pad_enum._pad_enum_kernel``.  Kept
    # local so this override handles every mode on its own (no delegation).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel_out
    o = offs  # output linear index

    in_offset = tl.zeros((BLOCK,), dtype=tl.int32)
    oob = tl.zeros((BLOCK,), dtype=tl.int1)

    for d in tl.static_range(NDIM):
        c = (o // out_stride[d]) % out_shape[d]  # output coordinate along dim d
        j = c - pad_left[d]  # position relative to input start

        if MODE == 3:  # constant
            dim_oob = (j < 0) | (j >= in_shape[d])
            oob = oob | dim_oob
            in_coord = tl.where(dim_oob, 0, j)
        elif MODE == 1:  # replicate
            in_coord = tl.maximum(0, tl.minimum(j, in_shape[d] - 1))
        elif MODE == 2:  # circular
            r = j % in_shape[d]
            in_coord = tl.where(r < 0, r + in_shape[d], r)
        else:  # MODE == 0, reflect (reflect_101)
            if in_shape[d] == 1:
                in_coord = tl.zeros((BLOCK,), dtype=tl.int32)
            else:
                m = j % (2 * (in_shape[d] - 1))
                m = tl.where(m < 0, m + 2 * (in_shape[d] - 1), m)
                in_coord = tl.where(m >= in_shape[d], 2 * (in_shape[d] - 1) - m, m)

        in_offset += in_coord * in_stride[d]

    load_mask = mask & (~oob)
    x_val = tl.load(x_ptr + in_offset, mask=load_mask, other=0.0)
    val = tl.full((BLOCK,), value, dtype=x_val.dtype)
    result = tl.where(oob, val, x_val)
    tl.store(out_ptr + offs, result, mask=mask)


@triton.jit
def _pad_enum_fill_kernel(out_ptr, value, n, BLOCK: tl.constexpr):
    """Write ``value`` over the whole output.  Single-comparison (prefix) mask."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    val = tl.full((BLOCK,), value, dtype=out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, val, mask=offs < n)


@triton.jit
def _pad_enum_copy_kernel(
    x_ptr,
    out_ptr,
    in_last,
    pad_last,
    in_shape_rev: tl.constexpr,
    in_stride_rev: tl.constexpr,
    out_stride_rev: tl.constexpr,
    pad_left_rev: tl.constexpr,
    N_OUTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy every contiguous input row into its (padded) output row.

    ``row`` enumerates the *input* rows (dims ``0..ndim-2``, row-major).  The
    per-row base offsets are derived with ``O(N_OUTER)`` divmods.  The memory
    ops use a single-comparison prefix mask (``offs < in_last``) so the backend
    lowers them to one contiguous DMA each.
    """
    row = tl.program_id(0)
    col_blk = tl.program_id(1)

    rem = row
    in_row_base = 0
    out_row_base = 0
    # *_rev tuples hold the outer dims from innermost to outermost, which is the
    # row-major decomposition order of `row`.
    for d in tl.static_range(N_OUTER):
        c = rem % in_shape_rev[d]
        rem = rem // in_shape_rev[d]
        in_row_base = in_row_base + c * in_stride_rev[d]
        out_row_base = out_row_base + (c + pad_left_rev[d]) * out_stride_rev[d]

    offs = col_blk * BLOCK + tl.arange(0, BLOCK)
    m = offs < in_last
    v = tl.load(x_ptr + in_row_base + offs, mask=m)
    tl.store(out_ptr + out_row_base + pad_last + offs, v, mask=m)


def _launch_general(inp, out, v, in_shape, out_shape, pad_left, ndim, mode):
    numel_out = out.numel()
    if ndim <= 1:
        block = 512
    elif ndim == 2:
        block = 2048
    else:
        block = 1024
    grid = (triton.cdiv(numel_out, block),)
    _pad_enum_general_kernel[grid](
        inp,
        out,
        numel_out,
        v,
        in_shape=tuple(in_shape),
        out_shape=tuple(out_shape),
        out_stride=tuple(out.stride()),
        in_stride=tuple(inp.stride()),
        pad_left=tuple(pad_left),
        NDIM=ndim,
        MODE=int(mode),
        BLOCK=block,
    )


def _copy_block(in_last):
    b = triton.next_power_of_2(in_last)
    if b < _COPY_BLOCK_MIN:
        return _COPY_BLOCK_MIN
    if b > _COPY_BLOCK_MAX:
        return _COPY_BLOCK_MAX
    return b


def _launch_const(inp, out, v, in_shape, out_shape, pad_left, ndim):
    numel_out = out.numel()
    in_last = in_shape[-1]

    grid_f = (triton.cdiv(numel_out, _FILL_BLOCK),)
    _pad_enum_fill_kernel[grid_f](out, v, numel_out, BLOCK=_FILL_BLOCK)

    n_in_rows = 1
    for d in range(ndim - 1):
        n_in_rows *= in_shape[d]
    if n_in_rows == 0 or in_last == 0:
        return

    if ndim == 1:
        in_shape_rev = (1,)
        in_stride_rev = (0,)
        out_stride_rev = (0,)
        pad_left_rev = (0,)
    else:
        # input & output are contiguous -> C-order element strides
        in_stride_full = [0] * ndim
        out_stride_full = [0] * ndim
        s_in = 1
        s_out = 1
        for d in range(ndim - 1, -1, -1):
            in_stride_full[d] = s_in
            out_stride_full[d] = s_out
            s_in *= in_shape[d]
            s_out *= out_shape[d]
        rng = list(range(ndim - 1))
        in_shape_rev = tuple(reversed([in_shape[d] for d in rng]))
        in_stride_rev = tuple(reversed([in_stride_full[d] for d in rng]))
        out_stride_rev = tuple(reversed([out_stride_full[d] for d in rng]))
        pad_left_rev = tuple(reversed([pad_left[d] for d in rng]))

    block = _copy_block(in_last)
    grid_c = (n_in_rows, triton.cdiv(in_last, block))
    _pad_enum_copy_kernel[grid_c](
        inp,
        out,
        in_last,
        pad_left[-1],
        in_shape_rev=in_shape_rev,
        in_stride_rev=in_stride_rev,
        out_stride_rev=out_stride_rev,
        pad_left_rev=pad_left_rev,
        N_OUTER=ndim - 1,
        BLOCK=block,
    )


def _s32(v):
    """Reinterpret a 32-bit pattern as a signed int32 (the payload takes i32)."""
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _pad_value_bits(value, dtype):
    """32-bit pattern broadcast by the payload's LM fill.

    fp32 -> the float bits; fp16/bf16 -> the 16-bit pattern duplicated in both
    halves, so the same constant can fill 2-byte elements as uint16 or 4-byte
    elements as uint32.
    """
    if dtype == torch.float32:
        return _s32(struct.unpack("<I", struct.pack("<f", float(value)))[0])
    h = int(torch.tensor(float(value), dtype=dtype).view(torch.int16).item())
    h &= 0xFFFF
    return _s32(h | (h << 16))


def _launch_xpu_const(inp, out, v, in_shape, out_shape, pad_left, ndim):
    """Hand-written `.xpu` constant pad (see the payload comment above).

    `n_out` is the number of *rows* (all dims but the last), the odometer runs
    over the outer dims, and unused odometer slots carry shape 1 / pad 0.
    """
    in_last = in_shape[-1]
    out_last = out_shape[-1]
    n_out = 1
    for d in range(ndim - 1):
        n_out *= out_shape[d]
    od = [1, 1, 1, 1]
    ish = [1, 1, 1, 1]
    plo = [0, 0, 0, 0]
    for i in range(ndim - 1):
        od[i] = out_shape[i]
        ish[i] = in_shape[i]
        plo[i] = pad_left[i]
    _XPU_KERNELS[inp.dtype][(_XPU_GRID,)](
        inp,
        out,
        _pad_value_bits(v, inp.dtype),
        n_out,
        out_last,
        in_last,
        pad_left[-1],
        od[0],
        od[1],
        od[2],
        od[3],
        ish[0],
        ish[1],
        ish[2],
        ish[3],
        plo[0],
        plo[1],
        plo[2],
        plo[3],
    )


def _pad_enum(self, pad, mode, value=None):
    logger.debug("GEMS_KUNLUNXIN _PAD_ENUM")

    assert len(pad) % 2 == 0, "Padding length must be divisible by 2"
    ndim = self.dim()
    k = len(pad) // 2
    assert k <= ndim, "Padding length too large"

    inp = self.contiguous()
    in_shape = list(inp.shape)
    out_shape = list(in_shape)
    pad_left = [0] * ndim
    for i in range(k):
        dim = ndim - 1 - i
        left = pad[2 * i]
        right = pad[2 * i + 1]
        out_shape[dim] = in_shape[dim] + left + right
        pad_left[dim] = left

    if mode == _MODE_REFLECT:
        for i in range(k):
            dim = ndim - 1 - i
            assert (
                pad[2 * i] < in_shape[dim] and pad[2 * i + 1] < in_shape[dim]
            ), "Padding size should be less than the corresponding input dimension"
    elif mode == _MODE_CIRCULAR:
        for i in range(k):
            dim = ndim - 1 - i
            assert (
                pad[2 * i] <= in_shape[dim] and pad[2 * i + 1] <= in_shape[dim]
            ), "Padding value causes wrapping around more than once"

    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    numel_out = out.numel()
    v = float(value) if (mode == _MODE_CONSTANT and value is not None) else 0.0

    if numel_out == 0:
        return out

    # Fast path: constant padding with non-negative pads is "fill + copy
    # contiguous input rows".
    n_in_rows = 1
    for d in range(ndim - 1):
        n_in_rows *= in_shape[d]
    use_fast = (
        mode == _MODE_CONSTANT
        and all(p >= 0 for p in pad)
        and numel_out >= _MIN_NUMEL_FAST
        and (in_shape[-1] >= _MIN_IN_LAST or n_in_rows <= _MAX_ROWS_WHEN_SHORT)
    )

    # Hand-written `.xpu` path: constant padding with non-negative pads, rank
    # up to 5, large enough that the extra launch is amortised.  Same algorithm
    # as `_launch_const`, but the row store is a byte-range LM2GM (any
    # alignment) rather than a plain-Triton `tl.store` (needs an 8-element
    # provable alignment), so it does not fall back to the scalar path.
    use_xpu = (
        mode == _MODE_CONSTANT
        and ndim >= 1
        and ndim <= 5
        and numel_out >= _MIN_NUMEL_XPU
        and inp.dtype in _XPU_KERNELS
        and out_shape[-1] * inp.element_size() >= _MIN_ROW_BYTES_XPU
        and all(p >= 0 for p in pad)
    )

    with torch_device_fn.device(inp.device):
        if use_xpu:
            _launch_xpu_const(inp, out, v, in_shape, out_shape, pad_left, ndim)
        elif use_fast:
            _launch_const(inp, out, v, in_shape, out_shape, pad_left, ndim)
        else:
            _launch_general(inp, out, v, in_shape, out_shape, pad_left, ndim, mode)
    return out
