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
"""HW MRADIX arg-sort for GCU400, written as TLE-Struct + TLE-Raw.

``torch.argsort`` is ``torch.sort(...).indices``, so this is the same LSD radix
order as sort_mradix.py, but it is a separate Struct/Raw pair on purpose: with
no values to restore the pipeline loses its whole second output stream, and the
division of labour between the two sides changes with it.

One hw thread per SIP owns a strided row set (persistent kernel, grid =
min(SIP_COUNT, M)); the TLE-Struct side converts one row to monotonic uint32
sort keys, stages keys + row positions into the four VDMEM ping-pong windows,
hands them to the TLE-Raw region (argsort_mradix_l1.tops) which runs ``ndigit``
LSD radix passes on the hardware MRADIX unit, and then reads the sorted
positions back and stores them in the dtype torch asks for.

What this buys over "sort(...)[1]", i.e. over running the sort kernel and
throwing its values away:

  - no value output tensor and no value staging/writeback: the key readback,
    the dtype restore and the value store are all gone (that is roughly a
    third of the sort kernel's HBM traffic at 16 bits and half of it at 8);
  - the positions are stored straight from VDMEM, widened to int64 by this side
    when the row fills the tile (no per-lane predicate, so the backend's masked
    64-bit store, which drops lanes, is never used), left as int32 for a
    partial tile, and emitted by the region itself for a narrow-key row set,
    where a store issued from this loop is the slower path (see
    _IDX_IN_REGION_MIN_ELEMS);
  - 8-bit rows join the envelope. What kept int8/uint8 off the mradix path in
    sort is the byte-lane value writeback, which does not exist here, so an
    8-bit row is just a two-digit sort.

Ordering contract: the MRADIX pass emits *descending* key order, so ascending
output sorts bitwise-inverted keys (``ASC``); the fold and the inversion are
bijections on the key width, so equal keys stay equal. LSD plus a per-lane
initial position keeps equal keys in row order, which is exactly
``torch.argsort(..., stable=True)``'s tie-break -- and a stable answer is a
valid answer for ``stable=False`` too.

Capacity envelope (enforced by ``argsort_mradix_supported``, mirrored from the
raw region): ``N <= MAX_SORT_COLS`` (16384). Four ping-pong windows of BLOCK_N
uint32 elements are 16 bytes per element (the backend charges 29 bytes per
element, i.e. 29 * 16384 = 475136 of the SIP's 896KB VDMEM at the cap);
fp16/bf16 rows use the same window width because the region sorts on 32-bit
keys. Everything outside the envelope (int64, bool, rows longer than the cap,
non-last ``dim``) stays on the generic path in argsort.py.
"""

import logging
from pathlib import Path

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

try:
    import triton.experimental.tle.language as tle  # noqa: F401
    from triton.experimental.tle.language import raw as tle_raw
    from triton.experimental.tle.raw import dialect

    HAS_TLE_RAW = True
except ImportError:  # pragma: no cover - TLE is part of the gcu400 toolchain
    HAS_TLE_RAW = False

_EXTERN_FUNC_NAME = "argsort_mradix_l1_kernel"

VEC_LEN = 128
NUM_WARPS = 4

# Same VDMEM budget as sort_mradix.py: four uint32 ping-pong windows, charged
# at 29 bytes of DSM per element, so BLOCK_N = 16384 is the largest power of
# two that fits the SIP's 896KB (29 * 16384 = 475136, 29 * 32768 overflows).
MAX_SORT_COLS = 16384

# Whether the positions are emitted by the raw region instead of by the Struct
# loop, and whether the Struct side may widen them to int64 itself.
#
# Widening in the Struct loop is the cheaper of the two on its own (one 8-byte
# store from VDMEM instead of a 4-byte store here plus a separate read-4/write-8
# pass), and it is what all the 32-bit rows use. But a Triton-issued store
# inside this row loop collapses once the row set is large *and* the input is
# narrower than 32 bits: measured on the core harness, a 4096x4096 row set goes
# from ~1.3ms to ~4.7ms for fp16/int16 and to ~3.6ms for int8, while the same
# shape with an int32 input is unaffected. That is the same cliff sort hits
# (see _IDX_IN_REGION_MIN_ELEMS in sort_mradix.py), so at or above this many
# elements a narrow-key row set hands the positions to the raw region -- which
# writes them with 4-subthread vector stores -- and pays a separate cast.
#
# The threshold sits just above the largest measured "Struct store wins" point
# (1024x1024 = 1M elements), which keeps 4096x4096 (16M) on the region path.
_IDX_IN_REGION_MIN_ELEMS = 1 << 21

# dtype -> width of the monotone sort key the launcher builds. A 16-bit key is
# zero-extended into a uint32 lane and needs four 5-bit digits; an 8-bit key
# needs two. Unlike the sort envelope this includes int8/uint8: no value is
# restored here, so the byte-lane store path that keeps them out of sort does
# not exist.
_KEY_BITS = {
    torch.float32: 32,
    torch.int32: 32,
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.int16: 16,
    torch.int8: 8,
    torch.uint8: 8,
}

if HAS_TLE_RAW:

    _TOPS_FILE = Path(__file__).with_name("argsort_mradix_l1.tops")

    # The raw region: argsort_mradix_l1.tops compiled as a standalone device
    # function. deferred=True makes the call site a dsl_region that is
    # materialized at the end of the pipeline instead of being inlined at the
    # call site, so the region can sit inside the row loop and is traced once,
    # not once per iteration.
    @dialect(
        name="tops",
        file=_TOPS_FILE,
        extern_func_name=_EXTERN_FUNC_NAME,
        deferred=True,
    )
    def _edsl_argsort_mradix_l1(*args, **kwargs): ...

    @triton.jit
    def _to_sort_key(
        x,
        KEY_BITS: tl.constexpr,
        MASK: tl.constexpr,
        TOP: tl.constexpr,
        SHIFT: tl.constexpr,
        ASC: tl.constexpr,
    ):
        """Monotone uint32 sort key, in the digit domain the raw region walks.

        The transform is the classic sign fold: a floating-point value maps to
        its own pattern with the sign bit flipped for negatives; a signed
        integer maps to its pattern with the sign bit flipped; an unsigned
        value is already monotone.

        Floating-point patterns are canonicalised first, because a rank order
        has to agree with the reference the accuracy tests compare against
        (CPU ``torch.argsort(stable=True)``), which is a *comparison* order:
          - every NaN maps to one quiet-NaN pattern. A plain sign fold is a
            bijection on the bit pattern, so it would order -NaN and +NaN
            separately, and -NaN would land at the *opposite* end of the row
            (its pattern folds to the smallest key). Since all NaNs fold to the
            quiet positive pattern they become equal, which puts them last
            ascending / first descending, and leaves their relative order to
            LSD stability -- exactly the reference behaviour.
          - -0.0 maps to +0.0. They compare equal, so they must carry the same
            key and let stability order them by position; folding the raw
            patterns instead would order them by sign bit.

        ``ASC`` selects ascending output: the MRADIX pass emits descending key
        order, so ascending rows are sorted as bitwise-inverted keys. The fold
        and the inversion are bijections on the key width, so equal keys stay
        equal and LSD stability is preserved.

        argsort needs no inverse: the key never leaves the VDMEM windows.
        """
        # Widen to a 32-bit lane: the region counts 32-bit lanes, and a 16/8-bit
        # key only needs 4/2 of its digits (the zero-extended upper bits are
        # all zero and never contribute to a digit that matters).
        if KEY_BITS == 32:
            u = x.to(tl.uint32, bitcast=True)
        elif KEY_BITS == 16:
            u = x.to(tl.uint16, bitcast=True).to(tl.uint32)
        else:
            u = x.to(tl.uint8, bitcast=True).to(tl.uint32)
        # Canonicalise the floating-point patterns before the fold (see above).
        # The magnitudes are per-dtype: bfloat16 keeps the float32 exponent
        # width, which is why it cannot share the float16 constants.
        if x.dtype.is_floating():
            if KEY_BITS == 32:
                MAG_MASK, INF_MAG, QUIET_NAN = 0x7FFFFFFF, 0x7F800000, 0x7FC00000
            elif x.dtype.is_bf16():
                MAG_MASK, INF_MAG, QUIET_NAN = 0x7FFF, 0x7F80, 0x7FC0
            else:
                MAG_MASK, INF_MAG, QUIET_NAN = 0x7FFF, 0x7C00, 0x7E00
            mag = u & tl.full((1,), MAG_MASK, tl.uint32)
            u = tl.where(mag > INF_MAG, tl.full((1,), QUIET_NAN, tl.uint32), u)
            u = tl.where(mag == 0, tl.zeros((1,), tl.uint32), u)
        # hi is 0xffffffff (all ones) for a negative float / 0x80000000 (its
        # sign bit) for a non-negative one: `sign * MASK | TOP`.
        if x.dtype.is_floating():
            sign = (u >> SHIFT).to(tl.uint32)
            hi = sign * tl.full((1,), MASK, tl.uint32) | tl.full(
                (1,), TOP, tl.uint32
            )
        elif x.dtype.is_int_signed():
            hi = tl.full((1,), TOP, tl.uint32)
        else:
            hi = tl.full((1,), 0, tl.uint32)
        k = u ^ hi
        if ASC:
            k = k ^ tl.full((1,), MASK, tl.uint32)
        return k

    # The Struct side of the pipeline: one launch per shape, one raw call per
    # row. The loop lives here (not in the region) because this side owns the
    # VDMEM windows and the row mapping is then plain Triton arithmetic.
    @triton.jit(do_not_specialize=["M", "N"])
    def argsort_mradix_l1_launch(
        input_ptr,
        output_idx_ptr,
        output_idx64_ptr,
        M,
        N,
        BLOCK_N: tl.constexpr,
        NVEC: tl.constexpr,
        KEY_BITS: tl.constexpr,
        MASK: tl.constexpr,
        TOP: tl.constexpr,
        SHIFT: tl.constexpr,
        ASC: tl.constexpr,
        NDIGIT: tl.constexpr,
        SRC_IS_B: tl.constexpr,
        IDX_IN_REGION: tl.constexpr,
        OUT_I64: tl.constexpr,
    ):
        row_base = tl.program_id(0)
        row_stride = tl.num_programs(0)
        # Exact trip count for this hw thread: no tail clamp, so every
        # iteration owns a real row.
        nrows = tl.cdiv(M - row_base, row_stride)
        # One allocation, four slices: tle.gpu.alloc rounds every buffer up,
        # while a single 4*BLOCK_N window pays that rounding once. The slice
        # offsets are multiples of BLOCK_N*4 bytes, i.e. 512B-aligned, which is
        # what the raw region's vector scatter needs.
        #
        # The windows are allocated by tl and handed to the region as pointers.
        # The region must not declare them __local__ as well: both allocators
        # start at VDMEM offset 0 and neither side knows about the other, so
        # raw static buffers would silently overlap these windows.
        buf = tle.gpu.alloc(
            [4 * BLOCK_N],
            dtype=tl.uint32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N
        k_pad = tl.zeros((1,), dtype=tl.uint32)
        i_pad = tl.full((1,), 0xFFFFFFFF, dtype=tl.uint32)
        offs_u = offs.to(tl.uint32)
        for r in range(0, nrows):
            row = row_base + r * row_stride
            # Key conversion (Struct): the row's own dtype -> monotone uint32
            # keys. Pad lanes take key 0, the last position of the descending
            # digit walk, and index 0xffffffff so a real key 0 outranks them.
            x = tl.load(input_ptr + row * N + offs, mask=mask, other=0)
            k = _to_sort_key(x, KEY_BITS, MASK, TOP, SHIFT, ASC)
            k = tl.where(mask, k, k_pad)
            iv = tl.where(mask, offs_u, i_pad)
            if SRC_IS_B:
                tl.store(tle.gpu.local_ptr(buf, (BLOCK_N + offs,)), k)
                tl.store(tle.gpu.local_ptr(buf, (3 * BLOCK_N + offs,)), iv)
            else:
                tl.store(tle.gpu.local_ptr(buf, (offs,)), k)
                tl.store(tle.gpu.local_ptr(buf, (2 * BLOCK_N + offs,)), iv)
            # The region runs NDIGIT passes and, by SRC_IS_B, always leaves the
            # sorted lanes in v_a/i_a. All four windows are declared as outputs,
            # which keeps its effects on them visible to the compiler.
            tle_raw.call(
                _edsl_argsort_mradix_l1,
                [
                    tle.gpu.local_ptr(buf, (0,)),
                    tle.gpu.local_ptr(buf, (BLOCK_N,)),
                    tle.gpu.local_ptr(buf, (2 * BLOCK_N,)),
                    tle.gpu.local_ptr(buf, (3 * BLOCK_N,)),
                    output_idx_ptr + row * N,
                    NVEC,
                    NDIGIT,
                    # The region emits the positions only for a narrow-key row
                    # set that is large enough to pay for the split; otherwise
                    # it is handed 0 and the Struct side stores them below.
                    N if IDX_IN_REGION else 0,
                ],
                output_indices=[0, 1, 2, 3],
            )
            # Writeback (Struct): the sorted positions are the only output.
            #   - the region already emitted them for a narrow-key row set
            #     (_IDX_IN_REGION_MIN_ELEMS), so nothing happens here;
            #   - a row that fills the tile is widened to int64 here; with no
            #     per-lane predicate the 8-byte store is a plain vector store,
            #     and this beats the region storing 32-bit positions plus a
            #     separate elementwise pass that reads them back to write 8
            #     bytes (an extra launch and an extra full read);
            #   - anything else leaves int32: a *masked* 64-bit store loses half
            #     of its lanes on GCU, so the widening is left to torch.
            if not IDX_IN_REGION:
                iout = tl.load(
                    tle.gpu.local_ptr(buf, (2 * BLOCK_N + offs,)), mask=mask
                )
                if OUT_I64:
                    tl.store(
                        output_idx64_ptr + row * N + offs, iout.to(tl.int64)
                    )
                else:
                    tl.store(
                        output_idx_ptr + row * N + offs,
                        iout.to(tl.int32),
                        mask=mask,
                    )
            # This barrier is load bearing, not just defensive: the raw region
            # deliberately leaves its last subthread barrier to this one, so
            # that the positions readback is ordered against the next row's
            # staging stores without paying a second per-row barrier inside the
            # region.
            tl.debug_barrier()


_MAX_GRID = None


def _max_grid(device_index=None):
    """Rows (hw threads) that can be in flight at once: one per SIP."""
    global _MAX_GRID
    if _MAX_GRID is None:
        props = torch.gcu.get_device_properties(device_index)
        _MAX_GRID = int(props.sip_count)
    return _MAX_GRID


def argsort_mradix_dtype_ok(dtype):
    """Would the mradix kernel accept a row of this dtype?"""
    return HAS_TLE_RAW and dtype in _KEY_BITS


def argsort_mradix_supported(x):
    """Is ``x`` (last-dim, contiguous, 2-D) inside the argsort envelope?"""
    if not argsort_mradix_dtype_ok(x.dtype):
        return False
    if x.ndim != 2 or x.numel() == 0:
        return False
    n = x.shape[-1]
    if n <= 0 or n > MAX_SORT_COLS:
        return False
    if not x.is_contiguous():
        return False
    return True


def argsort_mradix(x, descending, out_i64=True):
    """Rank the last dim of a contiguous 2-D view of ``x`` (stable, LSD radix).

    Returns the positions as int64 when the row fills the tile (every power-of-
    two row, i.e. the common case) and as int32 otherwise; the caller widens
    the int32 case.
    """
    logger.debug("GEMS_ENFLAME ARGSORT_MRADIX")

    n_rows, n_cols = x.shape
    block_n = max(VEC_LEN, triton.next_power_of_2(n_cols))
    key_bits = _KEY_BITS[x.dtype]
    ndigit = (key_bits + 4) // 5
    grid = min(_max_grid(x.device.index), n_rows)
    # Narrow keys at a large row set take the same split sort uses: the region
    # emits 32-bit positions and torch widens them afterwards. Everything else
    # stores the positions from the Struct loop, widened when the row fills the
    # tile.
    idx_in_region = key_bits < 32 and x.numel() >= _IDX_IN_REGION_MIN_ELEMS
    out_i64 = out_i64 and block_n == n_cols and not idx_in_region
    out = torch.empty(
        x.shape,
        device=x.device,
        dtype=torch.int64 if out_i64 else torch.int32,
    )
    argsort_mradix_l1_launch[(grid,)](
        x,
        # OUT_I64 is a constexpr, so the dead half of the position writeback is
        # never traced and both pointers may carry the same buffer.
        out,
        out,
        n_rows,
        n_cols,
        BLOCK_N=block_n,
        NVEC=block_n // VEC_LEN,
        KEY_BITS=key_bits,
        MASK=(1 << key_bits) - 1,
        TOP=1 << (key_bits - 1),
        SHIFT=key_bits - 1,
        ASC=int(not descending),
        NDIGIT=ndigit,
        # An odd digit count starts in window B and an even one in window A,
        # so the last pass always lands in window A (see the raw region).
        SRC_IS_B=ndigit % 2,
        IDX_IN_REGION=idx_in_region,
        # OUT_I64 is the whole widening contract: it is only set when the row
        # fills the tile (no per-lane predicate, so the 8-byte store is a plain
        # vector store) and the region is not already emitting the positions.
        OUT_I64=out_i64,
        num_warps=NUM_WARPS,
        num_stages=1,
    )
    return out
