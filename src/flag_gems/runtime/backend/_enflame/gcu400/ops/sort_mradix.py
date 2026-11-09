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
"""HW MRADIX full sort for GCU400, written as TLE-Struct + TLE-Raw.

One hw thread per SIP owns a strided row set (persistent kernel, grid =
min(SIP_COUNT, M)); the TLE-Struct side converts one row to monotonic uint32
sort keys, hands the four VDMEM ping-pong windows to the TLE-Raw region
(sort_mradix_l1.tops), which runs ``ndigit`` LSD radix passes on the hardware
MRADIX unit, and then reads the sorted keys/positions back and restores the
row's dtype.

Why this replaces the Triton LSD radix in sort.py: that implementation pays
``num_bins`` full-row compare+cumsum sweeps per element (2**k_bits bins, three
launches per pass), i.e. ~16xN work per pass. The hardware unit counts a whole
vector into its 32 bin accumulators in one instruction and produces the
in-vector scatter offsets from its own running base, so a pass costs about one
count scan plus one scatter of the row, with no per-bin loop and no extra
launches. The digit count follows the key width (7 for 32-bit, 4 for 16-bit,
2 for 8-bit keys), which is why fp16/bf16/int16 rows sort with 4/7 of the
passes a fp32 row needs.

Ordering contract: the MRADIX pass emits *descending* key order, so ascending
output is obtained by bitwise-inverting the monotone key (an order-reversing
bijection, i.e. equal keys stay equal). The sort is stable either way: LSD
with a per-lane initial position keeps equal keys in row order, which is
exactly ``torch.sort(..., stable=True)``'s tie-break.

Capacity envelope (enforced by ``sort_mradix_supported``, mirrored from the
raw region): ``x.dtype`` in fp32/int32/fp16/bf16/int16/int8/uint8 and
``N <= MAX_SORT_COLS`` (32768). Four ping-pong windows of BLOCK_N uint32
elements are 16 bytes per element, i.e. 512KB of the SIP's 896KB VDMEM at the
cap; fp16/bf16 rows use the same window width because the region sorts on
32-bit keys. Everything outside the envelope (bool, int64, oversized rows)
stays on the generic path in sort.py.
"""

from pathlib import Path

import logging

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

_EXTERN_FUNC_NAME = "sort_mradix_l1_kernel"

VEC_LEN = 128
NUM_WARPS = 4

# Four uint32 ping-pong windows (key A/B, index A/B). The backend charges
# 29 bytes of DSM per element of the window (16 bytes for the four slices plus
# the compiler's own scratch for the tile ops), so BLOCK_N = 16384 is the
# largest power of two that fits the SIP's 896KB: 29 * 16384 = 475136, while
# 29 * 32768 = 950272 overflows.
MAX_SORT_COLS = 16384

# A 16-bit row's positions (32-bit) may be written by the raw region instead of
# by the launcher's own Triton loop. Both halves are measured with the core
# benchmark harness (speedup vs torch, 1024x1024 -> 4096x4096 fp16):
#   fused   1.36 -> 0.33   (two stores in one Triton iteration)
#   split   1.14 -> 1.14
# i.e. issuing a 16-bit value store and a 32-bit index store from the same
# iteration collapses once the row set is large (4.8ms against 1.4ms for
# 4096x4096), while for a small row set the split pays a subthread barrier plus
# a second pass over the VDMEM window and loses ~0.2. Hence the split only for
# row sets at or above this many elements: just above the largest measured
# "fused wins" point (1024x1024), which keeps 4096x4096 on the split path.
_IDX_IN_REGION_MIN_ELEMS = 1 << 21

# dtype -> width of the monotonic sort key the launcher builds. A 16-bit key
# is zero-extended into a uint32 lane and only needs four 5-bit digits; an
# 8-bit key would need two.
#
# Only the widths the raw writeback can restore are listed: an 8-bit row needs
# a byte-lane (VUC) store path in sort_mradix_l1.tops, so int8/uint8 stay on
# the generic radix for now.
_KEY_BITS = {
    torch.float32: 32,
    torch.int32: 32,
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.int16: 16,
}

if HAS_TLE_RAW:

    _TOPS_FILE = Path(__file__).with_name("sort_mradix_l1.tops")

    # The raw region: sort_mradix_l1.tops compiled as a standalone device
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
    def _edsl_sort_mradix_l1(*args, **kwargs): ...

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
        its bit pattern with the sign bit flipped for negatives and set for
        non-negatives (so -0.0 < +0.0 and NaN sorts above +inf, like torch's
        radix kernels); a signed integer maps to its pattern with the sign bit
        flipped; an unsigned value is already monotone.

        ``ASC`` selects ascending output: the MRADIX pass emits descending key
        order, so ascending rows are sorted as bitwise-inverted keys. Both the
        fold and the inversion are bijections on the key width, so equal keys
        stay equal and LSD stability is preserved.
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
        # hi is 0xffffffff (all ones) for a negative float / 0x80000000 (its
        # sign bit) for a non-negative one: `sign * MASK | TOP`.
        if x.dtype.is_floating():
            sign = (u >> SHIFT).to(tl.uint32)
            hi = sign * tl.full((1,), MASK, tl.uint32) | tl.full((1,), TOP, tl.uint32)
        elif x.dtype.is_int_signed():
            hi = tl.full((1,), TOP, tl.uint32)
        else:
            hi = tl.full((1,), 0, tl.uint32)
        k = u ^ hi
        if ASC:
            k = k ^ tl.full((1,), MASK, tl.uint32)
        return k

    @triton.jit
    def _from_sort_key(
        k,
        KEY_BITS: tl.constexpr,
        MASK: tl.constexpr,
        TOP: tl.constexpr,
        SHIFT: tl.constexpr,
        ASC: tl.constexpr,
        dtype: tl.constexpr,
    ):
        """Inverse of ``_to_sort_key``: back to the row's own dtype."""
        if ASC:
            k = k ^ tl.full((1,), MASK, tl.uint32)
        # hi is the sign bit for a key that came from a non-negative value and
        # all ones for one that came from a negative value: `TOP + (1 - sign) *
        # (MASK - TOP)`, the same fold as UT2VT written without a 0-1 wrap.
        if dtype.is_floating():
            sign = (k >> SHIFT).to(tl.uint32)
            hi = tl.full((1,), TOP, tl.uint32) + (
                tl.full((1,), 1, tl.uint32) - sign
            ) * tl.full((1,), MASK - TOP, tl.uint32)
        elif dtype.is_int_signed():
            hi = tl.full((1,), TOP, tl.uint32)
        else:
            hi = tl.full((1,), 0, tl.uint32)
        u = k ^ hi
        if KEY_BITS == 32:
            r = u.to(dtype, bitcast=True)
        elif KEY_BITS == 16:
            r = u.to(tl.uint16).to(dtype, bitcast=True)
        else:
            r = u.to(tl.uint8).to(dtype, bitcast=True)
        return r

    # The Struct side of the pipeline: one launch per shape, one raw call per
    # row. The loop lives here (not in the region) because this side owns the
    # VDMEM windows and the row mapping is then plain Triton arithmetic.
    @triton.jit(do_not_specialize=["M", "N"])
    def sort_mradix_l1_launch(
        input_ptr,
        output_vals_ptr,
        output_idx_ptr,
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
        FULL_TILE: tl.constexpr,
        IDX_IN_REGION: tl.constexpr,
    ):
        # Buffer ownership: all four VDMEM windows are allocated HERE, on the
        # TLE-Struct side, and handed to the raw region as plain pointers (see
        # the OWNERSHIP note at its entry). The raw region must not declare
        # them __local__ as well: both allocators start at VDMEM offset 0 and
        # neither side knows about the other, so raw static buffers would
        # silently overlap these windows.
        row_base = tl.program_id(0)
        row_stride = tl.num_programs(0)
        # Exact trip count for this hw thread: no tail clamp, so every
        # iteration owns a real row.
        nrows = tl.cdiv(M - row_base, row_stride)
        # One allocation, four slices: tle.gpu.alloc rounds every buffer up
        # (four separate [BLOCK_N] windows ask for 928KB at BLOCK_N = 32768,
        # more than the 896KB VDMEM), while a single 4*BLOCK_N window pays that
        # rounding once. The slice offsets are multiples of BLOCK_N*4 bytes,
        # i.e. 512B-aligned, which is what the raw region's vector scatter
        # needs.
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
            # The region runs NDIGIT passes and, by SRC_IS_B, always leaves
            # the sorted lanes in v_a/i_a. All four windows are declared as
            # outputs, which keeps its effects on them visible to the compiler.
            tle_raw.call(
                _edsl_sort_mradix_l1,
                [
                    tle.gpu.local_ptr(buf, (0,)),
                    tle.gpu.local_ptr(buf, (BLOCK_N,)),
                    tle.gpu.local_ptr(buf, (2 * BLOCK_N,)),
                    tle.gpu.local_ptr(buf, (3 * BLOCK_N,)),
                    output_idx_ptr + row * N,
                    NVEC,
                    NDIGIT,
                    # The region emits the positions itself only for a 16-bit
                    # row; a 32-bit row asks for none (length 0) and stores them
                    # below, where the extra pair of stores is free.
                    N if IDX_IN_REGION else 0,
                ],
                output_indices=[0, 1, 2, 3],
            )
            # Writeback (Struct): restore the row's dtype from the sorted keys, and
            # emit the positions unless the region already did. Notes that drive
            # the shape of this block (measured with the core benchmark harness):
            #   - a *masked* 16-bit store lowers to tops_masked_store_v1_s16,
            #     ~60x slower than the unmasked vector store (3.8ms vs 0.1ms),
            #     so a row that exactly fills the tile skips the predicate;
            #   - a 16-bit value store and a 32-bit index store issued from the
            #     same iteration collapse to 4.8ms against 1.4ms for a
            #     4096x4096 fp16 row set, which is why a large 16-bit row set
            #     hands the positions to the raw region (see
            #     _IDX_IN_REGION_MIN_ELEMS); a 32-bit row keeps them here, where
            #     the pair of stores is free.
            kout = tl.load(tle.gpu.local_ptr(buf, (offs,)))
            vals = _from_sort_key(kout, KEY_BITS, MASK, TOP, SHIFT, ASC, x.dtype)
            if FULL_TILE:
                tl.store(output_vals_ptr + row * N + offs, vals)
            else:
                tl.store(output_vals_ptr + row * N + offs, vals, mask=mask)
            if not IDX_IN_REGION:
                # torch.sort returns int64 positions, but a 64-bit store with a
                # per-lane mask loses half of its lanes on GCU and
                # tensor.view(int32) is not a reliable alias of int64 storage on
                # this backend, so they leave as int32 and are widened by torch.
                iout = tl.load(
                    tle.gpu.local_ptr(buf, (2 * BLOCK_N + offs,)), mask=mask
                )
                if FULL_TILE:
                    tl.store(output_idx_ptr + row * N + offs, iout.to(tl.int32))
                else:
                    tl.store(
                        output_idx_ptr + row * N + offs,
                        iout.to(tl.int32),
                        mask=mask,
                    )
            # This barrier is load bearing, not just defensive: the raw region
            # deliberately leaves its last subthread barrier to this one, so
            # that the positions writeback's read of i_a is ordered against the
            # next iteration's staging stores without paying a second
            # per-row barrier inside the region.
            tl.debug_barrier()

_MAX_GRID = None


def _max_grid(device_index=None):
    """Rows (hw threads) that can be in flight at once: one per SIP."""
    global _MAX_GRID
    if _MAX_GRID is None:
        props = torch.gcu.get_device_properties(device_index)
        _MAX_GRID = int(props.sip_count)
    return _MAX_GRID


def sort_mradix_supported(x):
    """Is ``x`` (last-dim, contiguous, 2-D) inside the level-1 envelope?"""
    if not HAS_TLE_RAW:
        return False
    if x.dtype not in _KEY_BITS:
        return False
    if x.ndim != 2 or x.numel() == 0:
        return False
    n = x.shape[-1]
    if n <= 0 or n > MAX_SORT_COLS:
        return False
    if not x.is_contiguous():
        return False
    return True


def sort_mradix(x, out_shape, descending, out_vals=None, out_idx32=None):
    """Sort the last dim of a contiguous 2-D view of ``x`` (stable, LSD radix).

    Returns the sorted values and the int32 row positions; the caller widens
    the positions to int64 (torch.sort's index dtype).
    """
    logger.debug("GEMS_ENFLAME SORT_MRADIX")
    
    n_rows, n_cols = x.shape
    if out_vals is None:
        out_vals = torch.empty_like(x)
    if out_idx32 is None:
        out_idx32 = torch.empty(x.shape, device=x.device, dtype=torch.int32)
    block_n = max(VEC_LEN, triton.next_power_of_2(n_cols))
    key_bits = _KEY_BITS[x.dtype]
    ndigit = (key_bits + 4) // 5
    grid = min(_max_grid(x.device.index), n_rows)
    sort_mradix_l1_launch[(grid,)](
        x,
        out_vals,
        out_idx32,
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
        # A power-of-two row fills the tile exactly, so the value writeback can
        # skip the per-lane predicate (see the note in the kernel).
        FULL_TILE=block_n == n_cols,
        # 16-bit rows can hand the positions to the raw region, but only when
        # the row set is big enough to pay for the split (see the note in
        # _IDX_IN_REGION_MIN_ELEMS); 32-bit rows keep both stores on this side.
        IDX_IN_REGION=key_bits == 16 and x.numel() >= _IDX_IN_REGION_MIN_ELEMS,
        num_warps=NUM_WARPS,
        num_stages=1,
    )
    return out_vals.reshape(out_shape), out_idx32
