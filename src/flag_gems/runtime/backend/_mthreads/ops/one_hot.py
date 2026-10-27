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

import functools
import logging
import threading

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
# Hoisted level check: the per-call debug log costs ~0.4 us even when
# disabled; the hot path pays a single boolean load instead.
_DEBUG_LOG = logger.isEnabledFor(logging.DEBUG)

# Dense generation speculatively allocates and writes the output before
# reporting an invalid index.  S5000 crossover measurements show the dense
# kernel beating zero-fill + scatter through ~64M output elements (the fill
# plus the scattered 8-byte stores amplify DRAM traffic ~16x), so keep the
# speculative allocation at or below 512 MiB.
_DENSE_MAX_OUTPUT_ELEMENTS = 67_108_864

# Beyond this class count the per-row comparison loop inside one program gets
# long enough that zero-filling the output and scattering only the "1"
# positions becomes cheaper than computing every element.
_DENSE_MAX_CLASSES = 8192


# ── fast-path fill dispatch thresholds (S5000 crossover measurements) ────────
#
# The fast path picks between three dense fill kernels:
#   * zeros-fixup  (C <= 16 power of two, numel >= 2M, numel % 128 == 0):
#     constant-zero wide stores + in-kernel L2 fixup of the "1" positions.
#     Fill-rate stores beat per-element where() ALU by 7-9% on C=16 outputs.
#   * persistent   (numel >= 1M, other C in the dense budget): each CTA owns
#     one contiguous output span (DRAM page locality) and loops over row
#     tiles; removes both store masks when the tiling is exact.  6-10% over
#     the block-per-CTA kernel on C=64/256 outputs.
#   * block-tiled  (smaller shapes): launch-latency-bound, the persistent
#     grid underutilizes the GPU below ~1M elements.
_V7_MIN_NUM_ELEMENTS = 2_000_000
_V7_TILE_ROWS = 128
_V3P_MIN_NUM_ELEMENTS = 1_048_576

# Single-CTA direct verdict kernels replace the codes+reduce pair below this
# many elements (saves one ~9-26 us launch; the whole input fits one
# 8192-wide tile loop of a single CTA).  Crossover re-measured on S5000 with
# num_warps=16 direct kernels: the explicit direct kernel wins through 16384
# elements (24576+ the codes pair wins again), the two-launch pair is also
# bimodal (~53 us GPU-bound vs ~85-95 us host-bound, phase-dependent), so the
# single-launch direct path is preferred wherever it is not slower.
_DIRECT_MAX_ELEMENTS = 16_384
# Inferred min/max: the single-CTA direct kernel wins through 10240 elements
# (the official 10000-element shapes land inside); between 10240 and 65536
# the last-block kernel below wins (single launch, full parallelism), and
# above 65536 the minmax+reduce pair wins again.
_MINMAX_DIRECT_MAX_ELEMENTS = 10_240
_MINMAX_LB_MAX_ELEMENTS = 65_536
_MINMAX_LB_BLOCK = 512

# Cross-block reduce stays a single 1-CTA kernel up to this many blocks
# (1-CTA wins <=8192 blocks; the extra launch of the two-level scheme wins
# from ~10000 blocks up, official shapes hit 10000 status blocks).
_ONE_CTA_MAX_BLOCKS = 8_192
_PARTIAL_CHUNK = 4096

# Output sizes beyond the int64 addressable range fail inside the scatter
# allocation with a bare ``TypeError``; torch native fails on the same
# absurd sizes with a ``RuntimeError`` (SymInt unpacking), so the fast path
# raises the same error class explicitly.
_INT64_MAX_ELEMENTS = 2**63 - 1


# ── dense generation kernels (single pass over rows, tiled over classes) ─────


@libentry()
@triton.jit
def one_hot_dense_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
    CLASS_BLOCK: tl.constexpr,
):
    # 32-bit ids keep the offset math in i32, which avoids keeping per-row
    # base products alive as i64 register pairs across the class loop (193
    # vs 107 registers at CLASS_BLOCK=128) — for class counts not divisible
    # by 16 that extra register pressure hits a superlinear occupancy cliff
    # (measured up to 9.4x slower).  Safe because both dense paths are
    # guarded to outputs below 2**31 elements.
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < num_elements
    indices = tl.load(input_ptr + rows, mask=row_mask, other=0)

    cols = tl.arange(0, CLASS_BLOCK)
    for col_start in range(0, num_classes, CLASS_BLOCK):
        class_offsets = col_start + cols
        col_mask = class_offsets < num_classes
        values = tl.where(indices[:, None] == class_offsets[None, :], 1, 0)
        values = values.to(tl.int64)
        out_offsets = rows[:, None] * num_classes + class_offsets[None, :]
        tl.store(
            output_ptr + out_offsets, values, mask=row_mask[:, None] & col_mask[None, :]
        )


# ── fused min/max reduction (num_classes inference) ──────────────────────────


@libentry()
@triton.jit
def one_hot_status_reduce_kernel(
    status_ptr,
    out_ptr,
    num_blocks,
    BLOCK: tl.constexpr,
):
    """Reduce per-block status slots to a single code on the device.

    Returning the verdict through a 4-byte readback keeps the host-side
    blocking copy on the same tiny-transfer driver path as the reference
    implementation, which measures stable across processes; multi-KiB
    pageable readbacks are bimodal (fast/slow driver staging modes).
    """
    code = tl.full((), 0, tl.int32)
    for start in range(0, num_blocks, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < num_blocks
        values = tl.load(status_ptr + offs, mask=mask, other=0)
        code = tl.maximum(code, tl.max(values, axis=0))
    tl.store(out_ptr, code)


@libentry()
@triton.jit
def one_hot_minmax_kernel(
    input_ptr,
    stat_ptr,
    num_elements,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute per-block min/max so the host can infer ``num_classes``.

    Block minima land in ``stat[:num_blocks]`` and block maxima in
    ``stat[num_blocks:]``; the host performs one readback and two tiny CPU
    reductions, replacing the previous aminmax + two ``.item()`` syncs with a
    single synchronization.
    """
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    block_min = tl.min(tl.where(mask, indices, 1 << 62), axis=0)
    block_max = tl.max(tl.where(mask, indices, -(1 << 62)), axis=0)
    tl.store(stat_ptr + pid, block_min)
    tl.store(stat_ptr + num_blocks + pid, block_max)


# ── scatter kernels: only write the "1" positions (output zeroed first) ──────


@libentry()
@triton.jit
def one_hot_set_one_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    out_offsets = offsets * num_classes + indices
    tl.store(output_ptr + out_offsets, 1, mask=mask)


@libentry()
@triton.jit
def one_hot_validate_set_one_kernel(
    input_ptr,
    output_ptr,
    status_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    """Validate indices while scattering into a zero-initialized output.

    Status values preserve the existing error precedence: negative indices
    take priority over indices greater than or equal to ``num_classes``.
    Every program owns its own status slot, so the buffer needs no
    initialization and the kernel needs no atomics.
    """
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    negative = indices < 0
    too_large = indices >= num_classes
    valid = mask & ~negative & ~too_large
    safe_indices = tl.minimum(tl.maximum(indices, 0), num_classes - 1)
    tl.store(
        output_ptr + offsets * num_classes + safe_indices,
        1,
        mask=valid,
    )
    code = tl.where(negative, 2, tl.where(too_large, 1, 0))
    block_code = tl.max(tl.where(mask, code, 0), axis=0)
    tl.store(status_ptr + pid, block_code.to(tl.int32))


# ── fast-path fill kernels ────────────────────────────────────────────────────


@libentry()
@triton.jit
def one_hot_persistent_dense_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_classes,
    chunk_rows,  # runtime: rows per CTA (multiple of TILE_ROWS)
    TILE_ROWS: tl.constexpr,
    CLASS_BLOCK: tl.constexpr,
    EVEN: tl.constexpr,  # tiles always full + C % CLASS_BLOCK == 0
    CMP32: tl.constexpr,  # i32 compare + direct eq->int64 cast
    NS: tl.constexpr,  # num_stages for the row loop
):
    """Persistent-contiguous dense fill for large outputs (>= 1M rows).

    Each CTA owns one contiguous span of output rows (DRAM page locality)
    and loops over row tiles inside it.  EVEN removes both store masks when
    the class count tiles exactly and every row tile is full; the masked
    branch is the general-correctness fallback for any numel/C.
    32-bit offset math is safe: dense paths are guarded to outputs below
    2**31 elements.
    """
    pid = tl.program_id(0)
    base = pid * chunk_rows
    n = tl.minimum(chunk_rows, num_elements - base)
    cols = tl.arange(0, CLASS_BLOCK)
    for r0 in tl.range(0, n, TILE_ROWS, num_stages=NS):
        rows = base + r0 + tl.arange(0, TILE_ROWS)
        if EVEN:
            indices = tl.load(input_ptr + rows)
        else:
            indices = tl.load(input_ptr + rows, mask=rows < num_elements, other=0)
        if CMP32:
            idxv = indices.to(tl.int32)
        else:
            idxv = indices
        for cs in range(0, num_classes, CLASS_BLOCK):
            class_offsets = cs + cols
            if CMP32:
                values = (idxv[:, None] == class_offsets[None, :]).to(tl.int64)
            else:
                values = tl.where(idxv[:, None] == class_offsets[None, :], 1, 0)
                values = values.to(tl.int64)
            out_offsets = rows[:, None] * num_classes + class_offsets[None, :]
            if EVEN:
                tl.store(output_ptr + out_offsets, values)
            else:
                tl.store(
                    output_ptr + out_offsets,
                    values,
                    mask=(rows < num_elements)[:, None]
                    & (class_offsets < num_classes)[None, :],
                )


@libentry()
@triton.jit
def one_hot_zeros_fixup_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_classes,
    chunk_rows,
    TILE_ROWS: tl.constexpr,
    CLASS_BLOCK: tl.constexpr,
):
    """Zeros + in-kernel L2 fixup fill for large low-class-count outputs.

    Per row tile: (1) store constant zeros for the full row span (fill-rate
    stores, no per-element ALU), (2) CTA barrier, (3) scattered 8B stores of
    the "1"s.  The fixups hit lines this CTA just wrote, so they merge in L2
    instead of costing DRAM read-modify-write.  Requires CLASS_BLOCK == C
    (single class tile) and numel % TILE_ROWS == 0 (unmasked loads/stores).
    """
    pid = tl.program_id(0)
    base = pid * chunk_rows
    n = tl.minimum(chunk_rows, num_elements - base)
    cols = tl.arange(0, CLASS_BLOCK)
    for r0 in tl.range(0, n, TILE_ROWS):
        rows = base + r0 + tl.arange(0, TILE_ROWS)
        idx = tl.load(input_ptr + rows)
        for cs in range(0, num_classes, CLASS_BLOCK):
            off = rows[:, None] * num_classes + (cs + cols)[None, :]
            tl.store(output_ptr + off, tl.zeros((TILE_ROWS, CLASS_BLOCK), tl.int64))
        tl.debug_barrier()
        # The fill is speculative (the verdict readback raises on invalid
        # indices), so clamp the fixup target into the row exactly like the
        # scatter kernels: an unclamped out-of-range index would store past
        # the output and poison the context before the verdict is read.
        safe_idx = tl.minimum(tl.maximum(idx, 0), num_classes - 1)
        tl.store(output_ptr + rows * num_classes + safe_idx, 1)


# ── fast-path verdict kernels ─────────────────────────────────────────────────
#
# The fast path splits validation from generation: the verdict kernels read
# the input and reduce it all the way down to a single error code (explicit
# path) or a [min, max] pair (inferred path) left on the device, so the host
# pays one tiny blocking readback instead of copying a full per-block status
# array to the CPU.  The enqueue order is verdict-first: the validation /
# reduction kernels are launched before the fill plan lookup, the output
# allocation and the fill launch, so all of that host work overlaps the
# verdict round trip.  The blocking ``.item()`` that finally reads the
# verdict is stream-ordered behind the speculative fill, so an invalid index
# raises synchronously with the exact reference message while the output of
# a poisoned call never escapes.


@libentry()
@triton.jit
def one_hot_validate_codes_kernel(
    input_ptr,
    status_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-block error codes (validation half of the fused baseline kernel).

    Status values preserve the error precedence: negative indices take
    priority over indices >= num_classes.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < num_elements
    indices = tl.load(input_ptr + rows, mask=row_mask, other=0)
    negative = indices < 0
    too_large = indices >= num_classes
    code = tl.where(negative, 2, tl.where(too_large, 1, 0))
    block_code = tl.max(tl.where(row_mask, code, 0), axis=0)
    tl.store(status_ptr + pid, block_code.to(tl.int32))


@libentry()
@triton.jit
def one_hot_validate_direct_kernel(
    input_ptr,
    out_ptr,
    num_elements,
    num_classes,
    BLOCK: tl.constexpr,
):
    """Single-CTA validation: whole-input error code -> out_ptr[0].

    Small inputs only: one launch replaces the codes + reduce pair.
    """
    code = tl.full((), 0, tl.int32)
    for start in range(0, num_elements, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < num_elements
        indices = tl.load(input_ptr + offs, mask=mask, other=0)
        negative = indices < 0
        too_large = indices >= num_classes
        c = tl.where(mask, tl.where(negative, 2, tl.where(too_large, 1, 0)), 0)
        code = tl.maximum(code, tl.max(c, axis=0))
    tl.store(out_ptr, code)


@libentry()
@triton.jit
def one_hot_status_partial_reduce_kernel(
    status_ptr,
    partial_ptr,
    num_blocks,
    CHUNK: tl.constexpr,
):
    """Grid-level first stage for very large block counts: each CTA reduces
    a CHUNK-wide slice of the per-block status to one int32."""
    pid = tl.program_id(0)
    offs = pid * CHUNK + tl.arange(0, CHUNK)
    mask = offs < num_blocks
    values = tl.load(status_ptr + offs, mask=mask, other=0)
    tl.store(partial_ptr + pid, tl.max(values, axis=0))


@libentry()
@triton.jit
def one_hot_minmax_direct_kernel(
    input_ptr,
    out_ptr,
    num_elements,
    BLOCK: tl.constexpr,
):
    """Single-CTA global min/max -> out_ptr ([min, max]).

    Small inputs only (one launch).
    """
    vmin = tl.full((), 1 << 62, tl.int64)
    vmax = tl.full((), -(1 << 62), tl.int64)
    for start in range(0, num_elements, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < num_elements
        indices = tl.load(input_ptr + offs, mask=mask, other=0)
        vmin = tl.minimum(vmin, tl.min(tl.where(mask, indices, 1 << 62), axis=0))
        vmax = tl.maximum(vmax, tl.max(tl.where(mask, indices, -(1 << 62)), axis=0))
    tl.store(out_ptr, vmin)
    tl.store(out_ptr + 1, vmax)


@libentry()
@triton.jit
def one_hot_minmax_reduce_kernel(
    stat_ptr,
    out_ptr,
    num_blocks,
    BLOCK: tl.constexpr,
):
    """Cross-block reduce of the per-block min/max stat to [min, max].

    The reduction stays on the device all the way down to two scalars, so
    the host readback is one tiny blocking copy regardless of the block
    count.
    """
    vmin = tl.full((), 1 << 62, tl.int64)
    vmax = tl.full((), -(1 << 62), tl.int64)
    for start in range(0, num_blocks, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < num_blocks
        lo = tl.load(stat_ptr + offs, mask=mask, other=1 << 62)
        hi = tl.load(stat_ptr + num_blocks + offs, mask=mask, other=-(1 << 62))
        vmin = tl.minimum(vmin, tl.min(lo, axis=0))
        vmax = tl.maximum(vmax, tl.max(hi, axis=0))
    tl.store(out_ptr, vmin)
    tl.store(out_ptr + 1, vmax)


@libentry()
@triton.jit
def one_hot_minmax_partial_reduce_kernel(
    stat_ptr,
    stat2_ptr,
    num_blocks,
    out_blocks,
    CHUNK: tl.constexpr,
):
    """Grid-level first stage for very large block counts: each CTA reduces
    a CHUNK-wide slice of the [mins | maxes] stat into stat2's pair layout."""
    pid = tl.program_id(0)
    offs = pid * CHUNK + tl.arange(0, CHUNK)
    mask = offs < num_blocks
    lo = tl.load(stat_ptr + offs, mask=mask, other=1 << 62)
    hi = tl.load(stat_ptr + num_blocks + offs, mask=mask, other=-(1 << 62))
    tl.store(stat2_ptr + pid, tl.min(lo, axis=0))
    tl.store(stat2_ptr + out_blocks + pid, tl.max(hi, axis=0))


@libentry()
@triton.jit
def one_hot_minmax_lastblock_kernel(
    input_ptr,
    stat_ptr,
    out_ptr,
    cnt_ptr,
    num_elements,
    num_blocks,
    BLOCK: tl.constexpr,
    RB: tl.constexpr,
):
    """Single-launch min/max with a device-side last-block reduction.

    Every CTA reduces its BLOCK-wide slice to one (min, max) pair in the
    per-call stat buffer, then bumps the arrival counter with an
    acquire-release atomic.  The LAST CTA to arrive re-reduces the per-block
    stat and publishes [min, max].  Replaces the two-launch minmax+reduce
    pair in the 10240-65536 element band (one launch saved; the pair's
    second enqueue is otherwise exposed in the inferred serial chain).

    Plain (non-atomic) int64 stat stores are deliberate: the int64
    atomic_min/max paths return wrong values on this backend, so visibility
    is provided by the acq_rel arrival counter instead (release of the stat
    stores -> RMW chain -> acquire in the last CTA), which measured clean
    across 24000 stress launches.

    Buffer contract (see _lb_buffers): the counter buffer is zero at launch
    and the last CTA resets it after publishing, so steady-state reuse needs
    no re-initialization.  The verdict readback (a blocking .item()) is
    stream-ordered behind the kernel, so the reset has retired before the
    buffers can be reused.  Buffers are tagged with the launching stream:
    same-stream reuse is ordered by the GPU stream, and a stream switch
    reallocates to rule out overlap with a still-running tail (e.g. two
    host threads dispatching on different streams concurrently).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_elements
    indices = tl.load(input_ptr + offs, mask=mask, other=0)
    bmin = tl.min(tl.where(mask, indices, 1 << 62), axis=0)
    bmax = tl.max(tl.where(mask, indices, -(1 << 62)), axis=0)
    tl.store(stat_ptr + pid, bmin)
    tl.store(stat_ptr + num_blocks + pid, bmax)
    tl.debug_barrier()
    arrived = tl.atomic_add(cnt_ptr, 1, sem="acq_rel")
    if arrived == num_blocks - 1:
        vmin = tl.full((), 1 << 62, tl.int64)
        vmax = tl.full((), -(1 << 62), tl.int64)
        for start in range(0, num_blocks, RB):
            o = start + tl.arange(0, RB)
            m = o < num_blocks
            lo = tl.load(stat_ptr + o, mask=m, other=1 << 62)
            hi = tl.load(stat_ptr + num_blocks + o, mask=m, other=-(1 << 62))
            vmin = tl.minimum(vmin, tl.min(lo, axis=0))
            vmax = tl.maximum(vmax, tl.max(hi, axis=0))
        tl.store(out_ptr, vmin)
        tl.store(out_ptr + 1, vmax)
        tl.store(cnt_ptr, 0)


# ── launch configuration helpers ──────────────────────────────────────────────


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _dense_config(num_elements: int, num_classes: int):
    """Pick row-block, class-tile and warp sizes for the dense kernels.

    The per-program tile is capped at 4096 int64 values (32 KiB): S5000
    crossover measurements show wide row blocks (256 rows at 16 classes)
    writing larger contiguous spans beat smaller tiles by ~7% on
    bandwidth-bound outputs.  Row blocks are additionally sized so the
    per-block status buffer lands in a fast pageable-DtoH band: either tiny
    (<= 128 B) for small inputs, or 2 KiB and above for larger ones — S5000
    pageable copies in the 256 B-1 KiB range run through serialized
    ~512 B staging chunks and cost ~20 us.
    """
    class_block = min(max(_next_pow2(num_classes), 16), 256)
    hi = max(4, min(256, 4096 // class_block))
    if num_elements <= 2048:
        rows = _next_pow2(max(1, num_elements // 32))
    else:
        rows = _next_pow2(max(1, num_elements // 1024))
    rows = max(4, min(rows, hi))
    return rows, class_block, 4


def _minmax_block_size(num_elements: int):
    """Pick a BLOCK_SIZE for the fused min/max kernel.

    The status buffer holds 16 bytes per block; steer its total size away
    from the slow 256 B-1 KiB pageable-DtoH band (see ``_dense_config``).
    """
    for block in (1024, 2048, 4096, 8192):
        size = 16 * triton.cdiv(num_elements, block)
        if size <= 256 or size >= 2048:
            return block
    return 1024


def _scatter_block_size(num_elements: int) -> int:
    """Pick a BLOCK_SIZE for the scatter-only kernel."""
    if num_elements <= 1024:
        return 256
    elif num_elements <= 16384:
        return 512
    else:
        return 1024


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _persistent_chunk(num_elements: int, ncta: int, tile_rows: int):
    """CTA count / rows-per-CTA for the persistent-contiguous kernels."""
    chunk = ((num_elements + ncta - 1) // ncta + tile_rows - 1) // tile_rows * tile_rows
    grid = triton.cdiv(num_elements, chunk)
    return grid, chunk


def _fill_plan(num_elements: int, num_classes: int):
    """Choose the fast-path dense fill kernel + launch parameters.

    Dispatch (S5000 crossover measurements):
      * C <= 16 power of two, numel >= 2M, numel % 128 == 0  ->  zeros-fixup
      * numel >= 1M (any other C in the dense budget)        ->  persistent
      * smaller shapes                                       ->  block-tiled
    """
    if (
        num_classes <= 16
        and _is_pow2(num_classes)
        and num_classes >= 4
        and num_elements >= _V7_MIN_NUM_ELEMENTS
        and num_elements % _V7_TILE_ROWS == 0
    ):
        grid, chunk = _persistent_chunk(num_elements, 240, _V7_TILE_ROWS)
        return ("fixup", grid, chunk, _V7_TILE_ROWS, num_classes, 4, 1, True)

    if num_elements >= _V3P_MIN_NUM_ELEMENTS:
        if num_classes <= 64:
            ncta, tile_rows, num_warps, ns = 960, 128, 4, 1
        else:
            ncta, tile_rows, num_warps, ns = 240, 256, 8, 2
        class_block = 64
        for cand in (64, 32, 16, 8):
            if num_classes % cand == 0:
                class_block = cand
                break
        else:
            class_block = min(max(_next_pow2(num_classes), 8), 64)
        even = num_classes % class_block == 0 and num_elements % tile_rows == 0
        grid, chunk = _persistent_chunk(num_elements, ncta, tile_rows)
        return ("persistent", grid, chunk, tile_rows, class_block, num_warps, ns, even)

    if num_classes >= 128 and num_elements <= 65536:
        # P12: small-shape high-class fills are write-throughput bound and
        # the wide class tile thrashes; narrow 64-wide tiles with 8-row
        # blocks measured 1990 GB/s vs 1573 GB/s at (10000, 256) --
        # matching torch native's fill bandwidth.
        return ("block", triton.cdiv(num_elements, 8), 8, 0, 64, 4, 0, False)

    rows, class_block, num_warps = _dense_config(num_elements, num_classes)
    grid = triton.cdiv(num_elements, rows)
    return ("block", grid, rows, 0, class_block, num_warps, 0, False)


@functools.lru_cache(maxsize=4096)
def _fill_plan_cached(num_elements: int, num_classes: int):
    # _fill_plan is a pure function of (num_elements, num_classes); the
    # explicit hot path pays a cache hit (~0.3 us) instead of a recompute.
    return _fill_plan(num_elements, num_classes)


def _fill_plan_candidates(num_elements):
    """Power-of-two class counts within the dense budget, ascending.

    The inferred path precomputes fill plans for these while the min/max
    reduction runs; real class counts are powers of two in the
    overwhelming majority of workloads, and any miss recomputes on the
    spot with no correctness impact.
    """
    max_c = min(_DENSE_MAX_CLASSES, _DENSE_MAX_OUTPUT_ELEMENTS // num_elements)
    c = 1
    while c <= max_c:
        yield c
        c *= 2


def _launch_fill(flat_input, out, num_elements: int, num_classes: int, plan):
    """Enqueue the planned fill kernel on the current stream."""
    kind, grid, x, tile_rows, class_block, num_warps, ns, even = plan
    if kind == "fixup":
        one_hot_zeros_fixup_kernel[(grid,)](
            flat_input,
            out,
            num_elements,
            num_classes,
            x,  # chunk_rows
            TILE_ROWS=tile_rows,
            CLASS_BLOCK=class_block,
            num_warps=num_warps,
        )
    elif kind == "persistent":
        one_hot_persistent_dense_kernel[(grid,)](
            flat_input,
            out,
            num_elements,
            num_classes,
            x,  # chunk_rows
            TILE_ROWS=tile_rows,
            CLASS_BLOCK=class_block,
            EVEN=even,
            CMP32=1,
            NS=ns,
            num_warps=num_warps,
        )
    else:
        one_hot_dense_kernel[(grid,)](
            flat_input,
            out,
            num_elements,
            num_classes,
            BLOCK_SIZE=x,  # rows per block
            CLASS_BLOCK=class_block,
            num_warps=num_warps,
        )


# Torch CPU reductions fork an OpenMP team above this grain size, which costs
# ~10 ms per call on large hosts — far more than the entire one_hot call.  The
# scatter-path status buffers can exceed it for very large inputs, so
# host-side reductions over them are chunked to stay on the serial fast path.
_HOST_REDUCTION_GRAIN = 32768


def _host_max(values: torch.Tensor) -> int:
    n = values.numel()
    if n <= _HOST_REDUCTION_GRAIN:
        return int(values.max())
    result = None
    for i in range(0, n, _HOST_REDUCTION_GRAIN):
        chunk = int(values[i : i + _HOST_REDUCTION_GRAIN].max())
        if result is None or chunk > result:
            result = chunk
    return result


# ── last-block min/max buffers ────────────────────────────────────────────────
#
# stat needs no initialization (every slot is written before the last CTA
# reads it, ordered by the arrival counter); cnt must be zero at launch and
# the kernel self-resets it.  Buffers are cached per device AND tagged with
# the current stream handle they were last used on: reuse across a stream
# switch would let a new kernel race the previous kernel's tail, so a
# switch reallocates.

_LB_LOCK = threading.Lock()
_LB_STATE = {}  # device index -> [stat, cnt, capacity, stream handle]


def _lb_buffers(device, num_blocks):
    idx = device.index if device.index is not None else 0
    stream = torch_device_fn.current_stream(idx).musa_stream
    st = _LB_STATE.get(idx)
    if st is not None and st[2] >= num_blocks and st[3] == stream:
        return st[0], st[1]
    with _LB_LOCK:
        st = _LB_STATE.get(idx)
        if st is None or st[2] < num_blocks or st[3] != stream:
            stat = torch.empty(2 * num_blocks, device=device, dtype=torch.int64)
            cnt = torch.zeros(1, device=device, dtype=torch.int32)
            _LB_STATE[idx] = [stat, cnt, num_blocks, stream]
            return stat, cnt
        return st[0], st[1]


# ── fast-path verdict launch helpers ──────────────────────────────────────────


def _launch_validate_kernels(flat_input, num_elements, num_classes, target):
    """Enqueue the explicit-path validation writing the error code to target."""
    if num_elements <= _DIRECT_MAX_ELEMENTS:
        # num_warps=16 measured 30% faster than 8 on S5000 (the 8192-wide
        # single-CTA loop is latency-bound on issue width, not occupancy).
        # BLOCK=16384 keeps 8192 < n <= 16384 in a single loop iteration
        # (P12: one fewer serialized load round trip).
        block = 16384 if num_elements > 8192 else 8192
        one_hot_validate_direct_kernel[(1,)](
            flat_input,
            target,
            num_elements,
            num_classes,
            BLOCK=block,
            num_warps=16,
        )
        return
    rows, _class_block, _num_warps = _dense_config(num_elements, num_classes)
    num_blocks = triton.cdiv(num_elements, rows)
    status = torch.empty(num_blocks, device=flat_input.device, dtype=torch.int32)
    one_hot_validate_codes_kernel[(num_blocks,)](
        flat_input,
        status,
        num_elements,
        num_classes,
        BLOCK_SIZE=rows,
    )
    if num_blocks <= _ONE_CTA_MAX_BLOCKS:
        one_hot_status_reduce_kernel[(1,)](status, target, num_blocks, BLOCK=1024)
    else:
        partials = triton.cdiv(num_blocks, _PARTIAL_CHUNK)
        partial = torch.empty(partials, device=flat_input.device, dtype=torch.int32)
        one_hot_status_partial_reduce_kernel[(partials,)](
            status, partial, num_blocks, CHUNK=_PARTIAL_CHUNK
        )
        one_hot_status_reduce_kernel[(1,)](partial, target, partials, BLOCK=1024)


def _launch_minmax_kernels(flat_input, num_elements, target):
    """Enqueue the inferred-path reduction writing [min, max] to target."""
    if num_elements <= _MINMAX_DIRECT_MAX_ELEMENTS:
        one_hot_minmax_direct_kernel[(1,)](
            flat_input,
            target,
            num_elements,
            BLOCK=8192,
            num_warps=16,
        )
        return
    if num_elements <= _MINMAX_LB_MAX_ELEMENTS:
        # Single-launch last-block reduction beats both the single-CTA
        # direct kernel (serial 8192-wide loop) and the two-launch pair in
        # this band (one fewer enqueue in the inferred serial chain).
        num_blocks = triton.cdiv(num_elements, _MINMAX_LB_BLOCK)
        stat, cnt = _lb_buffers(flat_input.device, num_blocks)
        one_hot_minmax_lastblock_kernel[(num_blocks,)](
            flat_input,
            stat,
            target,
            cnt,
            num_elements,
            num_blocks,
            BLOCK=_MINMAX_LB_BLOCK,
            RB=1024,
            num_warps=4,
        )
        return
    BLOCK_SIZE = _minmax_block_size(num_elements)
    num_warps = max(4, min(8, BLOCK_SIZE // 1024))
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
    stat = torch.empty(2 * num_blocks, device=flat_input.device, dtype=torch.int64)
    one_hot_minmax_kernel[(num_blocks,)](
        flat_input,
        stat,
        num_elements,
        num_blocks,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    if num_blocks <= _ONE_CTA_MAX_BLOCKS:
        one_hot_minmax_reduce_kernel[(1,)](stat, target, num_blocks, BLOCK=1024)
    else:
        partials = triton.cdiv(num_blocks, _PARTIAL_CHUNK)
        stat2 = torch.empty(2 * partials, device=flat_input.device, dtype=torch.int64)
        one_hot_minmax_partial_reduce_kernel[(partials,)](
            stat, stat2, num_blocks, partials, CHUNK=_PARTIAL_CHUNK
        )
        one_hot_minmax_reduce_kernel[(1,)](stat2, target, partials, BLOCK=1024)


# ── fast-path entry point (device tensors only; called after the shared
#    semantics preamble in ``one_hot``) ────────────────────────────────────────


def _one_hot_fast(tensor, flat_input, num_elements, num_classes):
    """Device one_hot with tuned fill kernels and a standard verdict sync.

    Semantics are identical to the aten reference; only the kernel
    selection and the enqueue order differ.  The device context is only
    entered when the tensor does not live on the current device: the
    context manager itself costs ~5 us per call, and every allocation below
    passes an explicit device anyway.
    """
    idx = tensor.device.index
    if idx is not None and torch_device_fn.current_device() == idx:
        return _one_hot_fast_impl(tensor, flat_input, num_elements, num_classes)
    with torch_device_fn.device(tensor.device):
        return _one_hot_fast_impl(tensor, flat_input, num_elements, num_classes)


def _one_hot_fast_impl(tensor, flat_input, num_elements, num_classes):
    inferred_num_classes = num_classes == -1
    device_explicit = (not inferred_num_classes) and num_classes > 0

    if inferred_num_classes:
        # One fused reduction all the way to [min, max] + one blocking
        # readback replaces aminmax + blocking copy_ + host folds.  Fill
        # plans for the power-of-two candidate class counts within the
        # dense budget are warmed in the shared LRU cache while the
        # reduction runs; a miss (e.g. C=17) simply recomputes on the spot.
        # A single tolist() readback fetches both values with one
        # device-to-host copy + one synchronization.
        verdict = torch.empty(2, device=tensor.device, dtype=torch.int64)
        _launch_minmax_kernels(flat_input, num_elements, verdict)
        for cand in _fill_plan_candidates(num_elements):
            _fill_plan_cached(num_elements, cand)
        min_val, max_val = verdict.tolist()
        if min_val < 0:
            raise RuntimeError("Class values must be non-negative.")
        num_classes = max_val + 1
    elif not device_explicit:
        # A num_classes that is neither -1 nor positive keeps the
        # reference host-side validation semantics on device tensors.
        invalid = (tensor < 0) | (tensor >= num_classes)
        if invalid.any():
            if (tensor < 0).any():
                raise RuntimeError("Class values must be non-negative.")
            else:
                raise RuntimeError("Class values must be smaller than num_classes.")

    if num_classes < 1:
        raise RuntimeError("num_classes should be positive")

    if num_elements * num_classes > _INT64_MAX_ELEMENTS:
        raise RuntimeError(
            "one_hot: num_elements * num_classes overflows the addressable "
            "output size"
        )

    if device_explicit:
        if (
            num_elements * num_classes <= _DENSE_MAX_OUTPUT_ELEMENTS
            and num_classes <= _DENSE_MAX_CLASSES
        ):
            # Verdict first: the validation kernels are enqueued before the
            # plan lookup, the output allocation and the fill launch, so
            # all of that host work overlaps the verdict round trip.  The
            # fill is speculative; the blocking .item() below is
            # stream-ordered behind it, so an invalid index raises with the
            # exact reference message before the (garbage) output can
            # escape.  The output is allocated in its final shape and
            # returned directly (no re-view on the way out).
            verdict = torch.empty(1, device=tensor.device, dtype=torch.int32)
            _launch_validate_kernels(flat_input, num_elements, num_classes, verdict)
            plan = _fill_plan_cached(num_elements, num_classes)
            out = torch.empty(
                (*tensor.shape, num_classes),
                device=tensor.device,
                dtype=torch.int64,
            )
            _launch_fill(flat_input, out, num_elements, num_classes, plan)
            error = int(verdict.item())
            if error == 2:
                raise RuntimeError("Class values must be non-negative.")
            if error == 1:
                raise RuntimeError("Class values must be smaller than num_classes.")
            return out
        # Over the dense budget: original scatter path (GPU-bound sizes,
        # the readback is irrelevant here).
        out = torch.zeros(
            num_elements * num_classes,
            device=tensor.device,
            dtype=torch.int64,
        )
        BLOCK_SIZE = _scatter_block_size(num_elements)
        num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
        status = torch.empty(num_blocks, device=tensor.device, dtype=torch.int32)
        one_hot_validate_set_one_kernel[(num_blocks,)](
            flat_input,
            out,
            status,
            num_elements,
            num_classes,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        status_host = torch.empty(num_blocks, dtype=torch.int32)
        status_host.copy_(status)
        error = _host_max(status_host)
        if error == 2:
            raise RuntimeError("Class values must be non-negative.")
        if error == 1:
            raise RuntimeError("Class values must be smaller than num_classes.")
    else:
        # Inferred classes are valid by construction (0 <= v <= max),
        # so generation needs no validation pass at all.
        if (
            num_classes <= _DENSE_MAX_CLASSES
            and num_elements * num_classes <= _DENSE_MAX_OUTPUT_ELEMENTS
        ):
            plan = _fill_plan_cached(num_elements, num_classes)
            out = torch.empty(
                num_elements * num_classes,
                device=tensor.device,
                dtype=torch.int64,
            )
            _launch_fill(flat_input, out, num_elements, num_classes, plan)
        else:
            out = torch.zeros(
                num_elements * num_classes,
                device=tensor.device,
                dtype=torch.int64,
            )
            BLOCK_SIZE = _scatter_block_size(num_elements)
            one_hot_set_one_kernel[(triton.cdiv(num_elements, BLOCK_SIZE),)](
                flat_input,
                out,
                num_elements,
                num_classes,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return out.view(*tensor.shape, num_classes)


# ── main entry point ──────────────────────────────────────────────────────────


def one_hot(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    if _DEBUG_LOG:
        logger.debug("GEMS_MTHREADS ONE_HOT")

    if tensor.dtype != torch.int64:
        raise RuntimeError(
            "one_hot is only applicable to index tensor of type LongTensor."
        )

    if tensor.numel() == 0:
        if num_classes <= 0:
            raise RuntimeError(
                "Can not infer total number of classes from empty tensor."
            )
        shape = (*tensor.shape, num_classes)
        return torch.empty(shape, device=tensor.device, dtype=torch.int64)

    inferred_num_classes = num_classes == -1
    on_device = tensor.device.type != "cpu"

    if not on_device:
        if inferred_num_classes:
            num_classes = int(tensor.max().item()) + 1
            if (tensor < 0).any():
                raise RuntimeError("Class values must be non-negative.")
        else:
            invalid = (tensor < 0) | (tensor >= num_classes)
            if invalid.any():
                if (tensor < 0).any():
                    raise RuntimeError("Class values must be non-negative.")
                else:
                    raise RuntimeError("Class values must be smaller than num_classes.")
        if num_classes < 1:
            raise RuntimeError("num_classes should be positive")
        out = torch.zeros((*tensor.shape, num_classes), device="cpu", dtype=torch.int64)
        out.scatter_(-1, tensor.unsqueeze(-1), 1)
        return out

    # Device fast path: verdict-first enqueue order with tuned fill kernels
    # and a standard blocking verdict readback.
    flat_input = tensor.contiguous().view(-1)
    return _one_hot_fast(tensor, flat_input, flat_input.numel(), num_classes)
