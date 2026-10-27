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

# Inferred min/max: the single-CTA direct kernel wins through 10240 elements
# (the official 10000-element shapes land inside); above that the
# minmax+reduce pair takes over.
_MINMAX_DIRECT_MAX_ELEMENTS = 10_240

# Cross-block min/max reduce stays a single 1-CTA kernel up to this many
# blocks (1-CTA wins <=8192 blocks; the extra launch of the two-level
# scheme wins from ~10000 blocks up, official shapes hit 10000 stat blocks).
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


# ── fused validate + dense fill kernels ──────────────────────────────────────
#
# One multi-CTA launch folds the whole-input error verdict into a pre-zeroed
# 1-element int32 via atomic max: the zero-init is a ~5 us aten fill
# enqueue, far cheaper than any second Triton launch.  A single-CTA
# "direct verdict store" variant was measured and dropped: S5000
# single-CTA store bandwidth is ~19 GB/s, so even a 512 KiB output takes
# ~30 us of serial GPU time once validation compute rides the loop, more
# than the 5 us zero-init it saves (mirrors the c85ebe803 finding for the
# validation-only single-CTA band).


@libentry()
@triton.jit
def one_hot_validate_fill_kernel(
    input_ptr,
    output_ptr,
    verdict_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
    CLASS_BLOCK: tl.constexpr,
):
    """Dense fill and validation fused into a single launch.

    Each program validates its row block, folds it to one error code and
    accumulates it into the pre-zeroed verdict with an atomic max, then
    writes its output tiles (the dense equality form never stores out of
    bounds: indices outside [0, num_classes) simply match no column).
    The atomic is masked to blocks that actually contain an invalid index:
    all-valid blocks (the overwhelmingly common case) issue no atomic at
    all, so same-address serialization never taxes the dense band
    (10000-block grids measured +40 us with an unmasked atomic).
    Replaces the previous codes + reduce + fill triple launch.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < num_elements
    indices = tl.load(input_ptr + rows, mask=row_mask, other=0)
    negative = indices < 0
    too_large = indices >= num_classes
    code = tl.where(negative, 2, tl.where(too_large, 1, 0))
    block_code = tl.max(tl.where(row_mask, code, 0), axis=0)
    tl.atomic_max(verdict_ptr, block_code.to(tl.int32), mask=block_code > 0)

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


# ── fast-path verdict kernels ─────────────────────────────────────────────────
#
# The fast path splits validation from generation: the verdict kernels read
# the input and reduce it all the way down to a [min, max] pair (inferred
# path) left on the device, so the host pays one tiny blocking readback
# instead of copying a full per-block status array to the CPU.  The enqueue
# order is verdict-first: the reduction kernel is launched before the fill
# plan lookup, the output allocation and the fill launch, so all of that
# host work overlaps the verdict round trip.  The blocking readback that
# finally reads the verdict is stream-ordered behind the speculative fill,
# so an invalid index raises synchronously with the exact reference message
# while the output of a poisoned call never escapes.


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


# ── launch configuration helpers ──────────────────────────────────────────────


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _dense_config(num_elements: int, num_classes: int):
    """Pick row-block, class-tile and warp sizes for the dense kernels.

    The per-program tile is capped at 4096 int64 values (32 KiB): S5000
    crossover measurements show wide row blocks (256 rows at 16 classes)
    writing larger contiguous spans beat smaller tiles by ~7% on
    bandwidth-bound outputs.
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

    The per-block stat slots (16 bytes each) are reduced on the device by
    a 1-CTA follow-up kernel, so the block count only needs to stay at or
    below ``_ONE_CTA_MAX_BLOCKS`` for the direct reduce.
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


def _fill_plan(num_elements: int, num_classes: int):
    """Launch parameters for the block-tiled dense fill kernel."""
    rows, class_block, num_warps = _dense_config(num_elements, num_classes)
    grid = triton.cdiv(num_elements, rows)
    return grid, rows, class_block, num_warps


@functools.lru_cache(maxsize=4096)
def _fill_plan_cached(num_elements: int, num_classes: int):
    # _fill_plan is a pure function of (num_elements, num_classes); the
    # hot path pays a cache hit (~0.3 us) instead of a recompute.
    return _fill_plan(num_elements, num_classes)


def _launch_fill(flat_input, out, num_elements: int, num_classes: int, plan):
    """Enqueue the dense fill kernel on the current stream."""
    grid, rows, class_block, num_warps = plan
    one_hot_dense_kernel[(grid,)](
        flat_input,
        out,
        num_elements,
        num_classes,
        BLOCK_SIZE=rows,
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


# ── fast-path fused validate+fill launch helpers ─────────────────────────────


def _launch_validate_fill(flat_input, out, verdict, num_elements, num_classes):
    """Enqueue the multi-CTA fused validate+fill on the current stream.

    ``verdict`` must be a pre-zeroed 1-element int32 tensor: every program
    folds its row block to one error code and atomic-maxes it in, so the
    blocking ``.item()`` after the launch observes the global verdict.
    The zero-init is a cheap aten fill enqueue (~5 us), far below a second
    Triton launch.
    """
    grid, rows, class_block, num_warps = _fill_plan_cached(num_elements, num_classes)
    one_hot_validate_fill_kernel[(grid,)](
        flat_input,
        out,
        verdict,
        num_elements,
        num_classes,
        BLOCK_SIZE=rows,
        CLASS_BLOCK=class_block,
        num_warps=num_warps,
    )


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
    # One device lookup shared by every allocation below (~0.3 us each).
    device = tensor.device

    if inferred_num_classes:
        # One fused reduction all the way to [min, max] + one blocking
        # readback replaces aminmax + blocking copy_ + host folds.  A single
        # tolist() readback fetches both values with one device-to-host
        # copy + one synchronization.
        verdict = torch.empty(2, device=device, dtype=torch.int64)
        _launch_minmax_kernels(flat_input, num_elements, verdict)
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
            # Fused validate+fill: one kernel launch writes the dense output
            # and folds the whole-input error verdict into a pre-zeroed
            # 1-element int32 via per-program atomic max.  The fill is
            # speculative; the blocking .item() below is stream-ordered
            # behind it, so an invalid index raises with the exact reference
            # message before the (garbage) output can escape.  The output is
            # allocated in its final shape and returned directly (no re-view
            # on the way out).
            out = torch.empty(
                (*tensor.shape, num_classes),
                device=device,
                dtype=torch.int64,
            )
            verdict = torch.zeros(1, device=device, dtype=torch.int32)
            _launch_validate_fill(flat_input, out, verdict, num_elements, num_classes)
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
            device=device,
            dtype=torch.int64,
        )
        BLOCK_SIZE = _scatter_block_size(num_elements)
        num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
        status = torch.empty(num_blocks, device=device, dtype=torch.int32)
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
            # Allocated in the final shape and returned directly (no
            # re-view on the way out); the fill kernel writes through the
            # flat contiguous layout.
            plan = _fill_plan_cached(num_elements, num_classes)
            out = torch.empty(
                (*tensor.shape, num_classes),
                device=device,
                dtype=torch.int64,
            )
            _launch_fill(flat_input, out, num_elements, num_classes, plan)
            return out
        out = torch.zeros(
            num_elements * num_classes,
            device=device,
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

    # Device fast path: fused validate+fill launch and a standard blocking
    # verdict readback.  reshape(-1) folds the contiguous() + view(-1) pair
    # into one call (identical semantics: a view when already contiguous,
    # a copy otherwise).
    flat_input = tensor.reshape(-1)
    return _one_hot_fast(tensor, flat_input, flat_input.numel(), num_classes)
