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
from triton.runtime import driver as triton_driver

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# Chunking the output side by 4096 elements with one program per SP (grid=NUM_SIPS) is measured best on this card:
# After reducing grid from 4*NUM_SIPS to NUM_SIPS, all shapes are faster for fp16/fp32 (see optimization report R1/R2).
BLOCK = 4096

# R3 resample (2026-11-19): a 1GB pure-copy roofline shows grid=48/BLOCK=4096 reaching 2330GB/s
# (grid=24 only 1479GB/s); A/B on the self_int kernel itself shows grid=48 faster than grid=24
# by 9%~39% across every benchmark shape (fp16/fp32/bf16, 2D-4D), degrading from 96 on. The
# self_int family moves to 48; NUM_SIPS=24 is kept only for repeat_interleave_tensor (tiny rows).
SELF_INT_GRID = 48


@libentry()
@triton.jit
def repeat_interleave_kernel(
    x_ptr,
    out_ptr,
    N_total,
    inner_size,
    repeat_inner: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    SLAB: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Each program claims a contiguous slab (blocks are contiguous on the output), improving DRAM page locality
    pid = tl.program_id(0)
    arange = tl.arange(0, BLOCK)
    start = pid * SLAB
    end = min(start + SLAB, NUM_BLOCKS)

    for block_id in tl.range(start, end):
        off = block_id * BLOCK + arange

        inner_idx = off % inner_size
        outer_idx = off // repeat_inner
        src_idx = outer_idx * inner_size + inner_idx

        if MASKED:
            mask = off < N_total
            vals = tl.load(x_ptr + src_idx, mask=mask)
            tl.store(out_ptr + off, vals, mask=mask)
        else:
            # When the last block is fully valid, skip the predicate; measured about 3% gain for fp16
            vals = tl.load(x_ptr + src_idx)
            tl.store(out_ptr + off, vals)


@libentry()
@triton.jit
def repeat_interleave_flat_kernel(
    x_ptr,
    out_ptr,
    N_total,
    repeats: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    SLAB: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    arange = tl.arange(0, BLOCK)
    start = pid * SLAB
    end = min(start + SLAB, NUM_BLOCKS)

    for block_id in tl.range(start, end):
        off = block_id * BLOCK + arange
        src_idx = off // repeats

        if MASKED:
            mask = off < N_total
            vals = tl.load(x_ptr + src_idx, mask=mask)
            tl.store(out_ptr + off, vals, mask=mask)
        else:
            vals = tl.load(x_ptr + src_idx)
            tl.store(out_ptr + off, vals)


def repeat_interleave_self_int(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS_ENFLAME REPEAT_INTERLEAVE_SELF_INT")
    if dim is None:
        inp = inp.contiguous().flatten()
        dim = 0
    else:
        if dim < -inp.ndim or dim >= inp.ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )
        if dim < 0:
            dim = dim + inp.ndim

    inp_shape = list(inp.shape)
    output_shape = list(inp.shape)
    output_shape[dim] *= repeats

    if output_size is not None and output_size != output_shape[dim]:
        raise RuntimeError(
            "repeat_interleave: Invalid output_size, expected {} but got {}".format(
                output_shape[dim], output_size
            )
        )

    output = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)

    if repeats == 0:
        return output

    inp = inp.contiguous()
    N_total = output.numel()

    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, SELF_INT_GRID)
    slab = triton.cdiv(NUM_BLOCKS, grid_size)
    masked = (N_total % BLOCK) != 0

    inner_size = 1
    for i in range(dim + 1, len(inp_shape)):
        inner_size *= inp_shape[i]

    with torch_device_fn.device(inp.device):
        if inner_size == 1:
            repeat_interleave_flat_kernel[(grid_size,)](
                inp,
                output,
                N_total,
                repeats=repeats,
                NUM_BLOCKS=NUM_BLOCKS,
                SLAB=slab,
                MASKED=masked,
                BLOCK=BLOCK,
            )
        else:
            repeat_inner = repeats * inner_size
            repeat_interleave_kernel[(grid_size,)](
                inp,
                output,
                N_total,
                inner_size,
                repeat_inner=repeat_inner,
                NUM_BLOCKS=NUM_BLOCKS,
                SLAB=slab,
                MASKED=masked,
                BLOCK=BLOCK,
            )

    return output


@libentry()
@triton.jit
def repeat_interleave_tensor_kernel(
    repeats_ptr, cumsum_ptr, out_ptr, size, BLOCK_SIZE: tl.constexpr
):
    # Fixed-grid friendly: stride over repeat indices so the launch never
    # exceeds the 65535 grid.x limit for large repeat vectors (grid used to
    # be (size,), which overflows at size=65536).
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    for rid in tl.range(pid, size, num_pids):
        cumsum = tl.load(cumsum_ptr + rid)
        repeats = tl.load(repeats_ptr + rid)
        # Local base (do NOT advance out_ptr in-place: the stride loop would
        # accumulate the offset across iterations).
        out_base = out_ptr + cumsum - repeats

        tl.device_assert(repeats >= 0, "repeats can not be negative")

        for start_k in range(0, repeats, BLOCK_SIZE):
            offsets_k = start_k + tl.arange(0, BLOCK_SIZE)
            mask_k = offsets_k < repeats
            tl.store(out_base + offsets_k, rid, mask=mask_k)


@libentry()
@triton.jit
def repeat_interleave_gather_kernel(
    inp_ptr,
    out_ptr,
    repeats_ptr,
    cumsum_ptr,
    N,
    inner_size,
    BLOCK: tl.constexpr,
    EVEN_INNER: tl.constexpr,
):
    # Fused dim==0 path for repeat_interleave.self_Tensor: one kernel replaces
    # the materialize-indices + index_select pair. 2D grid over (input row i,
    # inner chunk c): each CTA loads its source chunk ONCE and stores it to
    # every output row of i's run, so source read traffic is N*inner instead
    # of total_rows*inner (the run length is repeats[i], a runtime loop bound
    # that skips repeats==0 rows for free). No cross-warp reduction anywhere:
    # a per-row cumsum search measured ~2us/row on this card, which dominated
    # every small-inner shape.
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    num_x = tl.num_programs(0)
    num_y = tl.num_programs(1)
    offs = tl.arange(0, BLOCK)
    inner_blocks = tl.cdiv(inner_size, BLOCK)
    for c in tl.range(pid_y, inner_blocks, num_y):
        coff = c * BLOCK + offs
        for i in tl.range(pid_x, N, num_x):
            # int32 domain is guaranteed by the host-side numel guards
            repeats_i = tl.load(repeats_ptr + i).to(tl.int32)
            cumsum_i = tl.load(cumsum_ptr + i).to(tl.int32)
            base = cumsum_i - repeats_i
            if EVEN_INNER:
                vals = tl.load(inp_ptr + i * inner_size + coff)
                for k in tl.range(0, repeats_i):
                    tl.store(out_ptr + (base + k) * inner_size + coff, vals)
            else:
                cmask = coff < inner_size
                vals = tl.load(inp_ptr + i * inner_size + coff, mask=cmask)
                for k in tl.range(0, repeats_i):
                    tl.store(out_ptr + (base + k) * inner_size + coff, vals, mask=cmask)


# Pure-copy roofline measurement (1GB int32, r+w 2330GB/s): grid=48 and BLOCK=4096 are optimal,
# and the gather kernel's x dim reuses that point; y is bounded by GCU_MAX_GRID_Y with a stride loop.
GATHER_GRID_X = 48
GCU_MAX_GRID_Y = 63
# When the inner size is huge (>=2^17) and a row segment is long enough, BLOCK=8192 contiguous
# stores measured another ~6% faster.
GATHER_BIG_INNER = 2**17
# Very small in-row counts take the composite path (index_select handles fragmented rows
# better), avoiding tiny store granularity.
MIN_FUSED_INNER = 16

# Triton's generic launch path costs about 28us per launch on this card, i.e. more than
# half of the whole aten op, because repeat_interleave.Tensor only writes a few hundred
# elements. Cache the compiled fill kernel and drive its launcher directly (the same
# warm-launch pattern the other vendor backends use). Keys mirror the specialization
# classes Triton derives from the arguments, so a cached launcher is reused only when a
# normal launch would have reused the same binary.
TINY_FILL_BLOCK = 32
_FAST_LAUNCH_CACHE = {}
_FAST_LAUNCH_LIMIT = 64
_FAST_LAUNCH_ENABLED = True


def _fast_launch_parts(repeats, cumsum, out, size, device_index):
    """Return cached ``(launch, function, metadata)`` for the fill kernel."""
    key = (
        device_index,
        repeats.dtype,
        size == 1,
        size % 16 == 0,
        repeats.data_ptr() % 16 == 0,
        cumsum.data_ptr() % 16 == 0,
        out.data_ptr() % 16 == 0,
    )
    parts = _FAST_LAUNCH_CACHE.get(key)
    if parts is not None:
        return parts
    if len(_FAST_LAUNCH_CACHE) >= _FAST_LAUNCH_LIMIT:
        _FAST_LAUNCH_CACHE.clear()
    compiled = repeat_interleave_tensor_kernel.jit_function.warmup(
        repeats,
        cumsum,
        out,
        size,
        BLOCK_SIZE=TINY_FILL_BLOCK,
        num_warps=1,
        grid=(1,),
    )
    if hasattr(compiled, "result"):
        compiled = compiled.result()
    compiled._init_handles()
    parts = (compiled.run, compiled.function, compiled.packed_metadata)
    _FAST_LAUNCH_CACHE[key] = parts
    return parts


def repeat_interleave_tensor(repeats, *, output_size=None):
    global _FAST_LAUNCH_ENABLED

    logger.debug("GEMS_ENFLAME REPEAT_INTERLEAVE_TENSOR")
    assert repeats.ndim == 1, "repeat_interleave only accept 1D vector as repeat"
    cumsum = repeats.cumsum(axis=0)
    result_size = cumsum[-1].item()
    assert result_size >= 0, "repeats can not be negative"
    out = torch.empty((result_size,), dtype=repeats.dtype, device=repeats.device)
    size = repeats.size(0)
    grid = (min(size, NUM_SIPS * 4),)
    if _FAST_LAUNCH_ENABLED:
        try:
            device_index = repeats.device.index or 0
            launch, function, metadata = _fast_launch_parts(
                repeats, cumsum, out, size, device_index
            )
            # The launcher arg list keeps the trailing constexpr slot (its value
            # is compiled in, so the slot is never read).
            launch(
                grid[0],
                1,
                1,
                triton_driver.active.get_current_stream(device_index),
                function,
                metadata,
                None,
                None,
                None,
                repeats,
                cumsum,
                out,
                size,
                TINY_FILL_BLOCK,
            )
            return out
        except Exception:  # noqa: BLE001 - any Triton API drift falls back
            _FAST_LAUNCH_ENABLED = False
            logger.debug(
                "repeat_interleave_tensor: warm launch unavailable", exc_info=True
            )
    repeat_interleave_tensor_kernel[grid](
        repeats,
        cumsum,
        out,
        size,
        BLOCK_SIZE=TINY_FILL_BLOCK,
        num_warps=1,
    )
    return out


def repeat_interleave_self_tensor(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS_ENFLAME REPEAT_INTERLEAVE_SELF_TENSOR")
    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if dim < -inp.ndim or dim >= inp.ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )
    if repeats.ndim == 0 or (repeats.ndim == 1 and repeats.size(0) == 1):
        return repeat_interleave_self_int(
            inp, repeats.item(), dim=dim, output_size=output_size
        )
    elif repeats.ndim > 1:
        raise RuntimeError("repeats must be 0-dim or 1-dim tensor")
    if repeats.numel() == 0:
        return inp.clone()
    inp_shape = list(inp.shape)
    if dim < 0:
        dim = dim + len(inp_shape)
    if repeats.size(0) != inp_shape[dim]:
        raise RuntimeError("repeats must have the same size as input along dim")

    if dim == 0:
        inner_size = 1
        for s in inp_shape[1:]:
            inner_size *= s
        N = inp_shape[0]
        # int32 safety domain: offsets/repeats/cumsum inside this kernel are all int32;
        # going out of range falls back to the composite path (int64 index mechanism unchanged).
        if inner_size >= MIN_FUSED_INNER and inp.numel() <= 0x7FFFFFFF:
            cumsum = repeats.cumsum(axis=0)
            total_rows = int(cumsum[-1].item())
            assert total_rows >= 0, "repeats can not be negative"
            if total_rows * inner_size <= 0x7FFFFFFF:
                out_shape = [total_rows] + inp_shape[1:]
                output = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
                if total_rows == 0:
                    return output
                if not inp.is_contiguous():
                    inp = inp.contiguous()
                if not repeats.is_contiguous():
                    repeats = repeats.contiguous()
                cap = 8192 if inner_size >= GATHER_BIG_INNER else 4096
                block = min(triton.next_power_of_2(inner_size), cap)
                repeat_interleave_gather_kernel[
                    (
                        min(N, GATHER_GRID_X),
                        min(triton.cdiv(inner_size, block), GCU_MAX_GRID_Y),
                    )
                ](
                    inp,
                    output,
                    repeats,
                    cumsum,
                    N,
                    inner_size,
                    BLOCK=block,
                    EVEN_INNER=(inner_size % block) == 0,
                    num_warps=4,
                )
                return output

    indices = repeat_interleave_tensor(repeats)
    res = torch.index_select(inp, dim, indices)
    return res
