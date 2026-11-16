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

# Kunlunxin (XPU) override of `flag_gems.ops.put.put` / `put.out`.
#
# Root cause on XPU3: the generic scatter kernel enforces the ATen index-bounds
# contract with a hand-written *PTX* trap inside `tl.inline_asm_elementwise`
# (`setp.ne.b32 ... @p trap`).  The XPU backend feeds inline asm to llc for the
# XPU target, so every `put`/`put_` launch fails to compile:
#
#   llc (LLVM 19) failed:
#   <inline asm>:1:4: error: Cannot deduce expr kind.
#           { .reg .pred p; setp.ne.b32 p, r4, 0; @p trap; mov.u32 r4, 0; }
#
# `tl.device_assert` is not an alternative: the XPU backend lowers it to a
# no-op (see _kunlunxin/ops/assert_async.py), so an out-of-range index would be
# silently dropped instead of raising.
#
# Fix (same shape as _kunlunxin/ops/assert_async.py and
# _validate_sparse_csc_tensor_args.py): keep the index check on the XPU device.
# Each program reduces its own out-of-bounds lanes to one int32 verdict and
# stores it to a small device buffer; the host reads only that verdict back (a
# tiny D2H copy of already-computed booleans, never the index/source data) and
# raises, mirroring ATen's `cuda_take_put_kernel() index out of bounds`
# device-side assert.  The scatter itself is unchanged apart from skipping
# invalid lanes, so an out-of-range index can never write outside `self`.
#
# Only the kernel and its launcher differ; the generic validation / layout /
# staging logic (including the complex and narrow-int accumulate paths) is
# reused by patching the generic module's launcher in place.
import importlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

# `flag_gems.ops` re-exports `put` as a *function*, which shadows the submodule
# attribute, so fetch the module object explicitly.
_generic = importlib.import_module("flag_gems.ops.put")

from flag_gems.runtime import torch_device_fn  # noqa: E402

logger = logging.getLogger(__name__)

# Mirrors the text of the CUDA device-side assertion so callers (and the
# functional tests, which look for a device-side failure) see the native
# contract.
_OOB_MSG = "device-side assert: cuda_take_put_kernel() index out of bounds"


@libentry()
@triton.jit(do_not_specialize=["N", "out_numel"])
def put_check_kernel(index_ptr, verdict_ptr, out_numel, N, BLOCK_SIZE: tl.constexpr):
    """Device-side bounds verdict for `put`: one int32 per program.

    Kept separate from the scatter: XPU3's buffer-size tuner rejects an int64
    `tl.atomic_add` scatter combined with a `tl.max` verdict in one kernel
    ("Failed to tune buffer size"), while either half compiles on its own.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    raw_index = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
    cur_index = tl.where(raw_index < 0, raw_index + out_numel, raw_index)
    index_valid = (cur_index >= 0) & (cur_index < out_numel)
    oob = tl.max((mask & (~index_valid)).to(tl.int32)) > 0
    tl.store(verdict_ptr + pid, oob.to(tl.int32))


@libentry()
@triton.jit(do_not_specialize=["N", "out_numel"])
def put_scatter_kernel(
    out_ptr,
    index_ptr,
    source_ptr,
    layout_ptr,
    out_numel,
    N,
    RANK: tl.constexpr,
    IS_ACCUMULATE: tl.constexpr,
    IS_CONTIGUOUS: tl.constexpr,
    ELEMS_PER_SLOT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter ``source`` into ``out``.

    Same as the generic kernel minus the PTX `trap` (which the XPU backend
    cannot compile).  An out-of-range index is skipped rather than clamped, so
    a bad index can never write outside ``out``; the bounds verdict is produced
    by :func:`put_check_kernel` on the same grid.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    raw_index = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int64)
    cur_index = tl.where(raw_index < 0, raw_index + out_numel, raw_index)
    index_valid = (cur_index >= 0) & (cur_index < out_numel)
    final_mask = mask & index_valid

    if IS_CONTIGUOUS:
        out_offsets = cur_index
    else:
        cur = cur_index
        out_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        for i in tl.static_range(RANK - 1, -1, -1):
            dim_size = tl.load(layout_ptr + i).to(tl.int64)
            dim_stride = tl.load(layout_ptr + RANK + i).to(tl.int64)
            out_offsets += (cur % dim_size) * dim_stride
            cur = cur // dim_size

    if ELEMS_PER_SLOT == 1:
        value = tl.load(source_ptr + offsets, mask=final_mask, other=0)
        if IS_ACCUMULATE:
            tl.atomic_add(out_ptr + out_offsets, value, mask=final_mask, sem="relaxed")
        else:
            tl.store(out_ptr + out_offsets, value, mask=final_mask)
    else:
        # Complex: scatter the interleaved (real, imag) pairs.
        real = tl.load(source_ptr + 2 * offsets, mask=final_mask, other=0)
        imag = tl.load(source_ptr + 2 * offsets + 1, mask=final_mask, other=0)
        real_off = 2 * out_offsets
        if IS_ACCUMULATE:
            tl.atomic_add(out_ptr + real_off, real, mask=final_mask, sem="relaxed")
            tl.atomic_add(out_ptr + real_off + 1, imag, mask=final_mask, sem="relaxed")
        else:
            tl.store(out_ptr + real_off, real, mask=final_mask)
            tl.store(out_ptr + real_off + 1, imag, mask=final_mask)


def _launch_scatter(meta, data, index, source, accumulate, elems_per_slot):
    """XPU launcher: generic scatter + on-device bounds verdict read back once."""
    N = index.numel()
    BLOCK_SIZE, num_warps = _generic._put_block_config(N)
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    contiguous = meta.is_contiguous()
    if contiguous:
        layout_t = index
        rank = 0
    else:
        layout_t = torch.tensor(
            list(meta.shape) + list(meta.stride()),
            dtype=torch.int64,
            device=meta.device,
        )
        rank = meta.ndim

    # `torch.empty` only allocates the per-program verdict scratch; no numeric
    # result goes through torch.
    verdict = torch.empty(grid[0], dtype=torch.int32, device=meta.device)

    with torch_device_fn.device(meta.device):
        put_check_kernel[grid](
            index, verdict, meta.numel(), N, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
        put_scatter_kernel[grid](
            data,
            index,
            source,
            layout_t,
            meta.numel(),
            N,
            RANK=rank,
            IS_ACCUMULATE=accumulate,
            IS_CONTIGUOUS=contiguous,
            ELEMS_PER_SLOT=elems_per_slot,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    # `tolist` copies only the already-computed boolean verdicts (<= grid ints)
    # back to the host; the index/source data stay on device.
    if any(verdict.tolist()):
        raise RuntimeError(_OOB_MSG)


# Route the generic wrapper (validation, layout, staging, complex, narrow-int
# accumulate) through the XPU launcher above.
_generic._launch_scatter = _launch_scatter

put = _generic.put
put_out = _generic.put_out
