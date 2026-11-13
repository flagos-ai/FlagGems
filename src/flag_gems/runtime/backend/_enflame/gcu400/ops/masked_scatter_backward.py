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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import bracket_next_power_of_2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Two-launch stream-compaction kernels for masked_scatter_backward.
#
# Semantics: out[j] = grad[i_j] for j < count (i_j = global position of the
# j-th True in mask), out[j] = 0 for count <= j < numel.
#
# Phase 1 counts True mask elements per program.  Phase 2 rebuilds each
# element's rank with tl.cumsum and writes EVERY output position exactly
# once: selected lanes store the gradient at their compacted rank, and
# unselected lanes store zero at ``count + (#unselected before me)``, which
# covers the zero-padding tail.  Because the output size (``numel``) is known
# on host, unlike masked_select no device->host size discovery (sync), no
# output memset and no post-copy are ever needed on the ``numel <= N`` path.
#
# Grid stays at the GCU400 persistent cap (48 programs, see index_select:
# full CTA expansion measured 0.66x slower) with a contiguous chunk of
# blocks per program.
# ---------------------------------------------------------------------------

_NUM_PROGS = 48  # MAX_GRID_DIM on GCU400 (utils.config_utils)
_TARGET_PROGS = 16  # small/medium-N block-count heuristic
_MIN_BLOCK = 256
_MAX_BLOCK = 8192
_MAX_N_FAST = 2**30  # keep every offset/position safely inside int32
_BIG_WARPS = 4  # num_warps for BLOCK_SIZE >= _WARP_SPLIT
_SMALL_WARPS = 2
_WARP_SPLIT = 2048


@libentry()
@triton.jit(do_not_specialize=["N", "num_blocks", "num_blocks_per_grid"])
def _msb_count_kernel(
    mask_ptr,
    counts_ptr,
    N,
    num_blocks,
    num_blocks_per_grid,
    BLOCK_SIZE: tl.constexpr,
):
    """Phase 1: per-program count of True mask elements (no atomics)."""
    pid = tl.program_id(0)
    start_block = pid * num_blocks_per_grid
    offsets = start_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    last_block = min(num_blocks - 1, start_block + num_blocks_per_grid - 1)

    for _ in range(start_block, last_block):
        mask_vals = tl.load(mask_ptr + offsets)
        acc += mask_vals.to(tl.int32)
        offsets += BLOCK_SIZE
    # Peeled last (possibly partial) block
    mask_vals = tl.load(mask_ptr + offsets, mask=offsets < N, other=0)
    acc += mask_vals.to(tl.int32)

    tl.store(counts_ptr + pid, tl.sum(acc, axis=0))


@libentry()
@triton.jit(do_not_specialize=["N", "numel", "num_blocks", "num_blocks_per_grid"])
def _msb_write_kernel(
    grad_ptr,
    mask_ptr,
    counts_ptr,
    out_ptr,
    N,
    numel,
    num_blocks,
    num_blocks_per_grid,
    NP_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WRITE_ZEROS: tl.constexpr,
):
    """Phase 2: rank via cumsum; write each output position exactly once.

    When WRITE_ZEROS is set (output allocated with empty, numel <= N) every
    position below numel receives exactly one store, so no memset is needed.
    Otherwise the output was pre-zeroed on host and only selected lanes
    store (numel > N tail case).
    """
    pid = tl.program_id(0)

    # Global prefix of this program and the grand total, rebuilt locally
    # from the phase-1 counts (kernel boundary provides the ordering).
    rids = tl.arange(0, NP_BLOCK)
    counts = tl.load(counts_ptr + rids, mask=rids < tl.num_programs(0), other=0)
    total = tl.sum(counts, axis=0)
    advance = tl.sum(tl.where(rids < pid, counts, 0), axis=0)

    start_block = pid * num_blocks_per_grid
    offsets = start_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    last_block = min(num_blocks - 1, start_block + num_blocks_per_grid - 1)

    for _ in range(start_block, last_block):
        mask_val = tl.load(mask_ptr + offsets).to(tl.int1)
        grad_raw = tl.load(grad_ptr + offsets)
        grad_val = tl.where(mask_val, grad_raw, tl.zeros_like(grad_raw))
        mask_ints = mask_val.to(tl.int32)
        selected_before = advance + tl.cumsum(mask_ints, axis=0) - mask_ints
        pos = tl.where(mask_val, selected_before, total + offsets - selected_before)
        if WRITE_ZEROS:
            tl.store(out_ptr + pos, grad_val, mask=pos < numel)
        else:
            tl.store(out_ptr + pos, grad_val, mask=mask_val & (pos < numel))
        advance += tl.sum(mask_ints, axis=0)
        offsets += BLOCK_SIZE

    # Peeled last (possibly partial) block
    inbounds = offsets < N
    mask_val = tl.load(mask_ptr + offsets, mask=inbounds, other=0).to(tl.int1)
    grad_raw = tl.load(grad_ptr + offsets, mask=inbounds, other=0)
    grad_val = tl.where(mask_val, grad_raw, tl.zeros_like(grad_raw))
    mask_ints = mask_val.to(tl.int32)
    selected_before = advance + tl.cumsum(mask_ints, axis=0) - mask_ints
    pos = tl.where(mask_val, selected_before, total + offsets - selected_before)
    if WRITE_ZEROS:
        tl.store(out_ptr + pos, grad_val, mask=inbounds & (pos < numel))
    else:
        tl.store(out_ptr + pos, grad_val, mask=inbounds & mask_val & (pos < numel))


def _masked_scatter_backward_fast(grad_output, mask, numel):
    """Two-launch path: per-program count + full-position write."""
    N = mask.numel()

    BLOCK_SIZE = bracket_next_power_of_2(
        triton.cdiv(N, _TARGET_PROGS), _MIN_BLOCK, _MAX_BLOCK
    )
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    num_progs = min(num_blocks, _NUM_PROGS)
    blocks_per_prog = triton.cdiv(num_blocks, num_progs)
    num_progs = triton.cdiv(num_blocks, blocks_per_prog)
    NP_BLOCK = triton.next_power_of_2(num_progs)
    num_warps = _BIG_WARPS if BLOCK_SIZE >= _WARP_SPLIT else _SMALL_WARPS

    if numel > N:
        # Tail [N, numel) cannot be covered by any mask element; pre-zero.
        out = torch.zeros(numel, dtype=grad_output.dtype, device=grad_output.device)
        write_zeros = False
    else:
        out = torch.empty(numel, dtype=grad_output.dtype, device=grad_output.device)
        write_zeros = True

    counts = torch.empty(num_progs, dtype=torch.int32, device=grad_output.device)

    with torch_device_fn.device(grad_output.device):
        _msb_count_kernel[(num_progs,)](
            mask,
            counts,
            N,
            num_blocks,
            blocks_per_prog,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        _msb_write_kernel[(num_progs,)](
            grad_output,
            mask,
            counts,
            out,
            N,
            numel,
            num_blocks,
            blocks_per_prog,
            NP_BLOCK=NP_BLOCK,
            BLOCK_SIZE=BLOCK_SIZE,
            WRITE_ZEROS=write_zeros,
            num_warps=num_warps,
        )

    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def masked_scatter_backward(grad_output, mask, sizes):
    """
    Backward of masked_scatter w.r.t. ``source``.

    Matches aten::masked_scatter_backward(grad_output, mask, sizes) -> Tensor,
    which is registered CompositeExplicitAutograd in PyTorch, i.e. every
    backend (including ours) must supply its own kernel for it.  The autograd
    formula for masked_scatter is::

        self:   grad_output.masked_fill(mask, 0)       (already covered)
        source: masked_scatter_backward(grad_output, mask, source.sizes())

    Semantics::

        mask_selected = grad_output.masked_select(mask)  # stream compaction
        if mask_selected.numel() < prod(sizes):
            pad the tail with zeros
        return mask_selected.view(sizes)

    The fast path handles same-shape contiguous inputs with two kernel
    launches and no host sync.  Everything else (broadcast masks,
    non-contiguous inputs, N beyond the int32 fast-path domain) falls back
    to ``masked_select``.
    """
    logger.debug("GEMS MASKED_SCATTER_BACKWARD")

    sizes = list(sizes)
    numel = 1
    for s in sizes:
        numel *= int(s)

    N = mask.numel()

    if N == 0:
        if numel > 0:
            return torch.zeros(
                numel, dtype=grad_output.dtype, device=grad_output.device
            ).view(sizes)
        return torch.empty(0, dtype=grad_output.dtype, device=grad_output.device).view(
            sizes
        )

    if (
        tuple(grad_output.shape) == tuple(mask.shape)
        and grad_output.is_contiguous()
        and mask.is_contiguous()
        and N <= _MAX_N_FAST
        and numel <= _MAX_N_FAST
    ):
        return _masked_scatter_backward_fast(grad_output, mask, numel).view(sizes)

    # ---- fallback: broadcast/non-contiguous/huge-N via masked_select ----
    from flag_gems.ops.masked_select import masked_select  # noqa: E402

    mask_selected = masked_select(grad_output, mask)

    diff_nelem = numel - mask_selected.numel()
    if diff_nelem > 0:
        out = torch.zeros(numel, dtype=mask_selected.dtype, device=mask_selected.device)
        out[: mask_selected.numel()] = mask_selected
        mask_selected = out

    return mask_selected.view(sizes)
