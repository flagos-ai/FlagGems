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
# Custom multi-block two-phase kernels.
#
# The standard masked_select is single-block for N ≤ 4096 (poor SM
# utilisation) and uses BLOCK_SIZE up to 4096 for larger N (deep per-CTA
# scans).  These kernels cap BLOCK_SIZE at 1024 so that:
#
#   - N ≤ 4096:    ~8  CTAs  — replaces single-block path
#   - N up to 110K: up to 108 CTAs — shallower per-CTA scan than masked_select
#   - N > 110K:     fall back to masked_select (would need row decomposition)
#
# The two-phase design mirrors masked_select's multi-CTA pipeline but is
# specialised for masked_scatter_backward's pre-zeroed output.
# ---------------------------------------------------------------------------

# Maximum element count the custom kernels can handle without row
# decomposition (108 SM × 1024 elements per block).
_MAX_N_CUSTOM = 108 * 1024  # 110592
_MAX_BLOCK_SIZE = 1024
_MIN_BLOCK_SIZE = 128


@libentry()
@triton.jit(do_not_specialize=["N"])
def _msb_count_kernel(
    mask_ptr,
    counts_ptr,
    counter_ptr,
    N,
    NP_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Phase 1: per-CTA mask count + cross-CTA exclusive scan.

    Each CTA sums the True elements in its chunk of the mask.  The last
    CTA to arrive (detected via an atomic barrier) performs an exclusive
    scan over all partial sums and writes the result back so that every
    CTA can later read its global write-offset from ``counts_ptr[pid]``.
    The total number of selected elements is stored at ``counts_ptr[np]``.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_vals = tl.load(mask_ptr + offsets, mask=offsets < N, other=0)
    count = tl.sum(mask_vals.to(tl.int32), axis=0)
    tl.store(counts_ptr + pid, count)

    # ---- cross-CTA exclusive scan (run by the last CTA) ----
    barrier = tl.atomic_add(counter_ptr, 1, sem="acq_rel")
    np = tl.num_programs(0)
    if barrier == np - 1:
        mask_np = tl.arange(0, NP_BLOCK) < np
        counts = tl.load(counts_ptr + tl.arange(0, NP_BLOCK), mask=mask_np, other=0)
        pre_sums = tl.cumsum(counts, axis=0)
        tl.store(
            counts_ptr + tl.arange(0, NP_BLOCK),
            pre_sums - counts,  # exclusive scan
            mask=mask_np,
        )
        tl.store(counts_ptr + np, tl.sum(counts, axis=0))


@libentry()
@triton.jit(do_not_specialize=["N"])
def _msb_write_kernel(
    grad_ptr,
    mask_ptr,
    counts_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Phase 2: local cumsum + scatter-write into the pre-zeroed output.

    Each CTA reads its global offset from ``counts_ptr[pid]`` (already
    exclusive-scanned by phase 1), computes an intra-block exclusive scan
    of the mask to determine local write positions, then scatter-writes
    the selected gradient elements into ``out``.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_val = tl.load(mask_ptr + offsets, mask=offsets < N, other=0).to(tl.int1)
    grad_val = tl.load(grad_ptr + offsets, mask=offsets < N, other=0)

    global_offset = tl.load(counts_ptr + pid)

    mask_ints = mask_val.to(tl.int32)
    local_pos = tl.cumsum(mask_ints, axis=0) - 1  # inclusive → exclusive

    pos = global_offset + local_pos
    tl.store(out_ptr + pos, grad_val, mask=(offsets < N) & mask_val)


# ---------------------------------------------------------------------------
# Host-side helpers
# ---------------------------------------------------------------------------


def _masked_scatter_backward_custom(grad_output, mask, sizes, numel):
    """Custom multi-block path for N ≤ 110K.

    Chooses BLOCK_SIZE and CTA count dynamically so that:
    - small N (≤4096) gets ~4–8 CTAs to replace the single-block path,
    - medium N gets up to ~108 CTAs for shallower per-CTA scans,
    while keeping n_blocks within the SM count so no row decomposition
    (and therefore no extra serialisation) is needed.
    """
    N = mask.numel()

    # Target CTA count scales with N.
    # Keep at least 8 CTAs for N≤4096 (replacing the single-block path);
    # for larger N scale up to 64 CTAs to keep per-CTA scans shallow.
    sm_count = torch_device_fn.get_device_properties(
        grad_output.device
    ).multi_processor_count
    if N <= 4096:
        target_ctas = 8
    else:
        target_ctas = min(sm_count, max(16, N // 1024))
    BLOCK_SIZE = bracket_next_power_of_2(
        max(N // target_ctas, _MIN_BLOCK_SIZE), _MIN_BLOCK_SIZE, _MAX_BLOCK_SIZE
    )
    n_blocks = triton.cdiv(N, BLOCK_SIZE)
    NP_BLOCK = triton.next_power_of_2(n_blocks)
    num_warps = min(16, BLOCK_SIZE // 32)

    out = torch.zeros(numel, dtype=grad_output.dtype, device=grad_output.device)

    with torch_device_fn.device(grad_output.device):
        counts = torch.empty(n_blocks + 1, dtype=torch.int32, device=grad_output.device)
        barrier = torch.zeros([], dtype=torch.int32, device=grad_output.device)

        _msb_count_kernel[(n_blocks,)](
            mask,
            counts,
            barrier,
            N,
            NP_BLOCK=NP_BLOCK,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        _msb_write_kernel[(n_blocks,)](
            grad_output,
            mask,
            counts,
            out,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
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

    For N ≤ ~110K we use dedicated multi-block kernels with capped block
    size to keep per-CTA scan depth shallow.  For larger tensors we
    delegate to ``masked_select`` whose multi-CTA pipeline is already
    well-optimised for GPU-scale parallelism.
    """
    logger.debug("GEMS MASKED_SCATTER_BACKWARD")

    sizes = list(sizes)
    numel = 1
    for s in sizes:
        numel *= int(s)

    N = mask.numel()

    # ---- custom multi-block path (N ≤ ~110K) ----
    if N <= _MAX_N_CUSTOM:
        out = _masked_scatter_backward_custom(grad_output, mask, sizes, numel)
        return out.view(sizes)

    # ---- large N: reuse masked_select ----
    from flag_gems.ops.masked_select import masked_select  # noqa: E402

    mask_selected = masked_select(grad_output, mask)

    diff_nelem = numel - mask_selected.numel()
    if diff_nelem > 0:
        out = torch.zeros(numel, dtype=mask_selected.dtype, device=mask_selected.device)
        out[: mask_selected.numel()] = mask_selected
        mask_selected = out

    return mask_selected.view(sizes)
