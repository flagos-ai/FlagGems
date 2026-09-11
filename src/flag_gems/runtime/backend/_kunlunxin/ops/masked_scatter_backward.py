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
"""Kunlunxin (TritonXPU) implementation of ``aten::masked_scatter_backward``.

Why a vendor file exists at all
-------------------------------
The generic implementation (``src/flag_gems/ops/masked_scatter_backward.py``)
synchronises its two phases with a cross-CTA ticket barrier::

    barrier = tl.atomic_add(counter_ptr, 1, sem="acq_rel")   # line 76
    if barrier == tl.num_programs(0) - 1:
        ...

**Consuming the result of ``tl.atomic_add`` does not compile on TritonXPU.**
``tritonxpu-create-gm2lm{atomic-sim=true}`` rewrites (and destroys) the
``tt.atomic_rmw`` op without rewiring its result, so the pass pipeline aborts
with ``'tt.atomic_rmw' op operation destroyed but still has uses``, surfaced to
Python as ``OutOfResources: uni_sram PassManager::run failed``.  Isolated
single-variable probe (grid=4, one atomic on a scalar):

    discard the result          -> OK
    use it in arithmetic        -> FAIL   (acq_rel and default sem alike)
    use it as an ``if`` predicate -> FAIL

Consequence on HEAD: every single case of ``masked_scatter_backward`` fails.
``N <= 110592`` dies in ``_msb_count_kernel``; larger ``N`` delegates to
``flag_gems.ops.masked_select`` (module-level import, so the live vendor
``masked_select`` override is bypassed) and dies on the very same construct at
``ops/masked_select.py:92``.

Design of this implementation
-----------------------------
No atomics anywhere.  The cross-block exclusive scan is done by a dedicated
single-program kernel between the counting and the writing launch, so the
inter-block ordering comes from launch boundaries instead of a device barrier.

Everything else follows the backend's measured constraints:

* **No masked stores and no ``other=``-derived store masks.**  A masked
  discrete store is not honoured on this backend (the whole tile is written)
  and a *compound* mask on a discrete store additionally writes the wrong
  payload.  Instead every store is fully unmasked and inactive lanes are
  redirected, by address, into a scratch tail that is appended to the output
  allocation (``tl.where(active, pos, SCRATCH + lane)``).  Addresses are unique
  inside a program, so no address is written twice.
* **The compaction store is the only discrete access.**  It is irreducible for
  a stream compaction; reads stay stride-1 block DMA and the zero padding of
  the ``[count, numel)`` tail is a contiguous unmasked store.
* **Shape scalars are ``tl.constexpr``.**  ``N``/``NUMEL``/``SCRATCH`` are
  compile-time constants so no runtime i32 scalar enters an address expression
  (measured 15-30x cliff elsewhere) and the tail mask folds away completely
  whenever ``N % BLOCK == 0``.
* **Tile widths.** ``_BLOCK = 8192`` for the pipeline and ``BLOCK <= 8192`` for
  the single-launch path.  BLOCK=16384 is ~11% faster end-to-end on the huge
  cells but is NOT used: a single program issuing two 16384-lane unmasked
  discrete stores wedged the card (``wait for noc idle timeout`` /
  ``kl3ChannelCheckErrors`` / ``KL_XID_KERNEL_EXCEPTION`` / ``status=299``;
  the driver recovered it with an automatic ``m3 mode1 reset``), while the same
  kernel at BLOCK <= 8192 has been exercised hundreds of times without a fault.
* Tile widths are always powers of two and never below 64.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Element budget for the single-launch path.  Above it the three-kernel
# (count -> scan -> write) pipeline is used.  MUST stay <= 8192: the tile width
# of _msb_single_kernel is next_pow2(max(n, numel)) and a 16384-lane double
# discrete store in one program wedges the device (see module docstring).
_SP_MAX = 8192
# Tile width of the multi-block pipeline.
_BLOCK = 8192
# Widest exclusive scan the single-program scan kernel will run.
_SCAN_MAX = 32768


@libentry()
@triton.jit
def _msb_single_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    N: tl.constexpr,
    NUMEL: tl.constexpr,
    SCRATCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One-launch compaction for ``max(N, NUMEL) <= BLOCK``.

    Two unmasked stores with provably disjoint address sets:
    ``[0, count)`` receives the compacted gradient, ``[count, NUMEL)`` receives
    zeros, and every lane that owns neither is parked in the scratch tail.
    """
    lane = tl.arange(0, BLOCK)
    valid = lane < N
    m = tl.load(mask_ptr + lane, mask=valid)
    g = tl.load(grad_ptr + lane, mask=valid)

    active = valid & (m != 0)
    ones = tl.where(active, 1, 0)
    pos = tl.cumsum(ones, axis=0) - 1
    total = tl.sum(ones, axis=0)

    tgt = tl.where(active, pos, SCRATCH + lane)
    tl.store(out_ptr + tgt, g)

    pad = (lane >= total) & (lane < NUMEL)
    ztgt = tl.where(pad, lane, SCRATCH + BLOCK + lane)
    tl.store(out_ptr + ztgt, tl.zeros([BLOCK], dtype=tl.float32))


@libentry()
@triton.jit
def _msb_count_zero_kernel(
    mask_ptr,
    counts_ptr,
    out_ptr,
    N: tl.constexpr,
    NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Phase 1: per-block popcount of the mask + contiguous zero fill of ``out``.

    Both halves are guarded by uniform (program-wide) predicates only, so the
    zero fill stays a stride-1 block DMA.  ``out`` is over-allocated to a whole
    number of tiles, hence the store needs no mask.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    if pid * BLOCK < N:
        m = tl.load(mask_ptr + offs, mask=offs < N)
        ones = tl.where((offs < N) & (m != 0), 1, 0)
        tl.store(counts_ptr + pid, tl.sum(ones, axis=0))
    else:
        tl.store(counts_ptr + pid, 0)

    if pid * BLOCK < NUMEL:
        tl.store(out_ptr + offs, tl.zeros([BLOCK], dtype=tl.float32))


@libentry()
@triton.jit
def _msb_scan_kernel(counts_ptr, off_ptr, NP: tl.constexpr):
    """Phase 2: exclusive scan of the per-block counts, one program, no atomics.

    ``counts`` is allocated with ``NP`` (power of two) entries and every entry
    is written by phase 1, so the load and the store are both unmasked.  The
    result goes to a separate buffer: writing it back in place would store to
    an address the same program already stored to, which is a known device-hang
    pattern on this backend.
    """
    i = tl.arange(0, NP)
    c = tl.load(counts_ptr + i)
    tl.store(off_ptr + i, tl.cumsum(c, axis=0) - c)


@libentry()
@triton.jit
def _msb_write_kernel(
    grad_ptr,
    mask_ptr,
    off_ptr,
    out_ptr,
    N: tl.constexpr,
    SCRATCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Phase 3: local scan + unmasked scatter into the already zeroed output.

    ``SCRATCH + offs`` (not ``SCRATCH + lane``) is essential: parking every
    program's inactive lanes in the *same* ``BLOCK``-wide scratch window makes
    the target addresses collide across programs, and colliding addresses in a
    discrete store serialise -- measured 142.3 ms vs 17.4 ms for 16.7M fp16
    lanes at BLOCK=8192, everything else identical.
    """
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    offs = pid * BLOCK + lane
    valid = offs < N

    m = tl.load(mask_ptr + offs, mask=valid)
    g = tl.load(grad_ptr + offs, mask=valid)

    active = valid & (m != 0)
    ones = tl.where(active, 1, 0)
    pos = tl.load(off_ptr + pid) + tl.cumsum(ones, axis=0) - 1

    tl.store(out_ptr + tl.where(active, pos, SCRATCH + offs), g)


def _flat_contiguous(t):
    """Flat, contiguous view of ``t`` without leaving the device.

    Non-contiguous inputs are materialised through ``aten::_copy_from``, the
    layout-transfer primitive already used elsewhere in this backend; the gems
    ``Tensor.contiguous()`` path is deliberately avoided because on a strided
    *source* it has been observed to wedge the card.
    """
    if t.is_contiguous():
        return t.reshape(-1)
    dst = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    torch.ops.aten._copy_from(t, dst)
    return dst.reshape(-1)


def masked_scatter_backward(grad_output, mask, sizes):
    """Backward of ``masked_scatter`` w.r.t. ``source``.

    ``grad_output.masked_select(mask)``, zero-padded up to ``prod(sizes)`` and
    viewed as ``sizes`` -- i.e. a stream compaction.
    """
    logger.debug("GEMS_KUNLUNXIN MASKED_SCATTER_BACKWARD")

    sizes = [int(s) for s in sizes]
    numel = 1
    for s in sizes:
        numel *= s

    device = grad_output.device
    dtype = grad_output.dtype

    if numel == 0:
        return torch.empty(sizes, dtype=dtype, device=device)

    if tuple(grad_output.shape) != tuple(mask.shape):
        grad_output, mask = torch.broadcast_tensors(grad_output, mask)

    grad_flat = _flat_contiguous(grad_output)
    mask_flat = _flat_contiguous(mask)
    n = mask_flat.numel()

    if n == 0:
        return torch.zeros(numel, dtype=dtype, device=device).view(sizes)

    with torch_device_fn.device(device):
        if max(n, numel) <= _SP_MAX:
            block = max(64, triton.next_power_of_2(max(n, numel)))
            out = torch.empty(numel + 2 * block, dtype=dtype, device=device)
            _msb_single_kernel[(1,)](
                grad_flat,
                mask_flat,
                out,
                N=n,
                NUMEL=numel,
                SCRATCH=numel,
                BLOCK=block,
            )
            return out[:numel].view(sizes)

        block = _BLOCK
        n_mask_blocks = triton.cdiv(n, block)
        n_out_blocks = triton.cdiv(numel, block)
        np_blocks = triton.next_power_of_2(max(n_mask_blocks, n_out_blocks))
        assert np_blocks <= _SCAN_MAX, (
            f"masked_scatter_backward: {n} elements need {np_blocks} scan lanes, "
            f"above the single-program scan limit {_SCAN_MAX}"
        )

        # ``scratch`` is both the start of the parking area for inactive lanes
        # and an upper bound on the largest compaction target, so it must cover
        # the output tiles *and* (for the ATen-invalid but memory-unsafe case
        # numel < count <= n) the mask tiles.  The parking area itself needs one
        # slot per input element -- see _msb_write_kernel.
        scratch = max(n_out_blocks, n_mask_blocks) * block
        out = torch.empty(scratch + n_mask_blocks * block, dtype=dtype, device=device)
        counts = torch.empty(np_blocks, dtype=torch.int32, device=device)
        offsets = torch.empty(np_blocks, dtype=torch.int32, device=device)

        _msb_count_zero_kernel[(np_blocks,)](
            mask_flat, counts, out, N=n, NUMEL=numel, BLOCK=block
        )
        _msb_scan_kernel[(1,)](counts, offsets, NP=np_blocks)
        _msb_write_kernel[(n_mask_blocks,)](
            grad_flat, mask_flat, offsets, out, N=n, SCRATCH=scratch, BLOCK=block
        )

    return out[:numel].view(sizes)
