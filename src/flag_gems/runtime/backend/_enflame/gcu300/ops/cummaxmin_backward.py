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

"""cummaxmin_backward -- atomic-free scatter-add.

Semantics: out = zeros_like(input); out.scatter_add_(dim, indices, grad_output),
accumulating in float32.

The target backend (enflame GCU) cannot legalize ``tt.atomic_rmw`` and has no
64-bit datapath, so a scatter-add must be done without atomics and the int64
index tensor is read through a zero-copy int32 view (low word, stride 2).

The indices produced by the ``cummax``/``cummin`` forward pass are
*non-decreasing* along ``dim`` (the running-argmax position never moves
backwards), therefore the set of source positions that map to one destination
``k`` is a contiguous run of ``j``.  Each destination is then written exactly
once, so a plain masked store is safe.

Kernel per (outer, inner) column:
  * stream the reduction axis in blocks of BLOCK,
  * build the run-local inclusive sum with a segmented scan keyed by ``indices``,
  * at every run end write the run total into out[..., idx[...], ...],
  * carry the still-open run's partial total across blocks.

Layout (element strides, inner size, whether lane masking is needed) is a
compile-time constant so the generated code uses contiguous vector accesses
instead of runtime-stride gather addressing.

One kernel launch does everything.  Each program owns one (outer, inner) column
of the output, so it zeroes that whole column itself and then, after an
intra-program barrier, writes the run totals into it: every output position is
written by the program that owns the column, program-ordered zeros first, so the
separate fill launch (and any grid-wide ordering requirement) disappears.  The
profile measured ~0.30 ms of per-call dispatch at 1024x1024 against 3.9 ms of
kernel time, and the fill stream at 3.3% of device time.
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _run_sum_combine(v1, k1, v2, k2):
    """Segmented (keyed) inclusive sum: restart whenever the key changes."""
    return tl.where(k1 == k2, v1 + v2, v2), k2


@triton.jit
def _column_scatter_kernel(
    grad_ptr,
    idx_ptr,
    out_ptr,
    S,
    GS_O: tl.constexpr,
    GS_R: tl.constexpr,
    GS_I: tl.constexpr,
    IS_O: tl.constexpr,
    IS_R: tl.constexpr,
    IS_I: tl.constexpr,
    INNER: tl.constexpr,
    NEED_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    col = tl.program_id(0)
    o = col // INNER
    i = col % INNER

    g_base = o * GS_O + i * GS_I
    i_base = o * IS_O + i * IS_I
    out_base = o * (S * INNER) + i

    lane = tl.arange(0, BLOCK)

    # Phase 1: zero the whole output column (the positions that receive no
    # contribution).  Pure store stream, no loads.
    zcast = tl.zeros([BLOCK], dtype=tl.float32).to(out_ptr.dtype.element_ty)
    for b0 in range(0, S, BLOCK):
        off = b0 + lane
        if NEED_MASK:
            tl.store(out_ptr + out_base + off * INNER, zcast, mask=off < S)
        else:
            tl.store(out_ptr + out_base + off * INNER, zcast)
    # The zeros above and the run totals below are written by the same program
    # but by different lanes, so the barrier's memory fence orders them.
    tl.debug_barrier()

    prev_k = -1
    carry = 0.0

    for b0 in range(0, S, BLOCK):
        off = b0 + lane
        if NEED_MASK:
            mask = off < S
            g = tl.load(grad_ptr + g_base + off * GS_R, mask=mask, other=0.0)
            k = tl.load(idx_ptr + i_base + off * IS_R, mask=mask, other=0)
            kn = tl.load(
                idx_ptr + i_base + (off + 1) * IS_R, mask=off + 1 < S, other=-1
            )
            is_end = (kn != k) & mask
        else:
            g = tl.load(grad_ptr + g_base + off * GS_R)
            k = tl.load(idx_ptr + i_base + off * IS_R)
            kn = tl.load(
                idx_ptr + i_base + (off + 1) * IS_R, mask=off + 1 < S, other=-1
            )
            is_end = kn != k

        gf = g.to(tl.float32)
        seg, _k = tl.associative_scan((gf, k.to(tl.int32)), 0, _run_sum_combine)

        # The run that was already open at block entry can only be the first run
        # of the block (keys are monotone), so a key compare is enough.
        val = seg + tl.where(k == prev_k, carry, 0.0)
        tl.store(out_ptr + out_base + k * INNER, val, mask=is_end)

        # Carry the still-open (trailing) run into the next block.
        kmax = tl.max(k, axis=0)
        trail = tl.sum(tl.where(k == kmax, gf, 0.0), axis=0)
        carry = tl.where(kmax == prev_k, carry + trail, trail)
        prev_k = kmax


# ---------------------------------------------------------------------------
# Host entry point
# ---------------------------------------------------------------------------


def _pick_block(red):
    """BLOCK is the reduction tile, and the warp count is pinned to one because
    the block-wide collectives (keyed scan + two tail reductions) get much more
    expensive as the number of warps they span grows (measured: 8 warps 2.7x
    slower than 4, 2 warps 1.32x faster than 4).  The tile itself was measured
    monotone at one warp over 8/16/32 elements per thread (BLOCK 256/512/1024:
    1.43 / 1.61 / 1.78 geo_mean), so it is pushed to 64 elements per thread."""
    block = 2048
    if red < block:
        block = triton.next_power_of_2(max(red, 32))
    return block, 1


def run(grad_output, input, indices, dim):
    ndim = grad_output.ndim
    if ndim == 0:
        dim = 0
    else:
        dim = dim % ndim

    shape = grad_output.shape
    device = grad_output.device
    n_elements = grad_output.numel()

    out = torch.empty(shape, dtype=grad_output.dtype, device=device)
    if n_elements == 0:
        return out

    if ndim == 0:
        red = 1
        inner = 1
        outer = 1
    else:
        red = shape[dim]
        inner = 1
        for d in range(dim + 1, ndim):
            inner *= shape[d]
        outer = 1
        for d in range(dim):
            outer *= shape[d]

    if grad_output.is_contiguous():
        gs_o, gs_r, gs_i = red * inner, inner, 1
    else:
        gs_r = grad_output.stride(dim) if ndim else 1
        gs_i = grad_output.stride(dim + 1) if dim + 1 < ndim else 1
        gs_o = red * gs_r

    # GCU has no 64-bit datapath: an int64 tensor is stored as *one* int32 word
    # per element (verified on this box: view(torch.int32) of a 6-element int64
    # tensor returns 12 words whose tail is not the high halves but 0x3F800000
    # padding).  Reading it as an int32 pair stride-2 "low word" therefore
    # returns interleaved garbage: idx=[0,1,2,2] was seen as [0,2,<garbage>,..],
    # which both corrupts the result and sends the scatter store to wild
    # addresses (SIP exception + device reset).  A dtype conversion is the
    # correct and on this backend copy-free way to get the int32 view.
    if indices.dtype == torch.int32:
        idx_i32 = indices
    else:
        idx_i32 = indices.to(torch.int32)

    if idx_i32.is_contiguous():
        is_o, is_r, is_i = red * inner, inner, 1
    else:
        is_r = idx_i32.stride(dim) if ndim else 1
        is_i = idx_i32.stride(dim + 1) if dim + 1 < ndim else 1
        is_o = red * is_r

    block, warps = _pick_block(red)
    need_mask = (red % block) != 0
    grid1 = (outer * inner,)
    _column_scatter_kernel[grid1](
        grad_output,
        idx_i32,
        out,
        red,
        gs_o,
        gs_r,
        gs_i,
        is_o,
        is_r,
        is_i,
        inner,
        need_mask,
        block,
        num_warps=warps,
    )

    return out


def cummaxmin_backward(grad_output, input, indices, dim):
    """ATen entry point for ``cummaxmin_backward``.

    ``input`` is accepted for signature compatibility; the implementation only
    needs the gradient, the indices and the reduction dimension.
    """
    return run(grad_output, input, indices, dim)
