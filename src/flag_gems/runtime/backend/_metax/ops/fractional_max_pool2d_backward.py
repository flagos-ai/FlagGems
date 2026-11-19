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

"""Triton fractional_max_pool2d_backward: scatter-add grad_output by forward indices.

grad_input is zero-initialised and then, for every output element o of
grad_output, grad_input_flat[plane(o) * H * W + indices[o]] += grad_output[o],
where plane(o) = o // (OH * OW) and indices are plane-relative (0 .. H*W-1).
Overlapping pooling windows make several outputs select the same input pixel,
so the accumulation must be an add (atomic_add), not a store.

Fast path (small planes): one program owns a whole (batch, channel) plane, so it
can zero that plane and then scatter its own outputs into it without any
cross-program race.  That folds the zero fill into the scatter kernel and keeps
the atomic destination lines in L2 between the two phases.
"""

import torch
import triton
import triton.language as tl

# Per-device cap on concurrently resident programs (device metadata only).
_DEV_CAP = {}
_CAP = 0

# Planes up to this many elements (per (batch, channel) plane) are fused into the
# single plane-resident kernel; larger planes fall back to the split path.
_PLANE_MAX = 8192


@triton.jit
def _fmp_bwd_zero(grad_in_ptr, n_in, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        grad_in_ptr + offs,
        tl.zeros([BLOCK], dtype=grad_in_ptr.dtype.element_ty),
        mask=offs < n_in,
    )


@triton.jit
def _fmp_bwd_plane(
    grad_out_ptr,
    idx_ptr,
    grad_in_ptr,
    n_chunks,
    n_in,
    n_out,
    ZN,
    OHW,
    HW,
    BLOCK: tl.constexpr,
):
    # One chunk of G whole planes per program iteration: zero the chunk, barrier
    # so every store of this CTA is visible, then scatter this chunk's outputs.
    # No other program ever touches these grad_input elements, so no global
    # coordination is needed, and the destination lines stay in L2 between the
    # two phases.  G is chosen host-side so a chunk is ~4K elements, which keeps
    # the blocks full even when a single plane is much smaller than BLOCK.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    ON = ZN // HW * OHW
    for c in tl.range(pid, n_chunks, nprog):
        d_base = c.to(tl.int64) * ZN
        d_ptr = grad_in_ptr + d_base
        z_end = tl.minimum(ZN, (n_in - d_base).to(tl.int32))
        for z in tl.range(0, z_end, BLOCK):
            zo = z + offs
            tl.store(
                d_ptr + zo,
                tl.zeros([BLOCK], dtype=grad_in_ptr.dtype.element_ty),
                mask=zo < z_end,
            )
        tl.debug_barrier()
        o_base = c.to(tl.int64) * ON
        o_end = tl.minimum(ON, (n_out - o_base).to(tl.int32))
        for o in tl.range(0, o_end, BLOCK):
            oo = o + offs
            m = oo < o_end
            idx = tl.load(idx_ptr + o_base + oo, mask=m, other=0)
            val = tl.load(grad_out_ptr + o_base + oo, mask=m, other=0.0)
            tl.atomic_add(
                d_ptr + idx.to(tl.int64),
                val.to(grad_in_ptr.dtype.element_ty),
                mask=m & (idx >= 0) & (idx < HW),
            )


@triton.jit
def _fmp_bwd_scatter(
    grad_out_ptr,
    idx_ptr,
    grad_in_ptr,
    n_out,
    OHW,
    HW,
    BLOCK: tl.constexpr,
):
    # Cyclic (grid-strided) assignment: every program walks the output range in
    # lockstep with the others, so the set of grad_input lines being atomically
    # updated at any instant is bounded by num_programs * BLOCK instead of the
    # whole grad_input plane space.  That keeps the atomic destination resident
    # in L2 and avoids repeated DRAM read-modify-write traffic per line.
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    for start in tl.range(pid * BLOCK, n_out, step):
        offs = start + tl.arange(0, BLOCK)
        m = offs < n_out
        idx = tl.load(idx_ptr + offs, mask=m, other=0)
        val = tl.load(grad_out_ptr + offs, mask=m, other=0.0)
        plane = offs // OHW
        dst = plane.to(tl.int64) * HW + idx.to(tl.int64)
        tl.atomic_add(
            grad_in_ptr + dst,
            val.to(grad_in_ptr.dtype.element_ty),
            mask=m & (idx >= 0) & (idx < HW),
        )


@triton.jit
def _fmp_bwd_scatter_strided(
    grad_out_ptr,
    idx_ptr,
    grad_in_ptr,
    n_out,
    C,
    OHW,
    OW,
    HW,
    gsn,
    gsc,
    gsh,
    gsw,
    isn,
    isc,
    ish,
    isw,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_out
    plane = offs // OHW
    rem = offs - plane * OHW
    oh = rem // OW
    ow = rem - oh * OW
    b = plane // C
    c = plane - b * C
    goff = b * gsn + c * gsc + oh * gsh + ow * gsw
    ioff = b * isn + c * isc + oh * ish + ow * isw
    idx = tl.load(idx_ptr + ioff, mask=m, other=0)
    val = tl.load(grad_out_ptr + goff, mask=m, other=0.0)
    dst = plane.to(tl.int64) * HW + idx.to(tl.int64)
    tl.atomic_add(
        grad_in_ptr + dst,
        val.to(grad_in_ptr.dtype.element_ty),
        mask=m & (idx >= 0) & (idx < HW),
    )


def fractional_max_pool2d_backward(
    grad_output, self, kernel_size, output_size, indices
):
    global _CAP
    grad_input = torch.empty_like(
        self, dtype=grad_output.dtype, memory_format=torch.contiguous_format
    )
    n_in = self.numel()
    if n_in != 0:
        n_out = grad_output.numel()
        BLOCK = 1024
        if n_out != 0:
            gs = grad_output.shape
            ss = self.shape
            OHW = gs[-2] * gs[-1]
            HW = ss[-2] * ss[-1]
            if grad_output.is_contiguous() and indices.is_contiguous():
                planes = n_in // HW
                if HW <= _PLANE_MAX and planes * OHW == n_out:
                    cap = _CAP
                    if cap == 0:
                        cap = (
                            2
                            * torch.cuda.get_device_properties(
                                grad_input.get_device()
                            ).multi_processor_count
                        )
                        _CAP = cap
                    if planes < cap:
                        cap = planes
                    g = 4096 // HW
                    if g < 1:
                        g = 1
                    q = planes // cap
                    if g > q:
                        g = q
                    n_chunks = (planes + g - 1) // g
                    n_prog = n_chunks if n_chunks < cap else cap
                    _fmp_bwd_plane[(n_prog,)](
                        grad_output,
                        indices,
                        grad_input,
                        n_chunks,
                        n_in,
                        n_out,
                        g * HW,
                        OHW,
                        HW,
                        BLOCK=BLOCK,
                    )
                    return grad_input
                _fmp_bwd_zero[((n_in + BLOCK - 1) // BLOCK,)](
                    grad_input, n_in, BLOCK=BLOCK
                )
                n_blocks = (n_out + BLOCK - 1) // BLOCK
                if n_blocks > 256:
                    dev = grad_output.get_device()
                    cap = _DEV_CAP.get(dev)
                    if cap is None:
                        cap = (
                            2
                            * torch.cuda.get_device_properties(
                                dev
                            ).multi_processor_count
                        )
                        _DEV_CAP[dev] = cap
                    if n_blocks > cap:
                        n_blocks = cap
                _fmp_bwd_scatter[(n_blocks,)](
                    grad_output, indices, grad_input, n_out, OHW, HW, BLOCK=BLOCK
                )
            else:
                C = grad_output.shape[-3]
                OW = grad_output.shape[-1]
                gst = grad_output.stride()
                ix = indices.stride()
                if grad_output.dim() == 4:
                    gsn, gsc, gsh, gsw = gst
                    isn, isc, ish, isw = ix
                else:
                    gsc, gsh, gsw = gst
                    isc, ish, isw = ix
                    gsn = 0
                    isn = 0
                _fmp_bwd_zero[((n_in + BLOCK - 1) // BLOCK,)](
                    grad_input, n_in, BLOCK=BLOCK
                )
                _fmp_bwd_scatter_strided[((n_out + BLOCK - 1) // BLOCK,)](
                    grad_output,
                    indices,
                    grad_input,
                    n_out,
                    C,
                    OHW,
                    OW,
                    HW,
                    gsn,
                    gsc,
                    gsh,
                    gsw,
                    isn,
                    isc,
                    ish,
                    isw,
                    BLOCK=BLOCK,
                )
        else:
            _fmp_bwd_zero[((n_in + BLOCK - 1) // BLOCK,)](grad_input, n_in, BLOCK=BLOCK)
    return grad_input
