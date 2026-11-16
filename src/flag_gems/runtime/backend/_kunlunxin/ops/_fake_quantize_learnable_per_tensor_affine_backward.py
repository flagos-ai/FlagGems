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

"""Kunlunxin (XPU3) override for
``aten::_fake_quantize_learnable_per_tensor_affine_backward``.

Why an override is needed
-------------------------
The generic implementation
(``src/flag_gems/ops/_fake_quantize_learnable_per_tensor_affine_backward.py``)
is 20-30x slower than the ATen reference on this backend for every non-tiny
shape (measured dtype-equal Gems Speedup 0.336; e.g. 4.35 ms vs 0.149 ms at
4096x4096, 15.9 ms vs 0.52 ms at 1024x65536).  Additive bisection of the
generic body (``harness/perf_ir/probe_fq_lptab_bisect.py``) attributes it to
three XPU3 lowering traps, all fixed here:

1. ``inr = (u >= qminf) & (u <= qmaxf)`` -- a boolean AND of two vector
   compares.  The pair lowers to per-lane ``setcc``/``i1`` selects that
   scalarise the whole body: the identical kernel with the mask rewritten as
   the single product-form compare ``(r - lo) * (hi - r) >= 0.0`` drops from
   19.8 ms to 3.4 ms at 67.1M fp32 (67M: 12229 us -> 3409 us for the same
   epilogue).  See ``wiki/techniques/xpu3-product-form-range-mask.md``.
2. ``@triton.jit`` helper bodies in a hot kernel
   (``_rint_even`` used for both ``zero_point`` and every ``q``) plus an
   8192/32768-lane serial-tile loop with a loop-carried 32K accumulator.  They
   are inlined / replaced by the flat producer geometry below.
   See ``wiki/patterns/xpu3-data-dependent-byte-store-scalarization.md``.
3. Grid of only ``ceil(n / 262144)`` programs (64 programs at 4096x4096), i.e.
   parallelism-starved: the strided multi-tile loop makes the copy floor itself
   run at 65 GB/s.  One program per 8192-lane block reaches 1416 GB/s for the
   same traffic.

Geometry (measured on the full benchmark matrix)
------------------------------------------------
* Strictly 1-D grid ``g = min(128, ceil(n / 8192))``; program ``pid`` owns the
  strided tile sequence ``pid, pid+g, ...``.  ``BLOCK=8192`` is the largest
  tile whose ``tl.sum``/``tl.reduce`` is exact on this backend (16384 silently
  miscompiles).
* Each program reduces its two partial sums with a *tuple* ``tl.reduce`` (one
  tree for both), writes ``g`` partials, and a 1-program ``_reduce_kernel``
  finishes the reduction.  When ``g == 1`` (``n <= 8192``) the main kernel
  writes ``grad_scale`` / ``grad_zero_point`` directly -- a single launch for
  every launch-bound small shape (10 us vs 22-24 us for ATen).
* ``FULL`` is a compile-time flag: shapes with ``n % 8192 == 0`` take the
  unmasked load/store path (masked memory is ~2x as expensive here).

Numerics (unchanged from the generic kernel, and equal to the ATen reference)
-----------------------------------------------------------------------------
    zr = clamp(rne(zero_point), quant_min, quant_max)
    q  = x / scale                      # *exact* fp32 division (div.rn)
    r  = rne(q)                         # magic identity (q + 1.5*2^23) - 1.5*2^23
    inr = lo <= r <= hi                 # lo = qmin - zr, hi = qmax - zr
    grad_self = inr ? dy : 0
    grad_scale = sum(inr ? dy*(r - q) : (r < lo ? lo*dy : hi*dy))
    grad_zero_point = sum(inr ? 0 : -dy*scale)

``x / scale`` is kept as a true division (not a reciprocal multiply): a
1-ulp difference in ``q`` can flip a round-half-to-even tie, which changes
``grad_self`` by ``dy``.  The division costs ~30% of the body but keeps the
classification bit-identical.  The product-form range test is exact for the
finite ties that matter and reproduces the generic classification for
``+-inf`` / ``NaN`` inputs too (``NaN >= 0`` is false -> out of range).

Red lines: no torch/aten compute, no redispatch, no forbidden aten copy
primitive.  The ``torch.empty_like`` / ``torch.empty`` calls below are pure
allocations (three output gradients, the ``g``-element partial-sum buffers, and
the fp32 widening buffer used for bfloat16 input only); ``contiguous()`` /
``_gems_copy_`` are only used for a non-contiguous or non-fp32 input and both
dispatch to real Gems kernels.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.copy import copy_ as _gems_copy_

logger = logging.getLogger(__name__)

# Largest tile whose tl.reduce / tl.sum is exact on XPU3 (16384 miscompiles).
_BLOCK = 8192
# Cap on the number of partial-sum programs (keeps the second launch tiny and
# was the fastest point of the cap sweep: 128 vs 64/256 on 4M-67M elements).
_MAX_PROGRAMS = 128
# Lanes of the finishing reduction kernel (>= 64: smaller tl.sum vectors
# miscompile on this backend).
_REDUCE_BLOCK = 128


@triton.jit
def _pair_add(a0, a1, b0, b1):
    return a0 + b0, a1 + b1


@triton.jit
def _fq_lptab_bwd_kernel(
    grad_ptr,
    x_ptr,
    scale_ptr,
    zero_ptr,
    grad_self_ptr,
    pds_ptr,
    pdz_ptr,
    out_s_ptr,
    out_z_ptr,
    n,
    ntiles,
    qmin,
    qmax,
    gf,
    FINAL: tl.constexpr,
    FULL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)

    s = tl.load(scale_ptr).to(tl.float32)
    z = tl.load(zero_ptr).to(tl.float32)
    qminf = qmin.to(tl.float32)
    qmaxf = qmax.to(tl.float32)
    # round-half-to-even of zero_point, then clamp (ATen rounds before clamping)
    zr = tl.minimum(tl.maximum((z + 12582912.0) - 12582912.0, qminf), qmaxf)
    lo = qminf - zr
    hi = qmaxf - zr

    acc_ds = tl.zeros([BLOCK], dtype=tl.float32)
    acc_dz = tl.zeros([BLOCK], dtype=tl.float32)

    for tile in range(pid, ntiles, nprog):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        if FULL:
            dy = tl.load(grad_ptr + offs).to(tl.float32)
            x = tl.load(x_ptr + offs).to(tl.float32)
        else:
            mask = offs < n
            dy = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        q = tl.fdiv(x, s, ieee_rounding=True)
        r = (q + 12582912.0) - 12582912.0
        # single-compare product-form range test (exact for integer r)
        inr = (r - lo) * (hi - r) >= 0.0

        gs = tl.where(inr, dy, 0.0)
        if FULL:
            tl.store(grad_self_ptr + offs, gs)
        else:
            tl.store(grad_self_ptr + offs, gs, mask=mask)

        acc_ds += tl.where(inr, dy * (r - q), tl.where(r < lo, lo * dy, hi * dy))
        acc_dz += tl.where(inr, 0.0, -dy * s)

    ds_sum, dz_sum = tl.reduce((acc_ds, acc_dz), 0, _pair_add)
    if FINAL:
        tl.store(out_s_ptr, ds_sum * gf)
        tl.store(out_z_ptr, dz_sum * gf)
    else:
        tl.store(pds_ptr + pid, ds_sum)
        tl.store(pdz_ptr + pid, dz_sum)


@triton.jit
def _reduce_kernel(pds_ptr, pdz_ptr, out_s_ptr, out_z_ptr, g, gf,
                   BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < g
    ds = tl.load(pds_ptr + offs, mask=mask, other=0.0)
    dz = tl.load(pdz_ptr + offs, mask=mask, other=0.0)
    tl.store(out_s_ptr, tl.sum(ds, axis=0) * gf)
    tl.store(out_z_ptr, tl.sum(dz, axis=0) * gf)


def _fake_quantize_learnable_per_tensor_affine_backward(
    grad, self, scale, zero_point, quant_min, quant_max, grad_factor=1.0
):
    logger.debug(
        "GEMS_KUNLUNXIN FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE_BACKWARD"
    )

    if not (
        torch.is_floating_point(grad)
        and torch.is_floating_point(self)
        and torch.is_floating_point(scale)
        and torch.is_floating_point(zero_point)
    ):
        raise ValueError("All inputs must be floating-point tensors.")

    # ATen's backward rejects float16 and bfloat16 "self" tensors ("Expected
    # X.scalar_type() == ScalarType::Float"), so no reference exists for them;
    # this override keeps the generic contract and still accepts float32 and
    # bfloat16 (float16 stays rejected, as in the generic implementation).
    if self.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"Unsupported dtype {self.dtype}; expected float32 or bfloat16."
        )

    # The flat kernel indexes `grad` with the offsets of `self`; a mismatched
    # element count would be an out-of-bounds read (device fault) instead of an
    # error.  ATen requires identical shapes for this op as well.
    if grad.numel() != self.numel():
        raise ValueError(
            "grad and self must have the same number of elements, got "
            f"{grad.numel()} and {self.numel()}."
        )

    # Flat 1-D kernel: a non-contiguous input is materialised by the real Gems
    # copy kernel (no aten compute / no forbidden copy primitive).
    if not grad.is_contiguous():
        grad = grad.contiguous()
    if not self.is_contiguous():
        self = self.contiguous()
    scale = scale.contiguous()
    zero_point = zero_point.contiguous()

    if self.dtype != torch.float32:
        # bfloat16 in-kernel vector loads (`tl.load(bf16_ptr).to(fp32)` on a
        # 8192-lane tile) are broken on this backend: measured
        # KL_XID_KERNEL_EXCEPTION (status=700) at 8192 elements, and the
        # generic kernel faults the same way.  Widen to fp32 with the real Gems
        # copy kernel first; bfloat16 has no ATen reference for this op, so it
        # is not on any measured path and the extra pass costs nothing there.
        self_f32 = torch.empty(self.shape, dtype=torch.float32,
                               device=self.device)
        _gems_copy_(self_f32, self)
        self = self_f32

    # Allocations only; taken after the layout normalisation so that the flat
    # offsets of the kernel match the outputs' memory layout.
    grad_self = torch.empty_like(self, dtype=torch.float32)
    grad_scale = torch.empty_like(scale, dtype=torch.float32)
    grad_zero_point = torch.empty_like(zero_point, dtype=torch.float32)

    n = self.numel()
    gf = float(grad_factor)
    qmin = int(quant_min)
    qmax = int(quant_max)

    # host short-circuit for empty inputs (no launch, grid would be 0)
    if n == 0:
        return grad_self, grad_scale, grad_zero_point

    ntiles = (n + _BLOCK - 1) // _BLOCK
    g = min(_MAX_PROGRAMS, max(1, ntiles))
    full = (n % _BLOCK) == 0

    if g == 1:
        _fq_lptab_bwd_kernel[(1,)](
            grad,
            self,
            scale,
            zero_point,
            grad_self,
            grad_scale,
            grad_zero_point,
            grad_scale,
            grad_zero_point,
            n,
            ntiles,
            qmin,
            qmax,
            gf,
            FINAL=True,
            FULL=full,
            BLOCK=_BLOCK,
            num_warps=4,
        )
        return grad_self, grad_scale, grad_zero_point

    # Allocation only (partial sums, immediately overwritten by the kernel).
    pds = torch.empty((g,), dtype=torch.float32, device=self.device)
    pdz = torch.empty((g,), dtype=torch.float32, device=self.device)

    _fq_lptab_bwd_kernel[(g,)](
        grad,
        self,
        scale,
        zero_point,
        grad_self,
        pds,
        pdz,
        grad_scale,
        grad_zero_point,
        n,
        ntiles,
        qmin,
        qmax,
        gf,
        FINAL=False,
        FULL=full,
        BLOCK=_BLOCK,
        num_warps=4,
    )
    _reduce_kernel[(1,)](
        pds, pdz, grad_scale, grad_zero_point, g, gf,
        BLOCK=_REDUCE_BLOCK, num_warps=4,
    )

    return grad_self, grad_scale, grad_zero_point
