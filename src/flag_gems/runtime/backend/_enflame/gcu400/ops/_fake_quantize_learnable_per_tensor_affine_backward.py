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

from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)


@triton.jit
def _rint_even(v):
    # round-half-to-even via the magic-constant trick, exact for |v| < 2**22.
    # gcu400 lowers the __ef_nearbyintf libdevice extern to an opaque call
    # (~14x slower than this on a bandwidth-bound pointwise body), while every
    # lane whose rounded value is consumed inside the quantization range has
    # |v| < 2**22; out-of-range lanes only feed the range classification, where
    # the trick stays within +/-2 of nearbyint for arbitrarily large |v|.
    # Unlike the forward op's (x*inv + C) pattern, the backward's two-step sum
    # (zr_f + x*inv, variable addend) was adversarially probed on this backend
    # (direct-tie and cancellation-tie families + random sweeps vs aten CPU):
    # no fma contraction divergence, so no min/max barrier is needed here.
    return (v + 12582912.0) - 12582912.0


@triton.jit
def _fq_tiny_kernel(
    grad_ptr,
    x_ptr,
    scale_ptr,
    zp_ptr,
    gs_ptr,
    out_s_ptr,
    out_z_ptr,
    n,
    num_tiles,
    qmin,
    qmax,
    gf,
    BLOCK: tl.constexpr,
):
    s = tl.load(scale_ptr).to(tl.float32)
    inv = tl_extra_shim.rcp_rn(s)
    z = tl.load(zp_ptr).to(tl.float32)
    qmin_i = qmin.to(tl.int64)
    qmax_i = qmax.to(tl.int64)
    qminf = qmin.to(tl.float32)
    qmaxf = qmax.to(tl.float32)
    # aten backward zero point: (int64) clamp(z + 0.5f, qmin, qmax). The clamp is
    # in float, then the cast truncates toward zero (matches static_cast<int64_t>).
    zr = tl.minimum(tl.maximum(z + 0.5, qminf), qmaxf).to(tl.int64)
    zr_f = zr.to(tl.float32)
    lo = (qmin_i - zr).to(tl.float32)
    hi = (qmax_i - zr).to(tl.float32)

    acc_ds = 0.0
    acc_dz = 0.0
    for tile in range(0, num_tiles):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        dy = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # aten CPU: int64_t xqi = nearbyint(zero_point + X * inv_scale); the
        # zero_point is added before the round, then the integral float narrows
        # to int64 and the range test is an int64 comparison.  For every fp32
        # value u the int64 range test equals the plain float compare (the
        # narrowing is exact for in-range magnitudes and both flag anything
        # larger as out of range), except that x86 maps every float->int64
        # conversion it cannot represent (NaN, +-inf, |u| >= 2**63) to
        # INT64_MIN, which lands in the qmin (lo) branch.  The hi_sel below
        # reproduces that with plain ordered compares: only an ordered value
        # above qmax but within int64 range takes the hi branch, everything
        # else (NaN/inf/huge) falls through to lo, matching aten.  No negated
        # compare is used: gcu400 canonicalizes ~(u>=q) into ordered u<q and
        # thereby loses the unordered NaN case.
        u = _rint_even(zr_f + x * inv)
        inr = (u >= qminf) & (u <= qmaxf)
        uc = tl.minimum(tl.maximum(u, qminf), qmaxf)
        xfqi = (uc - zr_f) * s
        tl.store(gs_ptr + offs, tl.where(inr, dy, 0.0), mask=mask)
        hi_sel = (u > qmaxf) & (u <= 9223372036854775808.0)
        ds_v = tl.where(
            inr,
            dy * (xfqi - x) * inv,
            tl.where(hi_sel, hi * dy, lo * dy),
        )
        dz_v = tl.where(inr, 0.0, -dy * s)
        acc_ds += tl.sum(ds_v, axis=0)
        acc_dz += tl.sum(dz_v, axis=0)

    tl.store(out_s_ptr, acc_ds * gf)
    tl.store(out_z_ptr, acc_dz * gf)


@triton.jit
def _fake_quantize_learnable_per_tensor_affine_backward_kernel(
    grad_ptr,
    x_ptr,
    scale_ptr,
    zp_ptr,
    gs_ptr,
    pds_ptr,
    pdz_ptr,
    n,
    qmin,
    qmax,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    num_tiles = (n + (BLOCK - 1)) // BLOCK

    s = tl.load(scale_ptr).to(tl.float32)
    inv = tl_extra_shim.rcp_rn(s)
    z = tl.load(zp_ptr).to(tl.float32)
    qmin_i = qmin.to(tl.int64)
    qmax_i = qmax.to(tl.int64)
    qminf = qmin.to(tl.float32)
    qmaxf = qmax.to(tl.float32)
    zr = tl.minimum(tl.maximum(z + 0.5, qminf), qmaxf).to(tl.int64)
    zr_f = zr.to(tl.float32)
    lo = (qmin_i - zr).to(tl.float32)
    hi = (qmax_i - zr).to(tl.float32)

    acc_ds = tl.zeros([BLOCK], dtype=tl.float32)
    acc_dz = tl.zeros([BLOCK], dtype=tl.float32)

    for tile in range(pid, num_tiles, nprog):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        dy = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        u = _rint_even(zr_f + x * inv)
        inr = (u >= qminf) & (u <= qmaxf)
        uc = tl.minimum(tl.maximum(u, qminf), qmaxf)
        xfqi = (uc - zr_f) * s

        tl.store(gs_ptr + offs, tl.where(inr, dy, 0.0), mask=mask)

        # The clamp above is kept even though ``uc == u`` on the in-range branch:
        # folding it away stayed bit-exact in the probe but cost 1.5% (1024^3)
        # to 11% (16M) through the core benchmark on this board (2026-11-20), so
        # the compiler schedules the clamped form better.
        hi_sel = (u > qmaxf) & (u <= 9223372036854775808.0)
        acc_ds += tl.where(
            inr,
            dy * (xfqi - x) * inv,
            tl.where(hi_sel, hi * dy, lo * dy),
        )
        acc_dz += tl.where(inr, 0.0, -dy * s)

    tl.store(pds_ptr + pid, tl.sum(acc_ds, axis=0))
    tl.store(pdz_ptr + pid, tl.sum(acc_dz, axis=0))


@triton.jit
def _reduce_kernel(pds_ptr, pdz_ptr, out_s_ptr, out_z_ptr, g, gf, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < g
    ds = tl.load(pds_ptr + offs, mask=mask, other=0.0)
    dz = tl.load(pdz_ptr + offs, mask=mask, other=0.0)
    tl.store(out_s_ptr, tl.sum(ds, axis=0) * gf)
    tl.store(out_z_ptr, tl.sum(dz, axis=0) * gf)


_G = 512
_WORK_PER_PROG = 131072
_TINY_BLOCK = 4096
_TINY_LIMIT = 2 * _TINY_BLOCK
# Static launch geometry (measured on gcu400 at n=2**30 fp32, 12GB of traffic,
# with the magic-constant rounding body): BLOCK=16384/num_warps=2 with 512
# strided programs -> 5.66ms (2277 GB/s vs a 2555 GB/s trivial-copy floor);
# BLOCK=32768/num_warps=4 -> 7.36ms, BLOCK=8192/num_warps=2 -> 7.09ms,
# BLOCK=65536 exceeds the 917504-byte DSM budget.  The __ef_nearbyintf extern
# previously used here dominated the old kernel: 75.3ms for the same tile.
_MAIN_BLOCK = 16384
_MAIN_WARPS = 2


def _fake_quantize_learnable_per_tensor_affine_backward(
    grad, self, scale, zero_point, quant_min, quant_max, grad_factor=1.0
):
    logger.debug("GEMS_ENFLAME FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE_BACKWARD")

    if not (
        torch.is_floating_point(grad)
        and torch.is_floating_point(self)
        and torch.is_floating_point(scale)
        and torch.is_floating_point(zero_point)
    ):
        raise ValueError("All inputs must be floating-point tensors.")

    # The ATen reference rejects float16 gradients ("Expected dY_.scalar_type()
    # == Float"); only float32 and bfloat16 inputs are supported.
    if self.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"Unsupported dtype {self.dtype}; expected float32 or bfloat16."
        )

    # The ATen reference implementation always returns float32 gradients
    # (it rejects float16 grads and promotes bfloat16 grads to float32), so we
    # match that contract: all three output gradients are float32.
    grad_self = torch.empty_like(self, dtype=torch.float32)
    grad_scale = torch.empty_like(scale, dtype=torch.float32)
    grad_zero_point = torch.empty_like(zero_point, dtype=torch.float32)

    n = self.numel()
    gf = float(grad_factor)
    qmin = int(quant_min)
    qmax = int(quant_max)

    if n <= _TINY_LIMIT:
        _fq_tiny_kernel[(1,)](
            grad,
            self,
            scale,
            zero_point,
            grad_self,
            grad_scale,
            grad_zero_point,
            n,
            (_TINY_BLOCK + n - 1) // _TINY_BLOCK if n > 0 else 0,
            qmin,
            qmax,
            gf,
            BLOCK=_TINY_BLOCK,
            num_warps=4,
        )
        return grad_self, grad_scale, grad_zero_point

    pds = torch.empty((_G,), dtype=torch.float32, device=self.device)
    pdz = torch.empty((_G,), dtype=torch.float32, device=self.device)

    g = min(_G, max(1, (n + _WORK_PER_PROG - 1) // _WORK_PER_PROG))

    _fake_quantize_learnable_per_tensor_affine_backward_kernel[(g,)](
        grad,
        self,
        scale,
        zero_point,
        grad_self,
        pds,
        pdz,
        n,
        qmin,
        qmax,
        BLOCK=_MAIN_BLOCK,
        num_warps=_MAIN_WARPS,
    )

    _reduce_kernel[(1,)](
        pds, pdz, grad_scale, grad_zero_point, g, gf, BLOCK=_G, num_warps=4
    )

    return grad_self, grad_scale, grad_zero_point
