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

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# gcu400: the kernel runs as a persistent grid-stride loop and the launch grid
# is capped at the SPC count, so the hardware grid.x limit of 65535 never binds
# for any window size (the benchmark drives window_length = 2**30). Interleaved
# do_bench retune 2026-11-17: cap 24 ties or beats 48/96/192 on every core shape.
_NUM_SIPS = 24

# Launch bands (interleaved do_bench median, 2026-11-17): tiny windows want
# 512-element blocks spread over several programs (~30% off small windows);
# everything else runs B4096/nw4 persistent. This ALU-bound kernel detests
# wide blocks with few warps -- B4096/nw2 spills into a 32 ms cliff at 2^30
# -- so keep the block/warps pairing exactly as measured.
_TINY_LIMIT = 8192

# int32 offset safe domain (i0.py precedent): block_id * BLOCK + arange stays
# below 2^31 - 2^17 for BLOCK <= 4096, so flat int32 offsets cannot wrap
# negative on the tail block; larger windows defer to the public generic op.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)


@triton.jit
def _bessel_i0_poly(x):
    # Numerical Recipes' rational/polynomial approximation of I0 in the
    # shared-rsqrt form landed for i0 (2026-11-16): one rsqrt feeds both
    # yb = 3.75/ax and the exp(ax) * p / sqrt(ax) tail, replacing two IEEE
    # divisions and one sqrt; fp32 deltas vs the previous body stay at the
    # ULP level (<= ~3e-7) and low-precision outputs round identically.
    ax = tl.abs(x)
    z = ax * 0.26666666666666666
    y = z * z
    p_small = tl.fma(y, 0.0045813, 0.0360768)
    p_small = tl.fma(y, p_small, 0.2659732)
    p_small = tl.fma(y, p_small, 1.2067492)
    p_small = tl.fma(y, p_small, 3.0899424)
    p_small = tl.fma(y, p_small, 3.5156229)
    res_small = tl.fma(y, p_small, 1.0)
    r = tl.rsqrt(ax)
    yb = 3.75 * r * r
    p_big = tl.fma(yb, 0.00392377, -0.01647633)
    p_big = tl.fma(yb, p_big, 0.02635537)
    p_big = tl.fma(yb, p_big, -0.02057706)
    p_big = tl.fma(yb, p_big, 0.00916281)
    p_big = tl.fma(yb, p_big, -0.00157565)
    p_big = tl.fma(yb, p_big, 0.00225319)
    p_big = tl.fma(yb, p_big, 0.01328592)
    p_big = tl.fma(yb, p_big, 0.39894228)
    res_big = tl.exp(ax) * p_big * r
    return tl.where(ax < 3.75, res_small, res_big)


@triton.jit
def _bessel_i0_poly_scalar(x):
    # Scalar twin for the per-launch denominator: plain mul/add Horner. A
    # scalar tl.fma lowers to an extern tops_fma_fp32 call whose ICALL
    # relocation ld.lld rejects (P03 family); the vector form above is fine.
    ax = tl.abs(x)
    z = ax * 0.26666666666666666
    y = z * z
    p_small = 0.0360768 + y * 0.0045813
    p_small = 0.2659732 + y * p_small
    p_small = 1.2067492 + y * p_small
    p_small = 3.0899424 + y * p_small
    p_small = 3.5156229 + y * p_small
    res_small = 1.0 + y * p_small
    r = tl.rsqrt(ax)
    yb = 3.75 * r * r
    p_big = -0.01647633 + yb * 0.00392377
    p_big = 0.02635537 + yb * p_big
    p_big = -0.02057706 + yb * p_big
    p_big = 0.00916281 + yb * p_big
    p_big = -0.00157565 + yb * p_big
    p_big = 0.00225319 + yb * p_big
    p_big = 0.01328592 + yb * p_big
    p_big = 0.39894228 + yb * p_big
    res_big = tl.exp(ax) * p_big * r
    return tl.where(ax < 3.75, res_small, res_big)


@libentry()
@triton.jit
def kaiser_window_kernel(
    out_ptr,
    window_length,
    half_length,
    beta,
    IS_FP64: tl.constexpr,
    IS_LOW_PRECISION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offs = tl.arange(0, BLOCK_SIZE)
    num_blocks = tl.cdiv(window_length, BLOCK_SIZE)

    out_dtype = out_ptr.dtype.element_ty

    # Loop-invariant scalars, evaluated once per program: the window-width
    # reciprocal replaces the per-element IEEE division, and the constant
    # denominator I0(beta) -- the previous body recomputed it per element --
    # becomes a single multiply on the hot path.
    inv_half = 1.0 / half_length
    inv_den = 1.0 / _bessel_i0_poly_scalar(beta)

    # gcu400: persistent grid-stride loop; the host caps the launch at the
    # SPC count and each program strides over the remaining element blocks.
    for block_id in tl.range(pid, num_blocks, num_programs):
        idx = block_id * BLOCK_SIZE + offs
        mask = idx < window_length

        # GCU400 has neither a double-precision path nor a libdevice Bessel op, so
        # every dtype is evaluated in fp32 with the polynomial I0 above. The
        # low-precision branch still reproduces torch's rounding of the element
        # index into the output dtype, which becomes observable once the index
        # exceeds 2048 (bfloat16) / 16384 (float16).
        if IS_LOW_PRECISION:
            n = idx.to(out_dtype).to(tl.float32)
        else:
            n = idx.to(tl.float32)
        x = (n - half_length) * inv_half
        val = _bessel_i0_poly(beta * tl.sqrt(tl.maximum(1.0 - x * x, 0.0)))
        val = val * inv_den

        tl.store(out_ptr + idx, val.to(out_dtype), mask=mask)


def kaiser_window(
    window_length,
    periodic=True,
    beta=12.0,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=None,
):
    logger.debug("GEMS_ENFLAME KAISER_WINDOW")
    assert window_length >= 0, "window_length must be non-negative"

    if dtype is None:
        dtype = torch.get_default_dtype()
    assert dtype.is_floating_point, "kaiser_window only supports floating point dtype"

    if layout != torch.strided:
        raise ValueError("Only strided layout is supported for kaiser_window.")

    if device is None:
        device = torch.device(runtime.device.name)

    if window_length >= _INT32_SAFE_LIMIT:
        # Flat int32 offsets would wrap past the 2 Gi-element safe domain;
        # defer to the public generic implementation instead of corrupting
        # silently (i0.py flat-guard precedent).
        from flag_gems.ops.kaiser_window import kaiser_window as _public

        return _public(
            window_length,
            periodic,
            beta,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )

    out = torch.empty(
        (window_length,),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )

    if window_length == 0:
        return out
    if window_length == 1:
        return torch.fill(out, 1.0)

    # N is the effective full size: window_length + 1 would be used by a periodic
    # *sampling* convention, but aten uses N = window_length - 1 for the symmetric
    # window and N = window_length for the periodic one.
    full_length = window_length if periodic else window_length - 1

    if window_length <= _TINY_LIMIT:
        block_size, num_warps = 512, 1
    else:
        block_size, num_warps = 4096, 4
    grid = (min(triton.cdiv(window_length, block_size), _NUM_SIPS),)
    # The external llvm converter intermittently fails on cold compiles
    # (abs.py / hardsigmoid.py / i0.py precedents); retrying the launch
    # recompiles and succeeds.
    for attempt in range(3):
        try:
            kaiser_window_kernel[grid](
                out,
                window_length,
                full_length / 2.0,
                float(beta),
                dtype == torch.float64,
                dtype in (torch.float16, torch.bfloat16),
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
            return out
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "kaiser_window launch failed (attempt %d): %s", attempt + 1, exc
            )
    return out


def kaiser_window_periodic(
    window_length,
    periodic,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=None,
):
    logger.debug("GEMS_ENFLAME KAISER_WINDOW_PERIODIC")
    return kaiser_window(
        window_length,
        periodic,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )


def kaiser_window_beta(
    window_length,
    periodic,
    beta,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=None,
):
    logger.debug("GEMS_ENFLAME KAISER_WINDOW_BETA")
    return kaiser_window(
        window_length,
        periodic,
        beta,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )


# Shared registration state is attached to torch: this file is loaded once each under the module paths
# ``gcu400.ops.kaiser_window``,
# ``_enflame.gcu400.ops.kaiser_window``, and the full package path,
# so the handle attached to torch guarantees registration happens only once per process.
_ATEN_STATE_ATTR = "_flag_gems_gcu400_kaiser_window_aten_state"


# Module-level handle keeps the Library (and its kernels) alive for the
# process; the registration itself must never break the flag_gems import.
