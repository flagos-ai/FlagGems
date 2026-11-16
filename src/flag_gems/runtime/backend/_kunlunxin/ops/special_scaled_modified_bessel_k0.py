# Copyright 2026, The FlagOS Contributors.
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

"""Kunlunxin override for ``aten::special_scaled_modified_bessel_k0``.

Two XPU3-specific defects in the generic implementation are corrected here,
leaving the float64 / unsupported-dtype / ``out=`` paths delegated unchanged:

1. **Integral / bool inputs**: the generic kernel loads an integer pointer and
   casts it with ``tl.load(...).to(tl.float32)`` *inside* the kernel. On XPU3
   that cast makes the ``w > 2`` branch test pick the large branch for
   ``x in {1, 2}`` (verified: K0e(1) came back as 0.842 instead of 1.144).
   Fix: promote the operand to float32 on the host (a dtype cast, the same
   promotion native ATen applies), then run the float32 Triton kernel.

2. **Smallest-normal input**: the generic small branch evaluates
   ``log(0.5 * x)``. For ``x`` in ``[finfo.tiny, 2*finfo.tiny)`` the product
   ``0.5*x`` is subnormal, and XPU3 ``libdevice.log`` flushes a subnormal
   argument to zero, so the result becomes ``+inf`` instead of ~88. Fix:
   evaluate ``log(x) - log(2)`` (``log(x)`` stays normal for these inputs;
   true subnormals still flush to ``-inf`` -> ``+inf``, matching the reference).

The polynomial tables and branch structure are the generic ones; only the log
argument and the host-side integer promotion change.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.special_scaled_modified_bessel_k0 import (
    special_scaled_modified_bessel_k0 as _generic_special_scaled_modified_bessel_k0,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

try:  # Triton >= 3.0 canonical path
    from triton.language.extra import libdevice as _ld
except Exception:  # pragma: no cover - fallback for older layouts
    from triton.language.extra.cuda import libdevice as _ld

logger = logging.getLogger(__name__)

_GAMMA = tl.constexpr(0.577215664901532860606512090082402431)
_LN2 = tl.constexpr(0.6931471805599453)
_BLOCK_SMALL = 512
_BLOCK_LARGE = 2048
_NUM_WARPS = 4
_LARGE_N = 10 << 20

# Integral/bool inputs are promoted to float32 (native ATen semantics).
_INTEGRAL_DTYPES = frozenset(
    (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    )
)

# Same trimmed tables as the generic implementation.
_IA = tl.constexpr(
    (
        6.151187326782565e-10,
        3.936759889140842e-08,
        1.9290123456790124e-06,
        6.944444444444444e-05,
        0.001736111111111111,
        0.027777777777777776,
        0.25,
        1.0,
        1.0,
    )
)
_SA = tl.constexpr(
    (
        1.6718048413148328e-09,
        1.0207455998272324e-07,
        4.72608024691358e-06,
        0.0001585648148148148,
        0.0036168981481481473,
        0.05092592592592592,
        0.375,
        1.0,
        0.0,
    )
)
_CB = tl.constexpr(
    (
        1.0730641014921863e-06,
        -2.924484614160032e-06,
        6.702810789178686e-06,
        -2.4178549539405763e-05,
        9.966534872760174e-05,
        -0.00047991778302170065,
        0.0030328793513160757,
        -0.031071451426337666,
        1.2185953387445558,
    )
)


@triton.jit
def _poly(t, COEFFS: tl.constexpr, NCOEF: tl.constexpr):
    acc = tl.full(t.shape, COEFFS[0], t.dtype)
    for i in tl.static_range(1, NCOEF):
        acc = acc * t + COEFFS[i]
    return acc


@triton.jit
def _k0e(w):
    """K0e(x) in float32 with the subnormal-safe logarithm."""
    half = 0.5 * w
    t = half * half
    i0 = _poly(t, _IA, 9)
    s = _poly(t, _SA, 9)
    ln2 = tl.full((), _LN2, w.dtype)
    lg = _ld.log(w) - ln2
    small = _ld.exp(w) * (-(lg + _GAMMA) * i0 + s)
    rs = _ld.rsqrt(w)
    z = 4.0 * (rs * rs) - 1.0
    large = _poly(z, _CB, 9) * rs
    return tl.where(w > 2.0, large, small)


@libentry()
@triton.jit
def _k0e_flat_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    if EVEN:
        # Full blocks: unmasked loads/stores let the vectorizer widen them.
        r = _k0e(tl.load(x_ptr + offs).to(tl.float32))
        tl.store(y_ptr + offs, r)
    else:
        mask = offs < n
        # Clamped unmasked load: a masked tail load returns garbage here.
        safe = tl.minimum(offs, n - 1)
        r = _k0e(tl.load(x_ptr + safe).to(tl.float32))
        tl.store(y_ptr + offs, r, mask=mask)


def _pick_block(n):
    return _BLOCK_LARGE if n >= _LARGE_N else _BLOCK_SMALL


def special_scaled_modified_bessel_k0(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SCALED_MODIFIED_BESSEL_K0")
    if out is not None:
        # Keep the generic out= semantics (dtype promotion / validation); only
        # lift integral inputs to float32 so the in-kernel integer cast defect
        # cannot bite there either.
        if isinstance(x, torch.Tensor) and x.dtype in _INTEGRAL_DTYPES:
            x = x.to(torch.float32)
        return _generic_special_scaled_modified_bessel_k0(x, out=out)
    if not isinstance(x, torch.Tensor) or x.device.type != "cuda":
        # Let the generic implementation raise its canonical errors.
        return _generic_special_scaled_modified_bessel_k0(x, out=out)
    if x.dtype in _INTEGRAL_DTYPES:
        x = x.to(torch.float32)
    elif x.dtype != torch.float32:
        # float64 / half / bfloat16 / float8 / complex: unchanged generic path
        # (float64 is platform-limited, the rest raise the native message).
        return _generic_special_scaled_modified_bessel_k0(x, out=out)

    xc = x.contiguous()
    y = torch.empty(xc.shape, dtype=torch.float32, device=xc.device)
    n = xc.numel()
    if n == 0:
        return y
    block = _pick_block(n)
    grid = (triton.cdiv(n, block),)
    with torch_device_fn.device(x.device):
        _k0e_flat_kernel[grid](
            xc,
            y,
            n,
            BLOCK=block,
            EVEN=(n % block) == 0,
            num_warps=_NUM_WARPS,
        )
    return y
