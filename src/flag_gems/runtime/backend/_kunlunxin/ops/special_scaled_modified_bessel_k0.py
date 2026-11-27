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
"""Kunlunxin implementation of aten::special_scaled_modified_bessel_k0."""

import logging

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice as _ld
except Exception:  # pragma: no cover
    from triton.language.extra.cuda import libdevice as _ld

logger = logging.getLogger(__name__)

_GAMMA = tl.constexpr(0.577215664901532860606512090082402431)
_LN2 = tl.constexpr(0.693147180559945309417232121458176568)
# Smallest positive normal float32; below it 0.5 * x is not representable.
_MIN_NORMAL = tl.constexpr(1.1754943508222875e-38)
_BLOCK = 1024
_NUM_WARPS = 4

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

_SUPPORTED_INPUT_DTYPES = frozenset(
    (
        torch.float32,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    )
)


@triton.jit
def _poly(t, COEFFS: tl.constexpr, NCOEF: tl.constexpr):
    acc = tl.full(t.shape, COEFFS[0], t.dtype)
    for i in tl.static_range(1, NCOEF):
        acc = acc * t + COEFFS[i]
    return acc


@triton.jit
def _bessel_k0e(w):
    half = 0.5 * w
    t = half * half
    i0 = _poly(t, _IA, 9)
    s = _poly(t, _SA, 9)
    # log(0.5 * w) == log(w) - log(2), a form that needs no subnormal product
    # and is therefore used above the normal range. Down here 0.5 * w is not
    # representable anyway, so the direct form keeps the reference's own
    # degenerate result for the smallest subnormal input (+inf).
    lg = tl.where(w >= _MIN_NORMAL, tl.log(w) - _LN2, tl.log(half))
    small = tl.exp(w) * (-(lg + _GAMMA) * i0 + s)
    rs = _ld.rsqrt(w)
    z = 4.0 * (rs * rs) - 1.0
    large = _poly(z, _CB, 9) * rs
    return tl.where(w > 2.0, large, small)


@triton.jit
def _k0e_flat(x_ptr, y_ptr, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    if EVEN:
        tl.store(y_ptr + offs, _bessel_k0e(tl.load(x_ptr + offs)))
    else:
        m = offs < N
        tl.store(
            y_ptr + offs,
            _bessel_k0e(tl.load(x_ptr + offs, mask=m, other=1.0)),
            mask=m,
        )


def _run(x, out=None):
    if not isinstance(x, torch.Tensor):
        raise TypeError("special_scaled_modified_bessel_k0: x must be a Tensor")
    if x.device.type != "cuda":
        raise RuntimeError("special_scaled_modified_bessel_k0 requires a CUDA tensor")
    if x.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise RuntimeError(
            f"special_scaled_modified_bessel_k0 not implemented for dtype {x.dtype}"
        )
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                "special_scaled_modified_bessel_k0.out: out must be a Tensor"
            )
        if not out.is_floating_point():
            raise RuntimeError(
                "scaled_modified_bessel_k0.out: output tensor must be floating point"
            )
        if out.shape != x.shape:
            raise RuntimeError("scaled_modified_bessel_k0.out: shape mismatch")
        if out.device != x.device:
            raise RuntimeError("scaled_modified_bessel_k0.out: device mismatch")

    # Integral and bool inputs are promoted on the host: the elementwise
    # exponential of a value converted inside the kernel is miscompiled on this
    # backend, while a real float32 input takes the accurate path.
    if x.dtype != torch.float32:
        xw = x.contiguous().to(torch.float32)
    else:
        xw = x.contiguous()
    if out is None:
        y = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        target = y
    else:
        y = out
        target = (
            y
            if y.is_contiguous() and y.dtype == torch.float32
            else torch.empty(x.shape, dtype=torch.float32, device=x.device)
        )

    n = xw.numel()
    if n:
        block = _BLOCK
        grid = (triton.cdiv(n, block),)
        _k0e_flat[grid](
            xw,
            target,
            n,
            BLOCK=block,
            EVEN=(n % block) == 0,
            num_warps=_NUM_WARPS,
        )
        if target is not y:
            y.copy_(target)
    return y


def special_scaled_modified_bessel_k0(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SCALED_MODIFIED_BESSEL_K0")
    return _run(x, out)


def special_scaled_modified_bessel_k0_out(x, out):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SCALED_MODIFIED_BESSEL_K0_OUT")
    return _run(x, out)
