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

from flag_gems.ops.special_modified_bessel_k0 import (
    special_modified_bessel_k0_out as default_special_modified_bessel_k0_out,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64, so only fp32 is specialized.
_SUPPORTED_DTYPES = {torch.float32}

_NUM_WARPS = 4
_MAX_RANK = 8

_INT32_LIMIT = 2147483647


@triton.jit
def _k0_f32(x):
    """Modified Bessel function of the second kind, order 0, for float32 x.

    Two-region approximation (Abramowitz & Stegun 9.8.5 / 9.8.6).  All
    intermediates are float32; expensive exact operations are expressed with
    hardware approximations (rsqrt, exp2, log2) and reciprocal multiplies.
    """
    u = tl.rsqrt(x)  # 1/sqrt(x); only used by the (x > 2) branch

    # ---- region 1: 0 <= x <= 2 -------------------------------------------
    # K0(x) = -ln(x/2) * I0(x) + P((x/2)^2)
    y = x * x * 0.25
    p = -0.57721566 + y * (
        0.42278420
        + y
        * (
            0.23069756
            + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.00000740)))
        )
    )
    # I0(x), |x| <= 2  (t = x/3.75, |t| <= 0.5333)
    t = x * 0.26666666666666666
    t2 = t * t
    i0 = 1.0 + t2 * (
        3.5156229
        + t2
        * (
            3.0899424
            + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))
        )
    )
    # -ln(x/2) = -(log2(x) - 1) * ln2
    small = p - ((tl.log2(x) - 1.0) * 0.6931471805599453) * i0

    # ---- region 2: x > 2 -------------------------------------------------
    # K0(x) = exp(-x) * Q(2/x) / sqrt(x)
    z = 2.0 * u * u
    q = 1.25331414 + z * (
        -0.07832358
        + z
        * (
            0.02189568
            + z * (-0.01062446 + z * (0.00587872 + z * (-0.00251540 + z * 0.00053208)))
        )
    )
    large = q * tl.exp2(-x * 1.4426950408889634) * u

    r = tl.where(x <= 2.0, small, large)
    # NaN inputs keep NaN (both comparisons below are false).
    r = tl.where(x < 0.0, float("nan"), r)
    r = tl.where(x == 0.0, float("inf"), r)
    return r


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=8),
        triton.Config({"BLOCK": 4096}, num_warps=8),
        triton.Config({"BLOCK": 8192}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _k0_contig_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    EVEN: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(x_ptr + offs, eviction_policy="evict_first")
    else:
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=1.0, eviction_policy="evict_first")
    r = _k0_f32(x.to(tl.float32))
    if EVEN:
        tl.store(
            out_ptr + offs,
            r.to(out_ptr.dtype.element_ty),
            eviction_policy="evict_first",
        )
    else:
        tl.store(
            out_ptr + offs,
            r.to(out_ptr.dtype.element_ty),
            mask=mask,
            eviction_policy="evict_first",
        )


@triton.jit
def _k0_strided_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    xs0,
    xs1,
    xs2,
    xs3,
    xs4,
    xs5,
    xs6,
    xs7,
    os0,
    os1,
    os2,
    os3,
    os4,
    os5,
    os6,
    os7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    rem = offs
    xoff = tl.zeros([BLOCK], dtype=tl.int64)
    ooff = tl.zeros([BLOCK], dtype=tl.int64)

    if RANK > 0:
        c = rem % d0
        rem = rem // d0
        xoff += c * xs0
        ooff += c * os0
    if RANK > 1:
        c = rem % d1
        rem = rem // d1
        xoff += c * xs1
        ooff += c * os1
    if RANK > 2:
        c = rem % d2
        rem = rem // d2
        xoff += c * xs2
        ooff += c * os2
    if RANK > 3:
        c = rem % d3
        rem = rem // d3
        xoff += c * xs3
        ooff += c * os3
    if RANK > 4:
        c = rem % d4
        rem = rem // d4
        xoff += c * xs4
        ooff += c * os4
    if RANK > 5:
        c = rem % d5
        rem = rem // d5
        xoff += c * xs5
        ooff += c * os5
    if RANK > 6:
        c = rem % d6
        rem = rem // d6
        xoff += c * xs6
        ooff += c * os6
    if RANK > 7:
        c = rem % d7
        rem = rem // d7
        xoff += c * xs7
        ooff += c * os7

    x = tl.load(x_ptr + xoff, mask=mask, other=1.0)
    r = _k0_f32(x.to(tl.float32))
    tl.store(out_ptr + ooff, r.to(out_ptr.dtype.element_ty), mask=mask)


def _pack_layout(x, out):
    dims = []
    xstr = []
    ostr = []
    for d, xs, os_ in zip(x.shape, x.stride(), out.stride()):
        if d == 1:
            continue
        dims.append(int(d))
        xstr.append(int(xs))
        ostr.append(int(os_))
    dims.reverse()
    xstr.reverse()
    ostr.reverse()
    rank = max(len(dims), 1)
    while len(dims) < _MAX_RANK:
        dims.append(1)
        xstr.append(0)
        ostr.append(0)
    return dims, xstr, ostr, rank


def _launch(x, out):
    n = x.numel()
    if n == 0:
        return
    if x.is_contiguous() and out.is_contiguous():
        # Every autotune BLOCK divides 8192, so a multiple of 8192 is a safe
        # predictor of an exact tiling for any selected configuration.
        even = (n % 8192) == 0
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
        _k0_contig_kernel[grid](
            x,
            out,
            n,
            EVEN=even,
            I64=(n > _INT32_LIMIT),
        )
    else:
        dims, xstr, ostr, rank = _pack_layout(x, out)
        grid = (triton.cdiv(n, 1024),)
        _k0_strided_kernel[grid](
            x,
            out,
            n,
            dims[0],
            dims[1],
            dims[2],
            dims[3],
            dims[4],
            dims[5],
            dims[6],
            dims[7],
            xstr[0],
            xstr[1],
            xstr[2],
            xstr[3],
            xstr[4],
            xstr[5],
            xstr[6],
            xstr[7],
            ostr[0],
            ostr[1],
            ostr[2],
            ostr[3],
            ostr[4],
            ostr[5],
            ostr[6],
            ostr[7],
            RANK=rank,
            BLOCK=1024,
            num_warps=_NUM_WARPS,
        )


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernel."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def special_modified_bessel_k0_out(self, out=None, **kwargs):
    """special_modified_bessel_k0.out: write K0(self) into out and return out."""
    logger.debug("GEMS_MTHREADS SPECIAL_MODIFIED_BESSEL_K0_OUT")

    x = self
    if out is None and "out" in kwargs:
        out = kwargs["out"]
    if not isinstance(x, torch.Tensor):
        raise TypeError("self must be a Tensor")
    if out is None:
        if x.is_floating_point():
            out = torch.empty_like(x)
        else:
            out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    else:
        if out.dtype != x.dtype:
            raise TypeError("out dtype must match input dtype")
        if out.device != x.device:
            raise TypeError("out device must match input device")
    if not _use_triton_kernel(x):
        return default_special_modified_bessel_k0_out(x, out)
    with torch_device_fn.device(x.device):
        _launch(x, out)
    return out
