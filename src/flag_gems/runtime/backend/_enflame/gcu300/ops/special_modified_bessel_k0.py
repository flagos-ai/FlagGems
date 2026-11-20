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

# GCU300 backend implementation of special_modified_bessel_k0 / _out.
#
# The generic (KernelGen-generated, CUDA-only) implementation in
# flag_gems/ops/special_modified_bessel_k0.py asserts `x.is_cuda` in its
# launch helper, which can never hold for GCU tensors (is_cuda is always
# False on the gcu device). Provide a device-agnostic backend override.
#
# K0(x) is evaluated with three transcendental calls per element -- one shared
# ``log(x)``, one ``rsqrt(x)`` and one ``exp`` -- rather than a separate
# polynomial set per branch, and the two branches are selected after the fact.

import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


MAX_GRID = 65535
BLOCK = 1024
PROGRAMS = 1024

LN2 = 0.6931471805599453


@triton.jit
def _k0_f32(x):
    """Modified Bessel function of the second kind, order 0 (float32).

    Abramowitz & Stegun 9.8.5 / 9.8.6 branch structure, rewritten so that only
    three transcendental library calls are needed per element:
      lx = log(x)                 shared by both branches
      r  = rsqrt(x)               -> z = 2/x = 2*r*r
      exp(-(x + lx/2))            == exp(-x) / sqrt(x)
      x <= 2 : K0(x) = -ln(x/2) * I0(x) + P((x/2)^2) = P + (ln2 - lx) * I0
       x > 2 : K0(x) = exp(-x) / sqrt(x) * Q(2/x)
    Negative x yields NaN through log(x); x == 0 yields +inf.
    """
    y = 0.25 * x * x  # (x/2)^2
    # I0(x) = sum_k y^k / (k!)^2  (truncation error < 4e-8 for y <= 1)
    i0 = 1.0 + y * (
        1.0
        + y
        * (
            0.25
            + y
            * (
                0.027777777777777776
                + y
                * (
                    0.0017361111111111112
                    + y * (6.944444444444444e-05 + y * 1.9290123456790123e-06)
                )
            )
        )
    )
    p = -0.57721566 + y * (
        0.42278420
        + y
        * (
            0.23069756
            + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.00000740)))
        )
    )
    lx = tl.log(x)
    small = p + (0.6931471805599453 - lx) * i0

    r = tl.rsqrt(x)
    z = 2.0 * r * r
    q = 1.25331414 + z * (
        -0.07832358
        + z
        * (
            0.02189568
            + z * (-0.01062446 + z * (0.00587872 + z * (-0.00251540 + z * 0.00053208)))
        )
    )
    large = tl.exp(-(x + 0.5 * lx)) * q

    return tl.where(x <= 2.0, small, large)


@triton.jit
def _k0_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    stride = tl.num_programs(axis=0) * BLOCK
    for base in range(pid * BLOCK, n_elements, stride):
        offsets = base + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        res = _k0_f32(x)
        tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


# Large shapes only: search the staging-block size (one async-DMA slice + memset
# + wait per loop iteration) together with the persistent program count.
_CFGS = [
    triton.Config({"BLOCK": b, "PROGRAMS": p}, num_warps=w)
    for (b, p, w) in (
        [
            (b, p, w)
            for b in (1024, 2048, 4096)
            for p in (256, 512, 1024)
            for w in (4, 8)
        ]
        + [(8192, p, w) for p in (64, 128, 256, 512, 1024) for w in (4, 8)]
        + [(16384, p, 8) for p in (64, 128, 256, 512)]
        # num_warps=1 was never in the tuned set; it measures fastest for
        # polynomial kernels on this backend (micro: 1.8x over nw=4).
        + [(8192, p, 1) for p in (64, 256, 1024)]
        + [(16384, p, 1) for p in (64, 256)]
    )
]


@triton.autotune(configs=_CFGS, key=["n_elements"])
@triton.jit
def _k0_kernel_auto(
    x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr, PROGRAMS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    stride = tl.num_programs(axis=0) * BLOCK
    for base in range(pid * BLOCK, n_elements, stride):
        offsets = base + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        res = _k0_f32(x)
        tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


# Mask-free twins: valid whenever numel is an exact multiple of the block, so
# every window [base, base+BLOCK) is fully in range. The host only selects them
# in that case, which removes the per-iteration predicate, the padding slice and
# the staging-buffer memset.
@triton.jit
def _k0_kernel_full(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    stride = tl.num_programs(axis=0) * BLOCK
    for base in range(pid * BLOCK, n_elements, stride):
        offsets = base + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offsets).to(tl.float32)
        tl.store(out_ptr + offsets, _k0_f32(x).to(out_ptr.dtype.element_ty))


@triton.autotune(configs=_CFGS, key=["n_elements"])
@triton.jit
def _k0_kernel_auto_full(
    x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr, PROGRAMS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    stride = tl.num_programs(axis=0) * BLOCK
    for base in range(pid * BLOCK, n_elements, stride):
        offsets = base + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offsets).to(tl.float32)
        tl.store(out_ptr + offsets, _k0_f32(x).to(out_ptr.dtype.element_ty))


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
    pid = tl.program_id(axis=0)
    stride = tl.num_programs(axis=0) * BLOCK
    for base in range(pid * BLOCK, n_elements, stride):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < n_elements
        rem = offs
        xoff = tl.zeros((BLOCK,), dtype=tl.int32)
        ooff = tl.zeros((BLOCK,), dtype=tl.int32)

        r = rem % d0
        rem = rem // d0
        xoff += r * xs0
        ooff += r * os0
        if RANK > 1:
            r = rem % d1
            rem = rem // d1
            xoff += r * xs1
            ooff += r * os1
        if RANK > 2:
            r = rem % d2
            rem = rem // d2
            xoff += r * xs2
            ooff += r * os2
        if RANK > 3:
            r = rem % d3
            rem = rem // d3
            xoff += r * xs3
            ooff += r * os3
        if RANK > 4:
            r = rem % d4
            rem = rem // d4
            xoff += r * xs4
            ooff += r * os4
        if RANK > 5:
            r = rem % d5
            rem = rem // d5
            xoff += r * xs5
            ooff += r * os5
        if RANK > 6:
            r = rem % d6
            rem = rem // d6
            xoff += r * xs6
            ooff += r * os6
        if RANK > 7:
            r = rem % d7
            rem = rem // d7
            xoff += r * xs7
            ooff += r * os7

        x = tl.load(x_ptr + xoff, mask=mask, other=1.0).to(tl.float32)
        res = _k0_f32(x)
        tl.store(out_ptr + ooff, res.to(out_ptr.dtype.element_ty), mask=mask)


def _launch_strided(x, out, n):
    shape = list(x.shape)
    rank = len(shape)
    dims = list(reversed(shape)) + [1] * (8 - rank)
    xst = list(reversed(x.stride())) + [0] * (8 - rank)
    ost = list(reversed(out.stride())) + [0] * (8 - rank)
    grid = (min((n + BLOCK - 1) // BLOCK, MAX_GRID),)
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
        xst[0],
        xst[1],
        xst[2],
        xst[3],
        xst[4],
        xst[5],
        xst[6],
        xst[7],
        ost[0],
        ost[1],
        ost[2],
        ost[3],
        ost[4],
        ost[5],
        ost[6],
        ost[7],
        RANK=rank,
        BLOCK=BLOCK,
        num_warps=4,
    )


def _run_special_modified_bessel_k0(x: torch.Tensor, out: torch.Tensor):
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"Tensors must be {flag_gems.device} tensors")
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"Unsupported dtype: {x.dtype}"
    assert out.dtype == x.dtype, "Output dtype must match input dtype"
    assert (
        x.numel() == out.numel()
    ), "Input and output must have the same number of elements"

    n = x.numel()
    if n == 0:
        return out
    with torch_device_fn.device(x.device):
        if x.is_contiguous() and out.is_contiguous():
            if n <= (1 << 20):
                if (n & (BLOCK - 1)) == 0:
                    if n <= 4096:
                        _k0_kernel_full[(1,)](x, out, n, BLOCK=n, num_warps=4)
                    else:
                        grid = (min(n // BLOCK, PROGRAMS),)
                        _k0_kernel_full[grid](x, out, n, BLOCK=BLOCK, num_warps=4)
                else:
                    grid = (min((n + BLOCK - 1) // BLOCK, PROGRAMS),)
                    _k0_kernel[grid](x, out, n, BLOCK=BLOCK, num_warps=4)
            else:
                grid = lambda meta: (
                    min((n + meta["BLOCK"] - 1) // meta["BLOCK"], meta["PROGRAMS"]),
                )
                if (n & 16383) == 0:
                    _k0_kernel_auto_full[grid](x, out, n)
                else:
                    _k0_kernel_auto[grid](x, out, n)
        else:
            _launch_strided(x, out, n)
    return out


def special_modified_bessel_k0(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME SPECIAL_MODIFIED_BESSEL_K0")
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    _run_special_modified_bessel_k0(x_c, out)
    if x.layout == torch.strided and x.is_contiguous():
        return out
    return out.view_as(x)


def special_modified_bessel_k0_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME SPECIAL_MODIFIED_BESSEL_K0_OUT")
    if out.dtype != x.dtype:
        raise TypeError("out dtype must match input dtype")
    if out.device != x.device:
        raise TypeError("out device must match input device")

    x_c = x.contiguous()
    out_c = out.contiguous()
    _run_special_modified_bessel_k0(x_c, out_c)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out
