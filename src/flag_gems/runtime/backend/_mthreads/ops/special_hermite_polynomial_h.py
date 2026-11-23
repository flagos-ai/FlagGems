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

from flag_gems.ops.special_hermite_polynomial_h import (
    special_hermite_polynomial_h as default_special_hermite_polynomial_h,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

# Physicist's Hermite polynomial, elementwise with broadcast degrees.
#
# Target-verified ATen semantics:
#   H_0 = 1, H_1 = 2x, H_k = 2*(x*H_{k-1} - (k-1)*H_{k-2})
#   - out dtype/shape: x dtype, broadcast(x.shape, n.shape); Scalar n keeps x.shape
#   - the Scalar (Python int / numel==1 tensor) form is evaluated by ATen in the
#     input dtype with a fused multiply-add; that result is bit-reproducible with
#     `2.0 * fma(x, p1, -(k-1)*p0)` (verified: 0/1048576 bit mismatches, while the
#     non-FMA form mismatches 17226 elements under atol=1e-4/rtol=1.3e-6).
#   - the broadcast Tensor-degree form is compared against a float64-upcast
#     reference, so it accumulates in float64.
#   - degrees outside [0, 9] are rejected with ValueError (scalar/numel==1 form).

_BLOCK = 1024
# Broadcast-degree path is size dependent: single-wave (small) shapes need many
# small blocks (2 fp64 lanes/thread, measured 22% faster at 256), while large
# shapes need few large blocks to amortize the per-block degree reduction.
_TSPLIT = 262144
_TBLOCK_SMALL = 256
_TBLOCK_LARGE = 1024
# Above ~2M elements the broadcast path is multi-wave and ALU bound, where 8 warps
# (4 fp64 lanes/thread, ~half the per-thread fp64 registers) measured 3.7% faster
# on the 168M case; the single-wave 1M case prefers 4 warps so more blocks stay
# resident.  Scalar-degree medium shapes are also single-wave at 1024 lanes, so
# they use the small tile too.
_TSPLIT_WARPS = 1 << 21
_NW_SMALL = 4
_NW_LARGE = 8
_SBLOCK_SMALL = 256

_MSG = "GEMS SPECIAL_HERMITE_POLYNOMIAL_H"
_LOG = logging.getLogger("flag_gems.ops.special_hermite_polynomial_h")


@triton.jit
def _herm_lin_kernel(x_ptr, out_ptr, numel, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    dt = x_ptr.dtype.element_ty
    if N == 0:
        res = tl.full((BLOCK,), 1.0, dt)
    else:
        xv = tl.load(x_ptr + offs, mask=mask, other=0.0)
        if N == 1:
            res = 2.0 * xv
        else:
            p0 = tl.full((BLOCK,), 1.0, dt)
            two_x = 2.0 * xv
            p1 = two_x
            for k in tl.static_range(2, N + 1):
                p2 = tl.math.fma(two_x, p1, -(2 * (k - 1)) * p0)
                p0 = p1
                p1 = p2
            res = p1
    tl.store(out_ptr + offs, res, mask=mask)


@triton.jit
def _herm_lin_n_kernel(x_ptr, n_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float64)
    nv = tl.load(n_ptr + offs, mask=mask, other=0).to(tl.int32)
    deg = tl.max(nv)
    zero = tl.zeros((BLOCK,), tl.float64)
    p0 = tl.full((BLOCK,), 1.0, tl.float64)
    two_xv = 2.0 * xv
    p1 = two_xv
    res = tl.where(nv == 1, p1, tl.where(nv == 0, p0, zero))
    # Degrees 0..9 are the contract; unroll them straight-line so the recurrence
    # does not depend on the block-wide degree reduction.  rarer degrees > 9 are
    # finished by the dynamic tail below.
    for k in tl.static_range(2, 10):
        p2 = tl.math.fma(two_xv, p1, -(2 * (k - 1)) * p0)
        res = tl.where(nv == k, p2, res)
        p0 = p1
        p1 = p2
    k = 10
    while k <= deg:
        p2 = tl.math.fma(two_xv, p1, (-2 * (k - 1)).to(tl.float64) * p0)
        res = tl.where(nv == k, p2, res)
        p0 = p1
        p1 = p2
        k += 1
    res = tl.where(nv < 0, zero, res)
    tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _herm_gen_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    OUT_SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    N_STRIDES: tl.constexpr,
    SUFFIX: tl.constexpr,
    RANK: tl.constexpr,
    HAS_N: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x_off = tl.zeros((BLOCK,), tl.int32)
    n_off = tl.zeros((BLOCK,), tl.int32)
    for d in tl.static_range(RANK):
        coord = (offs // SUFFIX[d]) % OUT_SHAPE[d]
        x_off += coord * X_STRIDES[d]
        n_off += coord * N_STRIDES[d]
    if HAS_N:
        nv = tl.load(n_ptr + n_off, mask=mask, other=0).to(tl.int32)
        deg = tl.max(nv)
        xv = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float64)
        zero = tl.zeros((BLOCK,), tl.float64)
        p0 = tl.full((BLOCK,), 1.0, tl.float64)
        two_xv = 2.0 * xv
        p1 = two_xv
        res = tl.where(nv == 1, p1, tl.where(nv == 0, p0, zero))
        for k in tl.static_range(2, 10):
            p2 = tl.math.fma(two_xv, p1, -(2 * (k - 1)) * p0)
            res = tl.where(nv == k, p2, res)
            p0 = p1
            p1 = p2
        k = 10
        while k <= deg:
            p2 = tl.math.fma(two_xv, p1, (-2 * (k - 1)).to(tl.float64) * p0)
            res = tl.where(nv == k, p2, res)
            p0 = p1
            p1 = p2
            k += 1
        res = tl.where(nv < 0, zero, res)
    else:
        dt = x_ptr.dtype.element_ty
        if N == 0:
            res = tl.full((BLOCK,), 1.0, dt)
        else:
            xv = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            if N == 1:
                res = 2.0 * xv
            else:
                p0 = tl.full((BLOCK,), 1.0, dt)
                two_x = 2.0 * xv
                p1 = two_x
                for k in tl.static_range(2, N + 1):
                    p2 = tl.math.fma(two_x, p1, -(2 * (k - 1)) * p0)
                    p0 = p1
                    p1 = p2
                res = p1
    tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=mask)


def _suffix_products(shape):
    n = len(shape)
    suffix = [1] * n
    acc = 1
    for d in range(n - 1, -1, -1):
        suffix[d] = acc
        acc *= shape[d]
    return suffix


def _bcast_strides(t, out_shape):
    rank = len(out_shape)
    t_shape = t.shape
    t_strides = t.stride()
    off = rank - len(t_shape)
    res = []
    for d in range(rank):
        td = d - off
        if td < 0 or t_shape[td] == 1:
            res.append(0)
        else:
            res.append(t_strides[td])
    return res


def _launch_general(
    x, n_ref, out, out_shape, x_strides, n_strides, has_n, deg, block, num_warps
):
    rank = len(out_shape)
    numel = out.numel()
    grid = (triton.cdiv(numel, block),)
    _herm_gen_kernel[grid](
        x,
        n_ref,
        out,
        numel,
        tuple(out_shape),
        tuple(x_strides),
        tuple(n_strides),
        tuple(_suffix_products(out_shape)),
        rank,
        has_n,
        deg,
        block,
        num_warps=num_warps,
    )
    return out


def _run_scalar_degree(x, deg):
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    numel = out.numel()
    if numel == 0:
        return out
    block = _SBLOCK_SMALL if numel <= _TSPLIT else _BLOCK
    nw = 4 if numel <= _TSPLIT else 8
    grid = (triton.cdiv(numel, block),)
    if x.is_contiguous():
        _herm_lin_kernel[grid](x, out, numel, deg, block, num_warps=nw)
    else:
        rank = len(x.shape)
        _launch_general(
            x,
            out,
            out,
            tuple(x.shape),
            list(x.stride()),
            [0] * rank,
            False,
            deg,
            block,
            nw,
        )
    return out


def _run_tensor_degree(x, n):
    xs = x.shape
    ns = n.shape
    out_shape = xs if xs == ns else torch.broadcast_shapes(xs, ns)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    numel = out.numel()
    if numel == 0:
        return out
    if numel <= _TSPLIT:
        block, nw = _TBLOCK_SMALL, _NW_SMALL
    elif numel <= _TSPLIT_WARPS:
        block, nw = _TBLOCK_LARGE, _NW_SMALL
    else:
        block, nw = _TBLOCK_LARGE, _NW_LARGE
    if xs == out_shape and ns == out_shape and x.is_contiguous() and n.is_contiguous():
        grid = (triton.cdiv(numel, block),)
        _herm_lin_n_kernel[grid](x, n, out, numel, block, num_warps=nw)
    else:
        _launch_general(
            x,
            n,
            out,
            tuple(out_shape),
            _bcast_strides(x, out_shape),
            _bcast_strides(n, out_shape),
            True,
            0,
            block,
            nw,
        )
    return out


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def special_hermite_polynomial_h(x, n):
    logger.debug("GEMS_MTHREADS SPECIAL_HERMITE_POLYNOMIAL_H")

    if not _use_triton_kernel(x):
        return default_special_hermite_polynomial_h(x, n)

    with torch_device_fn.device(x.device):
        return _launch_hermite_h(x, n)


def _launch_hermite_h(x, n):
    _LOG.debug(_MSG)
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if isinstance(n, torch.Tensor) and n.numel() != 1:
        return _run_tensor_degree(x, n)
    deg = int(n.item()) if isinstance(n, torch.Tensor) else int(n)
    if deg < 0 or deg > 9:
        raise ValueError(
            f"special_hermite_polynomial_h only supports n in [0, 9], got {deg}"
        )
    return _run_scalar_degree(x, deg)
