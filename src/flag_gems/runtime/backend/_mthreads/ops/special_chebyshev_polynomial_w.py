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

"""Triton implementation of the Chebyshev polynomial of the fourth kind W_n(x).

    W_0(x) = 1
    W_1(x) = 2x + 1
    W_n(x) = 2x * W_{n-1}(x) - W_{n-2}(x)

The three-term recurrence is a polynomial identity valid for every real x, so it
also covers |x| > 1 (matching the ATen reference, which is finite there).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.special_chebyshev_polynomial_w import (
    special_chebyshev_polynomial_w as default_special_chebyshev_polynomial_w,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_BLOCK = 1024
_WARPS = 4


class _UnsupportedDtype(NotImplementedError, ValueError, TypeError):
    """Raised for compute dtypes the operator has no implementation for."""


@triton.jit
def _cheb_w_flat_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    DEG: tl.constexpr,
    HAS_N: tl.constexpr,
    SCALAR_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask)
    w0 = x * 0.0 + 1.0
    w1 = 2.0 * x + 1.0

    if HAS_N:
        if SCALAR_N:
            nv = tl.load(n_ptr).to(tl.int32)
            for i in tl.range(2, nv + 1):
                tmp = w1
                w1 = 2.0 * x * w1 - w0
                w0 = tmp
            res = tl.where(nv == 0, w0, w1)
            res = tl.where(nv < 0, res * 0.0, res)
        else:
            n_i = tl.load(n_ptr + offs, mask=mask, other=0).to(tl.int32)
            nmax = tl.max(n_i, axis=0)
            for i in tl.range(2, nmax + 1):
                w2 = 2.0 * x * w1 - w0
                act = i <= n_i
                w0 = tl.where(act, w1, w0)
                w1 = tl.where(act, w2, w1)
            res = tl.where(n_i == 0, w0, w1)
            res = tl.where(n_i < 0, res * 0.0, res)
    else:
        if DEG < 0:
            res = w0 * 0.0
        elif DEG == 0:
            res = w0
        elif DEG == 1:
            res = w1
        else:
            for _ in tl.static_range(2, DEG + 1):
                tmp = w1
                w1 = 2.0 * x * w1 - w0
                w0 = tmp
            res = w1

    tl.store(out_ptr + offs, res, mask=mask)


@triton.jit
def _cheb_w_strided_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    DEG: tl.constexpr,
    HAS_N: tl.constexpr,
    SCALAR_N: tl.constexpr,
    ND: tl.constexpr,
    SHAPE: tl.constexpr,
    XS: tl.constexpr,
    NS: tl.constexpr,
    OS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    rem = offs.to(tl.int64)
    xo = tl.zeros((BLOCK,), dtype=tl.int64)
    no = tl.zeros((BLOCK,), dtype=tl.int64)
    oo = tl.zeros((BLOCK,), dtype=tl.int64)
    for d in tl.static_range(ND):
        sz = SHAPE[d]
        idx = rem % sz
        rem = rem // sz
        xo += idx * XS[d]
        no += idx * NS[d]
        oo += idx * OS[d]

    x = tl.load(x_ptr + xo, mask=mask)
    w0 = x * 0.0 + 1.0
    w1 = 2.0 * x + 1.0

    if HAS_N:
        if SCALAR_N:
            nv = tl.load(n_ptr).to(tl.int32)
            for i in tl.range(2, nv + 1):
                tmp = w1
                w1 = 2.0 * x * w1 - w0
                w0 = tmp
            res = tl.where(nv == 0, w0, w1)
            res = tl.where(nv < 0, res * 0.0, res)
        else:
            n_i = tl.load(n_ptr + no, mask=mask, other=0).to(tl.int32)
            nmax = tl.max(n_i, axis=0)
            for i in tl.range(2, nmax + 1):
                w2 = 2.0 * x * w1 - w0
                act = i <= n_i
                w0 = tl.where(act, w1, w0)
                w1 = tl.where(act, w2, w1)
            res = tl.where(n_i == 0, w0, w1)
            res = tl.where(n_i < 0, res * 0.0, res)
    else:
        if DEG < 0:
            res = w0 * 0.0
        elif DEG == 0:
            res = w0
        elif DEG == 1:
            res = w1
        else:
            for _ in tl.static_range(2, DEG + 1):
                tmp = w1
                w1 = 2.0 * x * w1 - w0
                w0 = tmp
            res = w1

    tl.store(out_ptr + oo, res, mask=mask)


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def special_chebyshev_polynomial_w(x, n, out=None):
    logger.debug("GEMS_MTHREADS SPECIAL_CHEBYSHEV_POLYNOMIAL_W")

    if not _use_triton_kernel(x):
        return default_special_chebyshev_polynomial_w(x, n, out=out)

    with torch_device_fn.device(x.device):
        return _launch_chebyshev_w(x, n, out)


def _launch_chebyshev_w(x, n, out=None):
    dtype = x.dtype
    if dtype is not torch.float32 and dtype is not torch.float64:
        raise _UnsupportedDtype(
            "special_chebyshev_polynomial_w only supports float32/float64, "
            f"got {dtype}"
        )

    if isinstance(n, torch.Tensor):
        n_tensor = n
        scalar_deg = 0
        has_n = True
        if n.shape != x.shape:
            if n.numel() == 1:
                scalar_n = True
            else:
                x, n_tensor = torch.broadcast_tensors(x, n)
                scalar_n = False
        else:
            scalar_n = False
    else:
        n_tensor = x
        scalar_deg = int(n)
        has_n = False
        scalar_n = False

    if out is None:
        out = torch.empty(x.shape, dtype=dtype, device=x.device)
    numel = out.numel()
    if numel == 0:
        return out

    grid = (triton.cdiv(numel, _BLOCK),)

    if (
        x.is_contiguous()
        and out.is_contiguous()
        and (not has_n or scalar_n or n_tensor.is_contiguous())
    ):
        _cheb_w_flat_kernel[grid](
            x,
            n_tensor,
            out,
            numel,
            DEG=scalar_deg,
            HAS_N=has_n,
            SCALAR_N=scalar_n,
            BLOCK=_BLOCK,
            num_warps=_WARPS,
        )
        return out

    ndim = x.ndim if x.ndim > 0 else 1
    shape = tuple(x.shape) if x.ndim > 0 else (1,)
    zero = tuple(0 for _ in range(ndim))
    xs = x.stride() if x.ndim > 0 else zero
    os_ = out.stride() if out.ndim > 0 else zero
    if has_n and not scalar_n:
        ns = n_tensor.stride()
    else:
        ns = zero

    _cheb_w_strided_kernel[grid](
        x,
        n_tensor,
        out,
        numel,
        DEG=scalar_deg,
        HAS_N=has_n,
        SCALAR_N=scalar_n,
        ND=ndim,
        SHAPE=shape,
        XS=xs,
        NS=ns,
        OS=os_,
        BLOCK=_BLOCK,
        num_warps=_WARPS,
    )
    return out
