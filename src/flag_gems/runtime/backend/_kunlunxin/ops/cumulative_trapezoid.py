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

from flag_gems.ops.cumulative_trapezoid import (
    cumulative_trapezoid as _shared_cumulative_trapezoid,
)

logger = logging.getLogger(__name__)

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_INT32_MAX = (1 << 31) - 1
_U_CACHE = {}


def _ones_upper(width, dev):
    """Upper-triangular ones: cumsum(x) == x @ u."""
    key = (width, dev)
    u = _U_CACHE.get(key)
    if u is None:
        u = torch.triu(torch.ones(width, width, dtype=torch.float32, device=dev))
        _U_CACHE[key] = u
    return u


@triton.jit
def _cumtrapz_scan_kernel(
    y_ptr,
    out_ptr,
    u_ptr,
    N,
    L,
    NG,
    dx,
    G: tl.constexpr,
    BM: tl.constexpr,
):
    """Fused pair-mean + inclusive scan over BM rows at a time.

    t[k] = (y[k] + y[k+1]) * dx / 2 and out[k] = sum_{j<=k} t[j].  With
    u[j, k] = 1 for j <= k, the scan of one G-wide group is t @ u and the
    running group total is carried in registers, so every element is read once
    and written once.  The last lane of the last group (t[N-1], which is not
    part of the result) spills onto the next row's out[0]; _cumtrapz_first_kernel
    rewrites those afterwards.
    """
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    kk = tl.arange(0, G)
    jj = tl.arange(0, G)
    u = tl.load(u_ptr + jj[:, None] * G + kk[None, :])
    in_base = rm[:, None] * N
    out_base = rm[:, None] * L
    carry = tl.zeros([BM], dtype=tl.float32)
    scale = 0.5 * dx
    for g in range(NG - 1):
        cols = g * G + kk
        x = tl.load(y_ptr + in_base + cols[None, :]).to(tl.float32)
        xp = tl.load(y_ptr + in_base + (cols + 1)[None, :]).to(tl.float32)
        t = (x + xp) * scale
        yt = tl.dot(t, u, input_precision="ieee") + carry[:, None]
        tl.store(out_ptr + out_base + cols[None, :], yt.to(out_ptr.dtype.element_ty))
        carry += tl.sum(t, axis=1)
    cols = (NG - 1) * G + kk
    x = tl.load(y_ptr + in_base + cols[None, :]).to(tl.float32)
    xp = tl.load(
        y_ptr + in_base + (cols + 1)[None, :],
        mask=(cols + 1)[None, :] < N,
        other=0.0,
    ).to(tl.float32)
    t = (x + xp) * scale
    yt = tl.dot(t, u, input_precision="ieee") + carry[:, None]
    tl.store(out_ptr + out_base + cols[None, :], yt.to(out_ptr.dtype.element_ty))


@triton.jit
def _cumtrapz_first_kernel(
    y_ptr,
    out_ptr,
    N,
    L,
    dx,
    BLOCK: tl.constexpr,
):
    """out[r, 0] = (y[r, 0] + y[r, 1]) * dx / 2."""
    pid = tl.program_id(0)
    rm = pid * BLOCK + tl.arange(0, BLOCK)
    ya = tl.load(y_ptr + rm * N).to(tl.float32)
    yb = tl.load(y_ptr + rm * N + 1).to(tl.float32)
    tl.store(out_ptr + rm * L, ((ya + yb) * (0.5 * dx)).to(out_ptr.dtype.element_ty))


def _fast_plan(rows, n, dtype):
    """(G, BM) for the fused scan, or None to use the generic kernel.

    The kernel is unmasked, so G has to divide n and BM has to divide rows.
    """
    if dtype not in _FAST_DTYPES or rows < 16 or n < 64:
        return None
    if rows * n > _INT32_MAX:
        return None
    if n % 128 == 0:
        g = 128
    elif n % 64 == 0:
        g = 64
    else:
        return None
    bm = 128
    while bm > 16 and rows % bm:
        bm //= 2
    if rows % bm or rows < bm:
        return None
    return g, bm


def cumulative_trapezoid(y, *, dx=1, dim=-1):
    logger.debug("GEMS CUMULATIVE_TRAPEZOID")

    assert y.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"cumulative_trapezoid: unsupported dtype {y.dtype}"

    ndim = y.dim()
    d = dim if dim >= 0 else dim + ndim

    sizes = list(y.shape)
    N = sizes[d]

    outer = 1
    for s in sizes[:d]:
        outer *= s
    inner = 1
    for s in sizes[d + 1 :]:
        inner *= s

    out_sizes = list(sizes)
    out_sizes[d] = max(N - 1, 0)

    if N <= 1 or outer * inner == 0:
        output = torch.empty(out_sizes, dtype=y.dtype, device=y.device)
        return output if dx == 1 else output * dx

    plan = _fast_plan(outer, N, y.dtype) if inner == 1 and y.is_contiguous() else None
    if plan is not None:
        g, bm = plan
        L = N - 1
        buf = torch.empty(outer * L + 1 + 64, dtype=y.dtype, device=y.device)
        output = buf.narrow(0, 0, outer * L).view(out_sizes)
        _cumtrapz_scan_kernel[(outer // bm,)](
            y,
            output,
            _ones_upper(g, y.device),
            N,
            L,
            N // g,
            float(dx),
            G=g,
            BM=bm,
            num_warps=4,
            buffer_size_limit=2048,
        )
        _cumtrapz_first_kernel[(outer // 16,)](
            y,
            output,
            N,
            L,
            float(dx),
            BLOCK=16,
            num_warps=1,
        )
        return output

    return _shared_cumulative_trapezoid(y, dx=dx, dim=dim)
