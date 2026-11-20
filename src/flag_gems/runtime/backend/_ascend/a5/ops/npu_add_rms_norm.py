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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _fused_full(
    x1_ptr,
    x2_ptr,
    g_ptr,
    y_ptr,
    rstd_ptr,
    res_ptr,
    eps,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    R: tl.constexpr,
    ITERS: tl.constexpr,
    EXACT: tl.constexpr,
    NS: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    row0 = pid * (R * ITERS) + tl.arange(0, R)

    if EXACT:
        g = tl.load(g_ptr + cols).to(tl.float32)
        for it in tl.range(0, ITERS, num_stages=NS):
            rows = row0 + it * R
            off = rows[:, None] * N + cols[None, :]
            a = tl.load(x1_ptr + off)
            b = tl.load(x2_ptr + off)
            r = a.to(tl.float32) + b.to(tl.float32)
            tl.store(res_ptr + off, r)
            sumsq = tl.sum(r * r, axis=1)
            rstd = 1.0 / tl.sqrt(sumsq * (1.0 / N) + eps)
            tl.store(rstd_ptr + rows, rstd)
            y = r * rstd[:, None] * g[None, :]
            tl.store(y_ptr + off, y)
    else:
        cmask = cols < N
        g = tl.load(g_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        for it in range(ITERS):
            rows = row0 + it * R
            off = rows[:, None] * N + cols[None, :]
            mask = (rows < M)[:, None] & cmask[None, :]
            a = tl.load(x1_ptr + off, mask=mask, other=0.0)
            b = tl.load(x2_ptr + off, mask=mask, other=0.0)
            r = a.to(tl.float32) + b.to(tl.float32)
            tl.store(res_ptr + off, r, mask=mask)
            sumsq = tl.sum(r * r, axis=1)
            rstd = 1.0 / tl.sqrt(sumsq * (1.0 / N) + eps)
            tl.store(rstd_ptr + rows, rstd, mask=rows < M)
            y = r * rstd[:, None] * g[None, :]
            tl.store(y_ptr + off, y, mask=mask)


@triton.jit
def _fused_chunked(
    x1_ptr,
    x2_ptr,
    g_ptr,
    y_ptr,
    rstd_ptr,
    res_ptr,
    eps,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NC: tl.constexpr,
    R: tl.constexpr,
    ITERS: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    row0 = pid * (R * ITERS) + tl.arange(0, R)

    if EXACT:
        for it in range(ITERS):
            rows = row0 + it * R
            base = rows[:, None] * N
            acc = tl.zeros((R,), dtype=tl.float32)
            for k in range(NC):
                off = base + (k * BLOCK_N + cols)[None, :]
                a = tl.load(x1_ptr + off)
                b = tl.load(x2_ptr + off)
                r = a.to(tl.float32) + b.to(tl.float32)
                tl.store(res_ptr + off, r)
                acc += tl.sum(r * r, axis=1)

            rstd = 1.0 / tl.sqrt(acc * (1.0 / N) + eps)
            tl.store(rstd_ptr + rows, rstd)

            for k in range(NC):
                off = base + (k * BLOCK_N + cols)[None, :]
                a = tl.load(x1_ptr + off)
                b = tl.load(x2_ptr + off)
                r = a.to(tl.float32) + b.to(tl.float32)
                g = tl.load(g_ptr + k * BLOCK_N + cols).to(tl.float32)
                y = r * rstd[:, None] * g[None, :]
                tl.store(y_ptr + off, y)
    else:
        for it in range(ITERS):
            rows = row0 + it * R
            rmask = rows < M
            base = rows[:, None] * N
            acc = tl.zeros((R,), dtype=tl.float32)
            for k in range(NC):
                koff = k * BLOCK_N + cols
                mask = rmask[:, None] & (koff < N)[None, :]
                off = base + koff[None, :]
                a = tl.load(x1_ptr + off, mask=mask, other=0.0)
                b = tl.load(x2_ptr + off, mask=mask, other=0.0)
                r = a.to(tl.float32) + b.to(tl.float32)
                tl.store(res_ptr + off, r, mask=mask)
                acc += tl.sum(r * r, axis=1)

            rstd = 1.0 / tl.sqrt(acc * (1.0 / N) + eps)
            tl.store(rstd_ptr + rows, rstd, mask=rmask)

            for k in range(NC):
                koff = k * BLOCK_N + cols
                mask = rmask[:, None] & (koff < N)[None, :]
                off = base + koff[None, :]
                a = tl.load(x1_ptr + off, mask=mask, other=0.0)
                b = tl.load(x2_ptr + off, mask=mask, other=0.0)
                r = a.to(tl.float32) + b.to(tl.float32)
                g = tl.load(g_ptr + koff, mask=koff < N, other=0.0).to(tl.float32)
                y = r * rstd[:, None] * g[None, :]
                tl.store(y_ptr + off, y, mask=mask)


@triton.jit
def _fused_two(
    x1_ptr,
    x2_ptr,
    g_ptr,
    y_ptr,
    rstd_ptr,
    res_ptr,
    eps,
    N: tl.constexpr,
    A: tl.constexpr,
    B: tl.constexpr,
    R: tl.constexpr,
):
    """Single mask-free pass over R rows of exactly A + B elements each.

    Both column blocks stay live across the reduction, so each element is
    loaded once and r = x1 + x2 is computed once; there are no padded lanes and
    no column mask, and gamma is issued before the input transfers.
    """
    pid = tl.program_id(0)
    oa = tl.arange(0, A)
    ob = A + tl.arange(0, B)
    rows = pid * R + tl.arange(0, R)
    p0 = rows[:, None] * N + oa[None, :]
    p1 = rows[:, None] * N + ob[None, :]

    g0 = tl.load(g_ptr + oa).to(tl.float32)
    g1 = tl.load(g_ptr + ob).to(tl.float32)
    r0 = tl.load(x1_ptr + p0).to(tl.float32) + tl.load(x2_ptr + p0).to(tl.float32)
    r1 = tl.load(x1_ptr + p1).to(tl.float32) + tl.load(x2_ptr + p1).to(tl.float32)

    tl.store(res_ptr + p0, r0)
    tl.store(res_ptr + p1, r1)

    acc = tl.sum(r0 * r0, axis=1) + tl.sum(r1 * r1, axis=1)
    rstd = tl.rsqrt(acc * (1.0 / N) + eps)
    tl.store(rstd_ptr + rows, rstd)

    sr = rstd[:, None]
    tl.store(y_ptr + p0, r0 * sr * g0[None, :])
    tl.store(y_ptr + p1, r1 * sr * g1[None, :])


@triton.jit
def _fused_two_loop(
    x1_ptr,
    x2_ptr,
    g_ptr,
    y_ptr,
    rstd_ptr,
    res_ptr,
    eps,
    N: tl.constexpr,
    A: tl.constexpr,
    B: tl.constexpr,
    R: tl.constexpr,
    ITERS: tl.constexpr,
    GRID: tl.constexpr,
):
    """Single mask-free pass over A + B columns, R rows per iteration.

    Rows are block-cyclic so every program's transfers are spread over the
    whole matrix. Two blocks instead of three keeps four descriptors per
    iteration (32 KB + 16 KB per tensor at R=2) instead of six.
    """
    pid = tl.program_id(0)
    oa = tl.arange(0, A)
    ob = A + tl.arange(0, B)
    row0 = pid * R + tl.arange(0, R)
    g0 = tl.load(g_ptr + oa).to(tl.float32)
    g1 = tl.load(g_ptr + ob).to(tl.float32)

    for it in tl.range(0, ITERS, num_stages=2):
        rows = row0 + it * (R * GRID)
        p0 = rows[:, None] * N + oa[None, :]
        p1 = rows[:, None] * N + ob[None, :]
        r0 = tl.load(x1_ptr + p0).to(tl.float32) + tl.load(x2_ptr + p0).to(tl.float32)
        r1 = tl.load(x1_ptr + p1).to(tl.float32) + tl.load(x2_ptr + p1).to(tl.float32)
        tl.store(res_ptr + p0, r0)
        tl.store(res_ptr + p1, r1)
        acc = tl.sum(r0 * r0, axis=1) + tl.sum(r1 * r1, axis=1)
        rstd = tl.rsqrt(acc * (1.0 / N) + eps)
        tl.store(rstd_ptr + rows, rstd)
        sr = rstd[:, None]
        tl.store(y_ptr + p0, r0 * sr * g0[None, :])
        tl.store(y_ptr + p1, r1 * sr * g1[None, :])


@triton.jit
def _fused_three(
    x1_ptr,
    x2_ptr,
    g_ptr,
    y_ptr,
    rstd_ptr,
    res_ptr,
    eps,
    N: tl.constexpr,
    B: tl.constexpr,
    R: tl.constexpr,
    ITERS: tl.constexpr,
    GRID: tl.constexpr,
):
    """Single pass over a row of exactly 3*B elements.

    The three residual chunks stay live in registers across the reduction, so
    each element is loaded once and r = x1 + x2 is computed once.
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, B)
    row0 = pid * R + tl.arange(0, R)
    g0 = tl.load(g_ptr + cols).to(tl.float32)
    g1 = tl.load(g_ptr + B + cols).to(tl.float32)
    g2 = tl.load(g_ptr + 2 * B + cols).to(tl.float32)

    for it in tl.range(0, ITERS, num_stages=2):
        rows = row0 + it * (R * GRID)
        p0 = rows[:, None] * N + cols[None, :]
        p1 = p0 + B
        p2 = p1 + B
        r0 = tl.load(x1_ptr + p0).to(tl.float32) + tl.load(x2_ptr + p0).to(tl.float32)
        r1 = tl.load(x1_ptr + p1).to(tl.float32) + tl.load(x2_ptr + p1).to(tl.float32)
        r2 = tl.load(x1_ptr + p2).to(tl.float32) + tl.load(x2_ptr + p2).to(tl.float32)
        tl.store(res_ptr + p0, r0)
        tl.store(res_ptr + p1, r1)
        tl.store(res_ptr + p2, r2)
        acc = (
            tl.sum(r0 * r0, axis=1) + tl.sum(r1 * r1, axis=1) + tl.sum(r2 * r2, axis=1)
        )
        rstd = 1.0 / tl.sqrt(acc * (1.0 / N) + eps)
        tl.store(rstd_ptr + rows, rstd)
        sr = rstd[:, None]
        tl.store(y_ptr + p0, r0 * sr * g0[None, :])
        tl.store(y_ptr + p1, r1 * sr * g1[None, :])
        tl.store(y_ptr + p2, r2 * sr * g2[None, :])


_CORES = 64
_CHUNK = 2048
_LANES = 8192


def _launch(x1, x2, gamma, epsilon=1e-06):
    n = x1.shape[-1]
    m = x1.numel() // n

    y = torch.empty_like(x1)
    res = torch.empty_like(x1)
    rstd = torch.empty(x1.shape[:-1] + (1,), dtype=torch.float32, device=x1.device)
    if m == 0:
        return y, rstd, res

    eps = float(epsilon)
    pow2 = (n & (n - 1)) == 0

    if (
        not pow2
        and n % _CHUNK == 0
        and n // _CHUNK in (2, 3)
        and n <= _LANES
        and m < _CORES
    ):
        # tiny M with a row that is an exact sum of two power-of-two blocks:
        # no padded lanes and no column mask, and one row per program so every
        # row gets its own AIV block
        _fused_two[(m,)](x1, x2, gamma, y, rstd, res, eps, n, n - _CHUNK, _CHUNK, 1)
        return y, rstd, res

    if n <= _LANES and (m < _CORES or pow2):
        block_n = triton.next_power_of_2(n)
        r = max(1, _LANES // block_n) if m >= _CORES else 1
        grid = min(triton.cdiv(m, r), _CORES)
        iters = triton.cdiv(m, r * grid)
        exact = block_n == n and grid * r * iters == m
        # NS requests software pipelining of the in-program row loop; only the
        # many-iteration large-M dispatch asks for it, the single-shot
        # (R=1, ITERS=1) dispatch stays unpipelined.
        ns = 2 if (exact and iters > 1) else 1
        _fused_full[(grid,)](
            x1, x2, gamma, y, rstd, res, eps, m, n, block_n, r, iters, exact, ns
        )
    elif n % _CHUNK == 0:
        nc = n // _CHUNK
        r = max(1, _LANES // _CHUNK)
        grid = min(triton.cdiv(m, r), _CORES)
        iters = triton.cdiv(m, r * grid)
        if nc == 3 and grid * r * iters == m:
            # single-pass variant: no padded lanes, no re-read, r computed once
            r = 2
            grid = min(triton.cdiv(m, r), _CORES)
            iters = triton.cdiv(m, r * grid)
            if grid * r * iters == m:
                # two exact blocks (4096 + 2048) instead of three 2048 chunks:
                # four descriptors per iteration, 32 KB + 16 KB per tensor
                _fused_two_loop[(grid,)](
                    x1,
                    x2,
                    gamma,
                    y,
                    rstd,
                    res,
                    eps,
                    n,
                    n - _CHUNK,
                    _CHUNK,
                    r,
                    iters,
                    grid,
                )
                return y, rstd, res
        exact = grid * r * iters == m
        _fused_chunked[(grid,)](
            x1, x2, gamma, y, rstd, res, eps, m, n, _CHUNK, nc, r, iters, exact
        )
    elif n <= _LANES:
        block_n = triton.next_power_of_2(n)
        _fused_full[(m,)](
            x1, x2, gamma, y, rstd, res, eps, m, n, block_n, 1, 1, False, 1
        )
    else:
        nc = triton.cdiv(n, _CHUNK)
        _fused_chunked[(m,)](
            x1, x2, gamma, y, rstd, res, eps, m, n, _CHUNK, nc, 1, 1, False
        )
    return y, rstd, res


def npu_add_rms_norm(x1, x2, gamma, epsilon=1e-6):
    """Fused residual addition and RMS normalization returning the output, inverse RMS and residual."""
    if x1.ndim < 1 or x2.shape != x1.shape or gamma.shape != (x1.shape[-1],):
        raise ValueError("x1 and x2 must match and gamma must cover the last dimension")
    if not x1.numel() or not 1 <= x1.shape[-1] <= 8192:
        raise ValueError("expected nonempty rows with width up to 8192")
    if any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (x1, x2, gamma)):
        raise ValueError("expected contiguous BF16 operands")
    if x1.device.type != "npu" or x2.device != x1.device or gamma.device != x1.device:
        raise ValueError("expected inputs on the same NPU")
    with torch_device_fn.device(x1.device):
        return _launch(x1, x2, gamma, epsilon)
