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

# ---------------------------------------------------------------------------
# Persistent, pipelined RMSNorm.
#
# Grid = NPROG programs; each program owns `iters` row-tiles whose tile id is
#   t = pid + i * NPROG
# so consecutive programs touch consecutive tiles (contiguous DRAM) and the
# loop body lets the Ascend backend software-pipeline (multibuffer) the DMA
# loads against the vector math.  gamma is loaded once per program and reused
# by every iteration, so its traffic is negligible.
# All tiles are mask-free: N is either a power of two, or is decomposed into
# power-of-two chunks (6144 = 4096 + 2048, 16384 = 8192 + 8192).
# ---------------------------------------------------------------------------


@triton.jit
def _k_pow2(
    X,
    G,
    Y,
    RST,
    iters,
    eps,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NPROG: tl.constexpr,
    HOIST_G: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, N)
    if HOIST_G:
        gf = tl.load(G + cols).to(tl.float32)
    for i in tl.range(0, iters):
        t = pid + i * NPROG
        rm = t * BLOCK_M + tl.arange(0, BLOCK_M)
        offs = rm[:, None] * N + cols[None, :]
        x = tl.load(X + offs).to(tl.float32)
        var = tl.sum(x * x, axis=1) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        if HOIST_G:
            yv = x * rstd[:, None] * gf[None, :]
        else:
            g = tl.load(G + cols).to(tl.float32)
            yv = x * rstd[:, None] * g[None, :]
        tl.store(Y + offs, yv.to(Y.dtype.element_ty))
        tl.store(RST + rm, rstd)


@triton.jit
def _k_6144(
    X,
    G,
    Y,
    RST,
    iters,
    eps,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NPROG: tl.constexpr,
):
    # N == 6144 == 4096 + 2048, both power-of-two contiguous chunks.
    pid = tl.program_id(0)
    c0 = tl.arange(0, 4096)
    c1 = tl.arange(0, 2048)
    g0 = tl.load(G + c0).to(tl.float32)
    g1 = tl.load(G + (c1 + 4096)).to(tl.float32)
    for i in tl.range(0, iters):
        t = pid + i * NPROG
        rm = t * BLOCK_M + tl.arange(0, BLOCK_M)
        base = rm[:, None] * N
        o0 = base + c0[None, :]
        o1 = base + (c1[None, :] + 4096)
        x0 = tl.load(X + o0).to(tl.float32)
        x1 = tl.load(X + o1).to(tl.float32)
        var = (tl.sum(x0 * x0, axis=1) + tl.sum(x1 * x1, axis=1)) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        r = rstd[:, None]
        tl.store(Y + o0, (x0 * r * g0[None, :]).to(Y.dtype.element_ty))
        tl.store(Y + o1, (x1 * r * g1[None, :]).to(Y.dtype.element_ty))
        tl.store(RST + rm, rstd)


@triton.jit
def _k_16384(X, G, Y, RST, iters, eps, N: tl.constexpr, NPROG: tl.constexpr):
    # Wide-row path: N == 16384 == 8192 + 8192.  gamma (64 KB as fp32) is
    # loaded once per program and kept resident, so gamma traffic is
    # NPROG*64 KB instead of rows*64 KB (the previous form loaded it inside the
    # row loop, i.e. 64 MB of gamma against 67 MB of x on [1024,16384] fp32).
    # That is why this path uses few programs, each covering many rows, unlike
    # the narrow-row kernels whose gamma tile is only 1-24 KB.
    # num_stages=1: a multibuffered variant of this tile needs 384 KB of UB
    # against the 248 KB available.
    pid = tl.program_id(0)
    c = tl.arange(0, 8192)
    g0 = tl.load(G + c).to(tl.float32)
    g1 = tl.load(G + (c + 8192)).to(tl.float32)
    for i in tl.range(0, iters):
        t = pid + i * NPROG
        base = t * N
        o0 = base + c
        o1 = base + (c + 8192)
        if X.dtype.element_ty == tl.float32:
            x0 = tl.load(X + o0)
            x1 = tl.load(X + o1)
        else:
            x0 = tl.load(X + o0).to(tl.float32)
            x1 = tl.load(X + o1).to(tl.float32)
        var = (tl.sum(x0 * x0) + tl.sum(x1 * x1)) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Y + o0, (x0 * rstd * g0).to(Y.dtype.element_ty))
        tl.store(Y + o1, (x1 * rstd * g1).to(Y.dtype.element_ty))
        tl.store(RST + t, rstd)


@triton.jit
def _k_generic(X, G, Y, RST, M, eps, N: tl.constexpr, BLOCK_N: tl.constexpr):
    # Fallback: one row per program, column-masked power-of-two tile.
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    cmask = cols < N
    g = tl.load(G + cols, mask=cmask, other=0.0).to(tl.float32)
    offs = pid * N + cols
    x = tl.load(X + offs, mask=cmask, other=0.0).to(tl.float32)
    var = tl.sum(x * x) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    yv = x * rstd * g
    tl.store(Y + offs, yv.to(Y.dtype.element_ty), mask=cmask)
    tl.store(RST + pid, rstd)


def _is_pow2(v):
    return v > 0 and (v & (v - 1)) == 0


def _plan(m, n):
    """Return (kind, BLOCK_M, NPROG, iters) or None for the generic fallback."""
    if n == 6144:
        kind = 2
        bm = 2 if (m % 2 == 0) else 1
    elif n == 16384:
        kind = 3
        bm = 1
    elif _is_pow2(n):
        kind = 0
        bm = max(1, 8192 // n)
        bm = min(bm, m)
    else:
        return None
    if bm < 1 or m % bm:
        return None
    tiles = m // bm
    if kind == 3:
        # Wide-row kernel: gamma is a resident 64 KB fp32 tile per program, so
        # program count is a direct gamma-traffic multiplier.  Few programs,
        # many rows each (>= 64 programs still keeps the device's 64 concurrent
        # vector blocks busy).
        nprog = max(64, min(tiles // 8, 1024))
    elif tiles >= 1024:
        # Measured: the device runs up to 64 concurrent vector blocks, so a
        # 32-program launch uses only half the machine.  For high tile counts
        # spreading the tiles over up to 1024 programs lifted bf16 [65536,512]
        # from 0.494x to 0.868x and fp32 [1024,16384] from 0.488x to 0.817x.
        nprog = min(tiles, 1024)
    elif tiles >= 128:
        # Same full-machine occupancy, but keep >= 2 tiles per program: with
        # one tile per program the per-program prologue/descriptor/gamma load
        # is no longer amortized (measured: scalar pipe becomes the largest
        # consumer, 0.436 of block time, on [1024,1024]; gamma re-read costs
        # ~50% of the payload bytes at one tile per program for N>=4096).
        nprog = max(64, tiles // 2)
    else:
        nprog = min(tiles, 32)
    while nprog > 1 and tiles % nprog:
        nprog //= 2
    return kind, bm, nprog, tiles // nprog


def _launch(input, gamma, epsilon=1e-06):
    x = input
    n = int(x.shape[-1])
    m = int(x.numel() // n)
    x2 = x.view(m, n)
    y = torch.empty_like(x)
    y2 = y.view(m, n)
    rstd = torch.empty(x.shape[:-1] + (1,), dtype=torch.float32, device=x.device)
    rf = rstd.view(m)
    eps = float(epsilon)

    plan = _plan(m, n)
    if plan is None:
        bn = 1 << max(0, n - 1).bit_length()
        _k_generic[(m,)](x2, gamma, y2, rf, m, eps, N=n, BLOCK_N=bn)
        return y, rstd

    kind, bm, nprog, iters = plan
    grid = (nprog,)
    if kind == 0:
        _k_pow2[grid](
            x2,
            gamma,
            y2,
            rf,
            iters,
            eps,
            N=n,
            BLOCK_M=bm,
            NPROG=nprog,
            HOIST_G=True,
            num_stages=2,
        )
    elif kind == 2:
        _k_6144[grid](
            x2, gamma, y2, rf, iters, eps, N=n, BLOCK_M=bm, NPROG=nprog, num_stages=2
        )
    else:
        _k_16384[grid](x2, gamma, y2, rf, iters, eps, N=n, NPROG=nprog, num_stages=1)
    return y, rstd


def npu_rms_norm(input, gamma, epsilon=1e-6):
    """RMS normalization returning both the output and the FP32 inverse RMS."""
    if input.ndim < 1 or gamma.shape != (input.shape[-1],):
        raise ValueError("gamma must match the normalized last dimension")
    if input.dtype not in (torch.bfloat16, torch.float32) or gamma.dtype != input.dtype:
        raise TypeError("expected matching BF16 or FP32 operands")
    if not 1 <= input.shape[-1] <= 16384 or not input.numel():
        raise ValueError("expected nonempty rows with width up to 16384")
    if not input.is_contiguous() or not gamma.is_contiguous():
        raise ValueError("expected contiguous operands")
    if input.device.type != "npu" or gamma.device != input.device:
        raise ValueError("expected inputs on the same NPU")
    with torch_device_fn.device(input.device):
        return _launch(input, gamma, epsilon)
