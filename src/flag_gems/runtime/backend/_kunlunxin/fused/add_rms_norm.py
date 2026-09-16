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
import builtins
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# 64 cores * 128 = 8192: on this XPU `tl.sum` is only complete for blocks
# <= 8192 without `buffer_size_limit` (HARNESS_SUMMARY 2.5), so 8192 doubles as
# the SRAM sweet spot and the reduction correctness ceiling.
MAX_BLOCK = 8192

# Launch-bound corner (same thresholds as rms_norm / fused_add_rms_norm):
# per-row launch costs ~0.6-0.9us, so tiny-N + huge-M needs a 2D row tile.
MULTIROW_N = 256
MULTIROW_M = 4096
TILE_BUDGET = 8192  # rows * cols per 2D tile; keeps the tile in SRAM
FLAT_BLOCK = 4096  # N == 1: elements per flat program


@libentry()
@triton.jit
def add_rms_norm_kernel(
    Y,  # output
    X1,  # input 1 (contiguous [M, N])
    X2,  # input 2 (contiguous [M, N])
    W,  # weight (contiguous [N])
    N: tl.constexpr,  # number of columns (normalized dim)
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,  # whether N is not a multiple of BLOCK_SIZE
):
    # Per-row kernel: one program per row. N is constexpr and the columns span
    # exactly [0, N) with no power-of-2 padding, so the row is one stride-1
    # contiguous block -> XPU OffsetAnalysis emits block DMA (runtime N or a
    # padded TILE_N would force discrete access, ~2x slower).
    pid = ext.program_id(0)
    Y += pid * N
    X1 += pid * N
    X2 += pid * N

    cols = tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        # NOTE (kunlunxin/XPU): when N is a multiple of BLOCK_SIZE every
        # `cols < N` mask is trivially all-true, but the masked tl.load/tl.store
        # still forces the slow XPU masked-memory path (up to ~2.4x slower on
        # fp16/bf16, byte-identical output). Take the unmasked fast path
        # whenever it is provably safe.
        mask = cols < N
        x1 = tl.load(X1 + cols, mask, other=0.0).to(tl.float32)
        x2 = tl.load(X2 + cols, mask, other=0.0).to(tl.float32)
        x = x1 + x2
        var = tl.sum(x * x, axis=0) / N
        rrms = 1 / tl.sqrt(var + eps)
        w = tl.load(W + cols, mask=mask, other=0.0)
        y = (x * rrms).to(Y.dtype.element_ty) * w
        tl.store(Y + cols, y, mask=mask)
    else:
        x1 = tl.load(X1 + cols).to(tl.float32)
        x2 = tl.load(X2 + cols).to(tl.float32)
        x = x1 + x2
        var = tl.sum(x * x, axis=0) / N
        rrms = 1 / tl.sqrt(var + eps)
        w = tl.load(W + cols)
        y = (x * rrms).to(Y.dtype.element_ty) * w
        tl.store(Y + cols, y)


@libentry()
@triton.jit
def add_rms_norm_tile_kernel(
    Y,  # output
    X1,  # input 1 (contiguous [M, N])
    X2,  # input 2 (contiguous [M, N])
    W,  # weight (contiguous [N])
    N: tl.constexpr,  # number of columns (normalized dim)
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,  # whether N is not a multiple of BLOCK_SIZE
):
    # Per-row looped kernel for N > 8192: the reduction is accumulated in a
    # fixed BLOCK_SIZE (<= 8192, tl.sum-safe) fp32 buffer in a first pass, then
    # the output is produced in a second pass. The tile kernel is used for
    # large normalized dims exactly like rms_norm_kerne_tile.
    pid = ext.program_id(0)
    Y += pid * N
    X1 += pid * N
    X2 += pid * N

    _var_base = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        if NEED_MASK:
            mask = cols < N
            x1 = tl.load(X1 + cols, mask, other=0.0).to(tl.float32)
            x2 = tl.load(X2 + cols, mask, other=0.0).to(tl.float32)
        else:
            x1 = tl.load(X1 + cols).to(tl.float32)
            x2 = tl.load(X2 + cols).to(tl.float32)
        x = x1 + x2
        _var_base += x * x / N
    var = tl.sum(_var_base)
    rrms = 1 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        if NEED_MASK:
            mask = cols < N
            x1 = tl.load(X1 + cols, mask, other=0.0).to(tl.float32)
            x2 = tl.load(X2 + cols, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask, other=0.0)
            y = ((x1 + x2) * rrms).to(Y.dtype.element_ty) * w
            tl.store(Y + cols, y, mask=mask)
        else:
            x1 = tl.load(X1 + cols).to(tl.float32)
            x2 = tl.load(X2 + cols).to(tl.float32)
            w = tl.load(W + cols)
            y = ((x1 + x2) * rrms).to(Y.dtype.element_ty) * w
            tl.store(Y + cols, y)


@libentry()
@triton.jit
def add_rms_norm_tile2d_kernel(
    Y,  # output
    X1,  # input 1 (contiguous [M, N])
    X2,  # input 2 (contiguous [M, N])
    W,  # weight (contiguous [N])
    eps: tl.constexpr,
    TILE_M: tl.constexpr,  # rows per program (M % TILE_M == 0 guaranteed)
    N: tl.constexpr,  # number of columns (normalized dim), used as tile width
):
    pid = ext.program_id(0)

    n_off = tl.arange(0, N)
    w = tl.load(W + n_off).to(tl.float32)

    m_off = pid * TILE_M + tl.arange(0, TILE_M)
    offs = m_off[:, None] * N + n_off[None, :]

    x1 = tl.load(X1 + offs).to(tl.float32)
    x2 = tl.load(X2 + offs).to(tl.float32)
    x = x1 + x2

    var = tl.sum(x * x, axis=1) / N
    rrms = 1.0 / tl.sqrt(var + eps)

    y = (x * rrms[:, None]).to(Y.dtype.element_ty) * w[None, :]
    tl.store(Y + offs, y.to(Y.dtype.element_ty))


@libentry()
@triton.jit
def add_rms_norm_multirow_kernel(
    Y,  # output
    X1,  # input 1 (contiguous [M, N])
    X2,  # input 2 (contiguous [M, N])
    W,  # weight (contiguous [N])
    M,  # number of rows (runtime; masked)
    eps: tl.constexpr,
    TILE_M: tl.constexpr,
    N: tl.constexpr,  # number of columns (normalized dim), used as tile width
):
    pid = ext.program_id(0)

    n_off = tl.arange(0, N)
    w = tl.load(W + n_off).to(tl.float32)

    m_off = pid * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_off < M
    offs = m_off[:, None] * N + n_off[None, :]

    x1 = tl.load(X1 + offs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(X2 + offs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    x = x1 + x2

    var = tl.sum(x * x, axis=1) / N
    rrms = 1.0 / tl.sqrt(var + eps)

    y = (x * rrms[:, None]).to(Y.dtype.element_ty) * w[None, :]
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=m_mask[:, None])


@libentry()
@triton.jit
def add_rms_norm_flat_kernel(
    Y,  # output
    X1,  # input 1 (contiguous [M, N]), N == 1
    X2,  # input 2 (contiguous [M, N]), N == 1
    W,  # weight (single element)
    xnumel,  # number of elements (== M * N == M)
    eps,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < xnumel
    x1 = tl.load(X1 + offs, mask, other=0.0).to(tl.float32)
    x2 = tl.load(X2 + offs, mask, other=0.0).to(tl.float32)
    x = x1 + x2
    rrms = 1.0 / tl.sqrt(x * x + eps)
    w = tl.load(W)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + offs, y, mask=mask)


def _pick_tile_m(M, N):
    """TILE_M for the unmasked 2D tile kernel, or None if not applicable.

    The tile kernel is strictly unmasked (any mask collapses block-DMA on XPU),
    so it is only valid when M % TILE_M == 0 AND N fits the tile SRAM budget.
    Same sweep as the XPU-validated rms_norm tile (rms_norm_perf_fix: TILE_M
    power-of-2 is measurably faster than an odd value).
    """
    if N <= 256:
        for cand in (32, 16):
            if M % cand == 0:
                return cand
        return None
    tm = 16
    while tm * N > 65536:  # keep the [TILE_M, N] fp32 tile within SRAM
        tm //= 2
    while tm >= 2:
        if M % tm == 0:
            return tm
        tm //= 2
    return None


def add_rms_norm(x1, x2, normalized_shape, weight, eps=1e-5):
    """
    Add two inputs element-wise and apply RMS normalization, on the
    Kunlunxin/XPU backend.

    Args:
        x1: First input tensor
        x2: Second input tensor (shape must match x1)
        normalized_shape: Shape to normalize over (typically the last dimension)
        weight: Optional weight tensor for the normalization
        eps: Epsilon value for numerical stability

    Returns:
        Normalized output tensor
    """
    logger.debug(
        "GEMS_KUNLUNXIN add_rms_norm, [input1 shape]: %s, [input2 shape]: %s, "
        "[weight shape]: %s",
        x1.size(),
        x2.size(),
        weight.size() if weight is not None else None,
    )
    dim = x1.ndim - len(normalized_shape)
    M = math.prod(x1.shape[:dim])
    N = math.prod(normalized_shape)

    assert x1.shape == x2.shape, f"Input shapes must match: {x1.shape} vs {x2.shape}"

    x1 = x1.contiguous()
    x2 = x2.contiguous()
    weight = weight.contiguous()
    y = torch.empty_strided(x1.size(), x1.stride(), dtype=x1.dtype, device=x1.device)

    with torch_device_fn.device(x1.device):
        if N > MAX_BLOCK:
            need_mask = (N % MAX_BLOCK) != 0
            add_rms_norm_tile_kernel[M,](
                y, x1, x2, weight, N, eps, MAX_BLOCK, need_mask
            )
        elif N == 1:
            grid = (triton.cdiv(M, FLAT_BLOCK),)
            add_rms_norm_flat_kernel[grid](y, x1, x2, weight, M, eps, FLAT_BLOCK)
        else:
            TILE_M = _pick_tile_m(M, N)
            if TILE_M is not None:
                # Unmasked 2D tile: strictly faster than any masked per-row /
                # masked multirow variant (see _pick_tile_m / rms_norm body).
                grid = (M // TILE_M,)
                add_rms_norm_tile2d_kernel[grid](y, x1, x2, weight, eps, TILE_M, N)
            elif N <= MULTIROW_N and M >= MULTIROW_M:
                # Small N + many rows with M not divisible by any TILE_M
                # candidate: batched masked multi-row fallback.
                TILE_M = builtins.max(1, TILE_BUDGET // N)
                grid = (triton.cdiv(M, TILE_M),)
                add_rms_norm_multirow_kernel[grid](y, x1, x2, weight, M, eps, TILE_M, N)
            else:
                BLOCK_SIZE = builtins.min(MAX_BLOCK, triton.next_power_of_2(N))
                need_mask = (N % BLOCK_SIZE) != 0
                add_rms_norm_kernel[M,](
                    y, x1, x2, weight, N, eps, BLOCK_SIZE, need_mask
                )

    return y
