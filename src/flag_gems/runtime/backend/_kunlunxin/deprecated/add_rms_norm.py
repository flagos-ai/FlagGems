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

MAX_BLOCK = 8192

MULTIROW_N = 256
MULTIROW_M = 4096
TILE_BUDGET = 8192
FLAT_BLOCK = 4096


@libentry()
@triton.jit
def add_rms_norm_kernel(
    Y,
    X1,
    X2,
    W,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    Y += pid * N
    X1 += pid * N
    X2 += pid * N

    cols = tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
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
    Y,
    X1,
    X2,
    W,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
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
    Y,
    X1,
    X2,
    W,
    eps: tl.constexpr,
    TILE_M: tl.constexpr,
    N: tl.constexpr,
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
    Y,
    X1,
    X2,
    W,
    M,
    eps: tl.constexpr,
    TILE_M: tl.constexpr,
    N: tl.constexpr,
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
    Y,
    X1,
    X2,
    W,
    xnumel,
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
    while tm * N > 65536:
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
                grid = (M // TILE_M,)
                add_rms_norm_tile2d_kernel[grid](y, x1, x2, weight, eps, TILE_M, N)
            elif N <= MULTIROW_N and M >= MULTIROW_M:
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
