import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)


UNROLL = 8

ROUNDS = 4


@libentry()
@triton.jit(do_not_specialize=["p", "scale", "philox_seed", "philox_offset"])
def _fd_elementwise_bulk_kernel(
    X,
    Y,
    p,
    scale,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    ROUNDS: tl.constexpr,
):
    UNROLL: tl.constexpr = 8
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    i4_0 = tl.program_id(0) * BLOCK * 2 + tl.arange(0, BLOCK)
    c0_0 = c0 + i4_0
    _O = c0_0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0_0, c1, _O, _O, n_rounds=ROUNDS)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)

    i4_1 = tl.program_id(0) * BLOCK * 2 + BLOCK + tl.arange(0, BLOCK)
    c0_1 = c0 + i4_1
    _O1 = c0_1 * 0
    r4, r5, r6, r7 = tl.philox(philox_seed, c0_1, c1, _O1, _O1, n_rounds=ROUNDS)
    r4 = uint_to_uniform_float(r4)
    r5 = uint_to_uniform_float(r5)
    r6 = uint_to_uniform_float(r6)
    r7 = uint_to_uniform_float(r7)

    m0 = r0 > p
    m1 = r1 > p
    m2 = r2 > p
    m3 = r3 > p
    m4 = r4 > p
    m5 = r5 > p
    m6 = r6 > p
    m7 = r7 > p

    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    off_4 = off_3 + BLOCK
    off_5 = off_4 + BLOCK
    off_6 = off_5 + BLOCK
    off_7 = off_6 + BLOCK

    x0 = tl.load(X + off_0)
    x1 = tl.load(X + off_1)
    x2 = tl.load(X + off_2)
    x3 = tl.load(X + off_3)
    x4 = tl.load(X + off_4)
    x5 = tl.load(X + off_5)
    x6 = tl.load(X + off_6)
    x7 = tl.load(X + off_7)

    y0 = tl.where(m0, x0 * scale, 0.0)
    y1 = tl.where(m1, x1 * scale, 0.0)
    y2 = tl.where(m2, x2 * scale, 0.0)
    y3 = tl.where(m3, x3 * scale, 0.0)
    y4 = tl.where(m4, x4 * scale, 0.0)
    y5 = tl.where(m5, x5 * scale, 0.0)
    y6 = tl.where(m6, x6 * scale, 0.0)
    y7 = tl.where(m7, x7 * scale, 0.0)

    tl.store(Y + off_0, y0)
    tl.store(Y + off_1, y1)
    tl.store(Y + off_2, y2)
    tl.store(Y + off_3, y3)
    tl.store(Y + off_4, y4)
    tl.store(Y + off_5, y5)
    tl.store(Y + off_6, y6)
    tl.store(Y + off_7, y7)


@libentry()
@triton.jit(do_not_specialize=["p", "scale", "philox_seed", "philox_offset"])
def _fd_elementwise_tail_kernel(
    X,
    Y,
    base,
    N,
    p,
    scale,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    ROUNDS: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    off = base + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0v = c0 + off.to(tl.uint32)
    _O = c0v * 0
    r0, _, _, _ = tl.philox(philox_seed, c0v, c1, _O, _O, n_rounds=ROUNDS)
    r0 = uint_to_uniform_float(r0)
    m = r0 > p

    mask = off < N
    x = tl.load(X + off, mask=mask, other=0.0)
    y = tl.where(m, x * scale, 0.0)
    tl.store(Y + off, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["p", "scale", "philox_seed", "philox_offset"])
def _fd_channel_kernel(
    X,
    Y,
    NC,
    spatial,
    p,
    scale,
    philox_seed,
    philox_offset,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)

    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)

    ch = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    ch_valid = ch < NC
    s_valid = s < spatial

    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    c0 = c0 + ch.to(tl.uint32)
    _O = c0 * 0
    r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
    rand_vals = uint_to_uniform_float(r0)
    m = tl.where(rand_vals > p, scale, 0.0)

    offset = ch[:, None] * spatial + s[None, :]
    tile_mask = ch_valid[:, None] & s_valid[None, :]
    x = tl.load(X + offset, mask=tile_mask, other=0.0)
    y = x * m[:, None]
    tl.store(Y + offset, y, mask=tile_mask)


_CHANNEL_2D_NC_LIMIT = 4096


@libentry()
@triton.jit(do_not_specialize=["p", "scale", "philox_seed", "philox_offset"])
def _fd_channel_mask_kernel(
    MASK,
    NC,
    p,
    scale,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    ROUNDS: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cmask = off < NC
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    cv = c0 + off.to(tl.uint32)
    _O = cv * 0
    r0, _, _, _ = tl.philox(philox_seed, cv, c1, _O, _O, n_rounds=ROUNDS)
    r0 = uint_to_uniform_float(r0)
    m = tl.where(r0 > p, scale, 0.0)
    tl.store(MASK + off, m, mask=cmask)


@libentry()
@triton.jit
def _fd_channel_apply_kernel(
    X,
    Y,
    MASK,
    numel,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ch = off // S
    if NEED_MASK:
        mask = off < numel
        mv = tl.load(MASK + ch, mask=mask, other=0.0)
        x32 = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        y = x32 * mv
        tl.store(Y + off, y, mask=mask)
    else:
        mv = tl.load(MASK + ch)
        x32 = tl.load(X + off).to(tl.float32)
        y = x32 * mv
        tl.store(Y + off, y)


MASK_BLOCK = 4096
APPLY_BLOCK = 65536


def _elementwise_launch_config(N):
    if N <= 512:
        return 512, 4
    elif N <= 1024:
        return 1024, 8
    else:
        return 1024, 16


def _tail_block(n_tail):
    b = triton.next_power_of_2(n_tail)
    if b < 64:
        b = 64
    if b > 1024:
        b = 1024
    return b


def _channel_config(spatial):
    bs = triton.next_power_of_2(spatial)
    if bs > 2048:
        bs = 2048
    bc = max(1, 8192 // bs)
    return bc, bs, 16


def _feature_dropout_impl(input, out, p):
    device = input.device
    N = input.shape[0]
    C = input.shape[1]
    NC = N * C
    spatial = 1
    for i in range(2, input.ndim):
        spatial *= input.shape[i]
    scale = 1.0 / (1.0 - p)

    with torch_device_fn.device(device):
        if spatial == 1:
            numel = NC
            block, num_warps = _elementwise_launch_config(numel)
            tile = block * UNROLL
            n_full = (numel // tile) * tile
            increment = triton.cdiv(numel, 4) * 4
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
            if n_full > 0:
                grid = (n_full // tile,)
                _fd_elementwise_bulk_kernel[grid](
                    input,
                    out,
                    p,
                    scale,
                    philox_seed,
                    philox_offset,
                    BLOCK=block,
                    ROUNDS=ROUNDS,
                    num_warps=num_warps,
                )
            n_tail = numel - n_full
            if n_tail > 0:
                tblock = _tail_block(n_tail)
                tgrid = (triton.cdiv(n_tail, tblock),)
                _fd_elementwise_tail_kernel[tgrid](
                    input,
                    out,
                    n_full,
                    numel,
                    p,
                    scale,
                    philox_seed,
                    philox_offset,
                    BLOCK=tblock,
                    ROUNDS=ROUNDS,
                    num_warps=4,
                )
        elif NC <= _CHANNEL_2D_NC_LIMIT:
            block_c, block_s, num_warps = _channel_config(spatial)
            grid = (triton.cdiv(NC, block_c), triton.cdiv(spatial, block_s))
            increment = triton.cdiv(NC, 4) * 4
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
            _fd_channel_kernel[grid](
                input,
                out,
                NC,
                spatial,
                p,
                scale,
                philox_seed,
                philox_offset,
                BLOCK_C=block_c,
                BLOCK_S=block_s,
                num_warps=num_warps,
            )
        else:
            mask = torch.empty(NC, device=device, dtype=torch.float32)
            increment = triton.cdiv(NC, 4) * 4
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
            _fd_channel_mask_kernel[(triton.cdiv(NC, MASK_BLOCK),)](
                mask,
                NC,
                p,
                scale,
                philox_seed,
                philox_offset,
                BLOCK=MASK_BLOCK,
                ROUNDS=ROUNDS,
                num_warps=8,
            )
            numel = NC * spatial
            _fd_channel_apply_kernel[(triton.cdiv(numel, APPLY_BLOCK),)](
                input,
                out,
                mask,
                numel,
                spatial,
                BLOCK=APPLY_BLOCK,
                NEED_MASK=numel % APPLY_BLOCK != 0,
                num_warps=32,
            )
    return out


def feature_dropout(input, p, train=True):
    logger.debug("GEMS_KUNLUNXIN FEATURE_DROPOUT")

    if not train or p == 0:
        return input.clone()
    if p == 1:
        return torch.zeros_like(input)
    if input.ndim < 2:
        raise RuntimeError(
            "Feature dropout requires at least 2 dimensions in the input"
        )
    assert 0.0 < p < 1.0, "p must be in (0, 1)"

    input = input.contiguous()
    out = torch.empty_like(input)
    return _feature_dropout_impl(input, out, p)


def feature_dropout_(input, p, train=True):
    logger.debug("GEMS_KUNLUNXIN FEATURE_DROPOUT_")

    if not train or p == 0:
        return input
    if p == 1:
        input.zero_()
        return input
    if input.ndim < 2:
        raise RuntimeError(
            "Feature dropout requires at least 2 dimensions in the input"
        )
    assert 0.0 < p < 1.0, "p must be in (0, 1)"

    input = input.contiguous()
    _feature_dropout_impl(input, input, p)
    return input
