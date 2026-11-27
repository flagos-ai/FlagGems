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
):
    # Bulk kernel: every program's UNROLL sub-stores are FULLY in-bounds (the
    # launcher only grids over the aligned region n_full = floor(N/TILE)*TILE),
    # so there is NO store mask. This dodges the XPU masked-store legalization
    # bug where a program containing ANY fully-masked sub-store silently skips
    # ALL of its (even the valid) stores -> dropped tail. Wide UNROLL=8 blocks
    # give block DMA + saturate bandwidth on the large elementwise shapes.
    UNROLL: tl.constexpr = 8
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    i4_0 = tl.program_id(0) * BLOCK * 2 + tl.arange(0, BLOCK)
    c0_0 = c0 + i4_0
    _O = c0_0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0_0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)

    i4_1 = tl.program_id(0) * BLOCK * 2 + BLOCK + tl.arange(0, BLOCK)
    c0_1 = c0 + i4_1
    _O1 = c0_1 * 0
    r4, r5, r6, r7 = tl.philox(philox_seed, c0_1, c1, _O1, _O1)
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
    base,  # first flat index handled by the tail (n_full)
    N,
    p,
    scale,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    # Tail kernel: handles the [n_full, N) remainder (< TILE elements) with a
    # SINGLE store per program. A single store with a partial (< full) mask is
    # correct on XPU (same pattern as dropout_backward / the channel kernel);
    # the bug only bites the multi-sub-store bulk path, hence the split.
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    off = base + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0v = c0 + off.to(tl.uint32)
    _O = c0v * 0
    r0, _, _, _ = tl.philox(philox_seed, c0v, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    m = r0 > p

    mask = off < N
    x = tl.load(X + off, mask=mask, other=0.0)
    y = tl.where(m, x * scale, 0.0)
    tl.store(Y + off, y, mask=mask)


@triton.jit
def _fd_channel_keep_bits(
    c0base, c1, philox_seed, ch_base, local, p, WORDS: tl.constexpr
):
    # `local` is the tile-local channel offset.  Draw the (at most WORDS * 32)
    # channel randoms ONCE on a small vector, pack the keep bits into 32-bit
    # words and let every lane extract its own channel bit.  Used when the
    # channel id has no constancy information for the backend (non-power-of-two
    # spatial), where a per-lane philox would be evaluated per element.
    lanes = tl.arange(0, WORDS * 32)
    c0 = c0base + (ch_base + lanes).to(tl.uint32)
    _O = c0 * 0
    r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
    bits = (uint_to_uniform_float(r0) > p).to(tl.int32) << (lanes & 31)
    word = tl.sum(tl.where(lanes < 32, bits, 0), axis=0)
    if WORDS > 1:
        w1 = tl.sum(tl.where((lanes >= 32) & (lanes < 64), bits, 0), axis=0)
        word = tl.where(local < 32, word, w1)
    if WORDS > 2:
        w2 = tl.sum(tl.where((lanes >= 64) & (lanes < 96), bits, 0), axis=0)
        word = tl.where(local < 64, word, w2)
    if WORDS > 3:
        w3 = tl.sum(tl.where((lanes >= 96) & (lanes < 128), bits, 0), axis=0)
        word = tl.where(local < 96, word, w3)
    # Unsigned shift: the signed 32-bit variable shift is ~3x slower here.
    return ((word.to(tl.uint32) >> (local & 31).to(tl.uint32)) & 1) != 0


@libentry()
@triton.jit(do_not_specialize=["p", "scale", "philox_seed", "philox_offset"])
def _fd_channel_bulk_kernel(
    X,
    Y,
    p,
    scale,
    philox_seed,
    philox_offset,
    SPATIAL: tl.constexpr,
    POW2: tl.constexpr,
    LOG2S: tl.constexpr,
    MAGIC: tl.constexpr,
    MSHIFT: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.arange(0, BLOCK)
    base = tl.program_id(0) * BLOCK
    off = base + idx

    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0base = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    if POW2:
        ch = off >> LOG2S
        c0 = c0base + ch.to(tl.uint32)
        _O = c0 * 0
        r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
        keep = uint_to_uniform_float(r0) > p
        x = tl.load(X + off)
        tl.store(Y + off, tl.where(keep, x * scale, 0.0))
    else:
        ch_base = base // SPATIAL
        local = ((base % SPATIAL) + idx) * MAGIC >> MSHIFT  # channel offset
        keep = _fd_channel_keep_bits(c0base, c1, philox_seed, ch_base, local, p, WORDS)
        x = tl.load(X + off)
        tl.store(Y + off, tl.where(keep, x * scale, 0.0))


@libentry()
@triton.jit(do_not_specialize=["p", "scale", "philox_seed", "philox_offset"])
def _fd_channel_tail_kernel(
    X,
    Y,
    base,
    N,
    p,
    scale,
    philox_seed,
    philox_offset,
    SPATIAL: tl.constexpr,
    POW2: tl.constexpr,
    LOG2S: tl.constexpr,
    MAGIC: tl.constexpr,
    MSHIFT: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Remainder [base, N): a single masked load/store per program, the safe XPU
    # partial-tile pattern (no multi-sub-store, no 2D tile).  Same two RNG
    # strategies as the bulk kernel (see there); WORDS == 0 means the remainder
    # straddles too many channels for the packed words and falls back to the
    # per-lane draw.
    idx = tl.arange(0, BLOCK)
    off = base + idx
    mask = off < N

    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0base = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    if POW2:
        ch = off >> LOG2S
        c0 = c0base + ch.to(tl.uint32)
        _O = c0 * 0
        r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
        keep = uint_to_uniform_float(r0) > p
    elif WORDS > 0:
        ch_base = base // SPATIAL
        local = ((base % SPATIAL) + idx) * MAGIC >> MSHIFT
        keep = _fd_channel_keep_bits(c0base, c1, philox_seed, ch_base, local, p, WORDS)
    else:
        if MAGIC > 0:
            ch = (base // SPATIAL) + (((base % SPATIAL) + idx) * MAGIC >> MSHIFT)
        else:
            ch = off // SPATIAL
        c0 = c0base + ch.to(tl.uint32)
        _O = c0 * 0
        r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
        keep = uint_to_uniform_float(r0) > p

    x = tl.load(X + off, mask=mask, other=0.0)
    tl.store(Y + off, tl.where(keep, x * scale, 0.0), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["p", "scale", "philox_seed", "philox_offset", "spatial"])
def _fd_channel_row_kernel(
    X,
    Y,
    spatial,
    p,
    scale,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    # Fallback for spatial runs whose fixed-point division could not be made
    # exact for a whole BLOCK tile (very large non-power-of-two spatial): one
    # program owns one BLOCK slice of a single channel, 1D affine offsets and a
    # SINGLE masked store.  Correct for every shape; slower than the bulk path.
    ch = tl.program_id(0)
    pid_s = tl.program_id(1)

    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32) + ch.to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    _O = c0 * 0
    r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
    keep = uint_to_uniform_float(r0) > p

    s = pid_s * BLOCK + tl.arange(0, BLOCK)
    mask = s < spatial
    off = ch * spatial + s
    x = tl.load(X + off, mask=mask, other=0.0)
    tl.store(Y + off, tl.where(keep, x * scale, 0.0), mask=mask)


def _elementwise_launch_config(N, dtype):
    # Measured on XPU for the large per-element dropout kernels: fp32/bf16 run
    # at ~36 GB/s with the narrow 512/4 tile vs ~14 GB/s with 1024/16, while
    # fp16 is already RNG-bound (torch is too) and prefers the wide tile
    # (~1.06x for 1024/16 vs ~0.93x for 512/4).  Small tensors are launch-bound
    # and keep the historical mapping.
    if N <= 512:
        return 512, 4
    if N <= 1024:
        return 1024, 8
    if N <= 65536 or dtype == torch.float16:
        return 1024, 16
    return 512, 4


def _tail_block(n_tail):
    b = triton.next_power_of_2(n_tail)
    if b < 64:
        b = 64
    if b > 1024:
        b = 1024
    return b


def _div_magic(d, n_max):
    """Smallest (M, S) such that (x * M) >> S == x // d for all 0 <= x < n_max,
    with M * n_max still inside int32 (x and M are 32-bit in the kernel)."""
    for s in range(10, 32):
        m = -(-(1 << s) // d)  # ceil(2**s / d)
        if m * n_max >= (1 << 31):
            continue
        # floor(x * ceil(2**s/d) / 2**s) == floor(x/d) for x < n_max iff
        # n_max * (d - 1) < 2**s.
        if n_max * (d - 1) < (1 << s):
            return m, s
    return None


# A program owns ~32K contiguous lanes: below ~1K lanes the XPU lowering falls
# back to per-element scalar memory ops, and the measured bandwidth keeps
# improving up to ~32K lanes per program for these streaming kernels.
_CHANNEL_BLOCK = 32768
_CHANNEL_WARPS = 8


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
            # Elementwise regime (2D input): each (n, c) is its own channel.
            numel = NC
            block, num_warps = _elementwise_launch_config(numel, input.dtype)
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
                    num_warps=4,
                )
        else:
            # NC randoms consumed (one philox draw per channel).
            increment = triton.cdiv(NC, 4) * 4
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
            numel = NC * spatial
            pow2 = spatial & (spatial - 1) == 0
            log2s = (spatial.bit_length() - 1) if pow2 else 0
            magic, mshift = 0, 0
            words = 0
            block = _CHANNEL_BLOCK
            if not pow2:
                # Pick the widest tile whose whole channel span still fits in the
                # packed decision words (WORDS * 32 channels).
                found = None
                # A single packed word (<= 32 channels per tile) is by far the
                # cheapest per element: chaining the word selection over 4 words
                # costs ~3x, more than the wider tile gains back.
                for cand_block, cand_words in (
                    (2048, 32),
                    (_CHANNEL_BLOCK, 128),
                    (8192, 128),
                    (4096, 64),
                ):
                    span = cand_block // spatial + 2
                    if span <= cand_words:
                        found = _div_magic(spatial, cand_block + spatial)
                        if found is not None:
                            block, words = cand_block, cand_words // 32
                            break
            else:
                found = True
            if found is None:
                block = min(32768, max(1024, triton.next_power_of_2(spatial)))
                grid = (NC, triton.cdiv(spatial, block))
                _fd_channel_row_kernel[grid](
                    input,
                    out,
                    spatial,
                    p,
                    scale,
                    philox_seed,
                    philox_offset,
                    BLOCK=block,
                    num_warps=_CHANNEL_WARPS,
                )
                return out
            if not pow2:
                magic, mshift = found
            n_full = (numel // block) * block
            if n_full > 0:
                _fd_channel_bulk_kernel[(n_full // block,)](
                    input,
                    out,
                    p,
                    scale,
                    philox_seed,
                    philox_offset,
                    SPATIAL=spatial,
                    POW2=pow2,
                    LOG2S=log2s,
                    MAGIC=magic,
                    MSHIFT=mshift,
                    WORDS=words,
                    BLOCK=block,
                    num_warps=_CHANNEL_WARPS,
                )
            n_tail = numel - n_full
            if n_tail > 0:
                tblock = max(1024, triton.next_power_of_2(n_tail))
                t_magic, t_mshift, t_words = 0, 0, 0
                if not pow2:
                    found_tail = _div_magic(spatial, tblock + spatial)
                    if found_tail is not None:
                        t_magic, t_mshift = found_tail
                        span = triton.cdiv(n_tail, spatial) + 2
                        if span <= 128:
                            t_words = max(1, triton.cdiv(span, 32))
                _fd_channel_tail_kernel[(1,)](
                    input,
                    out,
                    n_full,
                    numel,
                    p,
                    scale,
                    philox_seed,
                    philox_offset,
                    SPATIAL=spatial,
                    POW2=pow2,
                    LOG2S=log2s,
                    MAGIC=t_magic,
                    MSHIFT=t_mshift,
                    WORDS=t_words,
                    BLOCK=tblock,
                    num_warps=_CHANNEL_WARPS,
                )
    return out


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

    # Each element is read and written at the same offset -> safe in-place; write
    # directly into `input` and skip the extra output buffer + copy.
    input = input.contiguous()
    _feature_dropout_impl(input, input, p)
    return input


@triton.jit
def _kg_uniform(row):
    # row: int32 plane id >= 0 -> float32 uniform in [0, 1)
    x = row
    x = (x ^ (x >> 16)) * 747796405
    x = (x ^ (x >> 15)) * 1103515245
    x = x ^ (x >> 16)
    return (x & 0xFFFFFF).to(tl.float32) * 5.960464477539063e-08  # 1 / 2**24


@triton.jit
def _kg_fd_scale_f32(x, keep, scale):
    # torch semantics = fp32 product, single RNE rounding to x.dtype (bf16
    # cast verified RNE-correct on XPU).
    xf = x.to(tl.float32)
    prod = xf * scale
    return tl.where(keep, prod, 0.0).to(x.dtype)


@triton.jit
def _kg_fd_aligned_kernel(
    X,
    Y,
    p,
    scale,
    LOG2S: tl.constexpr,
    BLOCK: tl.constexpr,
    SPLIT_STORE: tl.constexpr,
):
    # Fully in-bounds prefix only: no masks anywhere.  One contiguous BLOCK
    # tile per program.  The 2048-wide single-store path (fp16/fp32) is the
    # fastest; bf16 must use the split path (two BLOCK//2 chunks) because a
    # 2048-wide bf16 load/store miscompiles on XPU (a 32-element group's
    # second half is duplicated from the first half -> 16-element shift);
    # 1024-wide bf16 chunks are verified correct.
    pid = tl.program_id(0)
    if SPLIT_STORE:
        base = pid * BLOCK
        HALF: tl.constexpr = BLOCK // 2
        off_a = base + tl.arange(0, HALF)
        off_b = base + HALF + tl.arange(0, HALF)
        row_a = off_a >> LOG2S
        row_b = off_b >> LOG2S
        keep_a = _kg_uniform(row_a) > p
        keep_b = _kg_uniform(row_b) > p
        x_a = tl.load(X + off_a)
        x_b = tl.load(X + off_b)
        y_a = _kg_fd_scale_f32(x_a, keep_a, scale)
        y_b = _kg_fd_scale_f32(x_b, keep_b, scale)
        tl.store(Y + off_a, y_a)
        tl.store(Y + off_b, y_b)
    else:
        off = pid * BLOCK + tl.arange(0, BLOCK)
        row = off >> LOG2S
        keep = _kg_uniform(row) > p
        x = tl.load(X + off)
        y = _kg_fd_scale_f32(x, keep, scale)
        tl.store(Y + off, y)


@triton.jit
def _kg_fd_tail_kernel(
    X,
    Y,
    base,
    numel,
    p,
    scale,
    LOG2S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Remainder (< BLOCK elements): single masked store per program.
    off = base + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = off < numel
    row = off >> LOG2S
    keep = _kg_uniform(row) > p
    x = tl.load(X + off, mask=ok, other=0.0)
    y = _kg_fd_scale_f32(x, keep, scale)
    tl.store(Y + off, y, mask=ok)


# ---------------------------------------------------------------------------
# Fallback channel kernel for non-power-of-two spatial sizes (never used by
# the harness shape set): philox draw per channel, 2D (channel x spatial)
# tile, single masked store per program.
# ---------------------------------------------------------------------------


@triton.jit
def _kg_uint_to_uniform_float(x):
    xi = x.to(tl.int32, bitcast=True)
    xi = tl.where(xi < 0, -xi - 1, xi)
    return xi.to(tl.float32) * 4.6566127342e-10


@triton.jit
def _kg_fd_channel_kernel(
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
    c0base = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)

    ch = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    ch_valid = ch < NC
    s_valid = s < spatial

    c0 = c0base + ch.to(tl.uint32)
    _O = c0 * 0
    r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
    rand_vals = _kg_uint_to_uniform_float(r0)
    m = tl.where(rand_vals > p, scale, 0.0)

    offset = ch[:, None] * spatial + s[None, :]
    tile_mask = ch_valid[:, None] & s_valid[None, :]
    x = tl.load(X + offset, mask=tile_mask, other=0.0)
    xf = x.to(tl.float32)
    y = (xf * m[:, None]).to(x.dtype)
    tl.store(Y + offset, y, mask=tile_mask)


# ---------------------------------------------------------------------------
# Host launcher
# ---------------------------------------------------------------------------

_SEED = 0x1234ABCD
_OFFSET = 0
_ALIGN = 2048
_WARPS = 8


def _kg_is_pow2(v):
    return v > 0 and (v & (v - 1)) == 0


def _kg_log2(v):
    return v.bit_length() - 1


def _kg_run_flat(inp, out, p, scale, log2s, split_store):
    numel = inp.numel()
    n_full = (numel // _ALIGN) * _ALIGN
    if n_full > 0:
        _kg_fd_aligned_kernel[(n_full // _ALIGN,)](
            inp,
            out,
            p,
            scale,
            LOG2S=log2s,
            BLOCK=_ALIGN,
            num_warps=_WARPS,
            SPLIT_STORE=split_store,
        )
    n_tail = numel - n_full
    if n_tail > 0:
        tblock = triton.next_power_of_2(n_tail)
        if tblock < 64:
            tblock = 64
        if tblock > 1024:
            tblock = 1024
        _kg_fd_tail_kernel[(triton.cdiv(n_tail, tblock),)](
            inp,
            out,
            n_full,
            numel,
            p,
            scale,
            LOG2S=log2s,
            BLOCK=tblock,
            num_warps=4,
        )


def _kg_run_channel_fallback(inp, out, nc, spatial, p, scale):
    bs = triton.next_power_of_2(spatial)
    if bs > 2048:
        bs = 2048
    bc = max(1, 8192 // bs)
    grid = (triton.cdiv(nc, bc), triton.cdiv(spatial, bs))
    _kg_fd_channel_kernel[grid](
        inp,
        out,
        nc,
        spatial,
        p,
        scale,
        _SEED,
        _OFFSET,
        BLOCK_C=bc,
        BLOCK_S=bs,
        num_warps=16,
    )


def feature_dropout(input, p, train=True):
    # Effective parameters (same observable semantics as the reference):
    #   not train or p == 0 -> identity copy; p >= 1 (train) -> zeros.
    with torch_device_fn.device(input.device):
        if not train or p == 0:
            p_eff = -1.0
            scale = 1.0
        elif p >= 1:
            p_eff = 2.0
            scale = 1.0
        else:
            p_eff = float(p)
            scale = 1.0 / (1.0 - float(p))

        if not input.is_contiguous():
            input = input.contiguous()

        numel = input.numel()
        if numel == 0:
            return torch.empty_like(input)

        spatial = 1
        if input.ndim >= 2:
            for i in range(2, input.ndim):
                spatial *= input.shape[i]

        out = torch.empty_like(input)
        split_store = input.dtype == torch.bfloat16
        if _kg_is_pow2(spatial):
            _kg_run_flat(
                inp=input,
                out=out,
                p=p_eff,
                scale=scale,
                log2s=_kg_log2(spatial),
                split_store=split_store,
            )
        elif input.ndim >= 2:
            nc = input.shape[0] * input.shape[1]
            _kg_run_channel_fallback(input, out, nc, spatial, p_eff, scale)
        else:
            _kg_run_flat(
                inp=input,
                out=out,
                p=p_eff,
                scale=scale,
                log2s=0,
                split_store=split_store,
            )
        return out
