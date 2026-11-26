# SPDX-License-Identifier: Apache-2.0
"""Static, K-contiguous BF16 projections for the Iluvatar backend."""

import torch
import triton
import triton.language as tl

GEOMETRIES = ((2048, 2560), (2048, 2048), (2048, 12288), (6144, 2048), (2048, 130560))
# Configuration order is rows, columns, reduction, warps, stages, partitions.
TILES = {
    (128, 2048, 12288): (64, 256, 32, 8, 1, 1, False),
    (128, 2048, 130560): (64, 256, 64, 16, 1, 1, False),
    (128, 2048, 2048): (64, 128, 64, 8, 2, 1, False),
    (128, 2048, 2560): (64, 64, 128, 16, 2, 1, False),
    (128, 6144, 2048): (64, 128, 64, 8, 2, 1, False),
    (16, 2048, 12288): (16, 128, 128, 8, 1, 1, False),
    (16, 2048, 130560): (16, 64, 128, 4, 2, 1, False),
    (16, 2048, 2048): (16, 64, 128, 4, 2, 1, False),
    (16, 2048, 2560): (16, 64, 128, 4, 2, 1, False),
    (16, 6144, 2048): (16, 32, 128, 4, 2, 1, False),
    (32, 2048, 12288): (32, 256, 64, 8, 1, 1, False),
    (32, 2048, 130560): (32, 256, 64, 8, 1, 1, False),
    (32, 2048, 2048): (32, 64, 128, 4, 2, 1, False),
    (32, 2048, 2560): (32, 128, 64, 4, 2, 1, False),
    (32, 6144, 2048): (32, 64, 128, 8, 2, 1, False),
    (48, 2048, 12288): (64, 256, 64, 16, 1, 1, False),
    (48, 2048, 130560): (64, 256, 64, 16, 1, 1, False),
    (48, 2048, 2048): (64, 64, 128, 16, 2, 1, False),
    (48, 2048, 2560): (32, 128, 64, 4, 2, 1, False),
    (48, 6144, 2048): (64, 64, 128, 16, 2, 1, False),
    (64, 2048, 12288): (64, 256, 64, 16, 1, 1, False),
    (64, 2048, 130560): (64, 256, 64, 16, 1, 1, False),
    (64, 2048, 2048): (64, 64, 128, 16, 2, 1, False),
    (64, 2048, 2560): (64, 64, 128, 16, 2, 1, False),
    (64, 6144, 2048): (64, 64, 128, 16, 2, 1, False),
}


@triton.jit(do_not_specialize=["M"])
def _projection(
    A,
    W,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    TINY: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    tile = tl.program_id(0)
    part = tl.program_id(1)
    ntiles = N // BN
    row_tile = tile // ntiles * BM
    rows = tl.arange(0, BM)
    if TINY:
        output_rows = row_tile + rows
        input_rows = tl.minimum(output_rows, M - 1)
    else:
        output_rows = tl.minimum(row_tile, M - BM) + rows
        input_rows = output_rows
    cols = tile % ntiles * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    # Splitting the K tiles also covers uneven partitions, e.g. 2048 / 3.
    if SPLITS == 1:
        start = 0
        stop = K // BK
    else:
        start = K // BK * part // SPLITS
        stop = K // BK * (part + 1) // SPLITS
    pa = A + input_rows[:, None] * K + start * BK + inner[None, :]
    if TRANSPOSED:
        pb = W + cols[:, None] * K + start * BK + inner[None, :]
    else:
        pb = W + start * BK + inner[:, None] + cols[None, :] * K
    for _ in range(stop - start):
        av = tl.load(pa)
        wt = tl.load(pb)
        if TRANSPOSED:
            acc = tl.dot(av, tl.trans(wt), acc)
        else:
            acc = tl.dot(av, wt, acc)
        pa += BK
        pb += BK
    address = (part * M + output_rows[:, None]) * N + cols[None, :]
    tl.store(
        C + address,
        acc,
        (output_rows < M)[:, None] & (output_rows >= row_tile)[:, None],
    )


@triton.jit(do_not_specialize=["M"])
def _sum_parts(P, C, M, N: tl.constexpr, PARTS: tl.constexpr, WIDTH: tl.constexpr):
    ix = tl.program_id(0) * WIDTH + tl.arange(0, WIDTH)
    valid = ix < M * N
    acc = tl.zeros((WIDTH,), tl.float32)
    for part in tl.static_range(PARTS):
        acc += tl.load(P + part * M * N + ix, valid, 0.0)
    tl.store(C + ix, acc, valid)


def bucket(m):
    if m <= 64:
        return triton.cdiv(m, 16) * 16
    return 128 if m <= 128 else 2048


def eligible(a, b, c):
    m, k = a.shape
    n = b.shape[1]
    return (
        1 <= m <= 2048
        and (k, n) in GEOMETRIES
        and a.dtype == b.dtype == c.dtype == torch.bfloat16
        and a.stride() == (k, 1)
        and b.stride() == (1, k)
        and c.stride() == (n, 1)
        and (a.data_ptr() | b.data_ptr() | c.data_ptr()) % 256 == 0
    )


def initial_tile(m, k, n):
    group = bucket(m)
    if group == 2048:
        return 128, 128, 32, 8, 2, 1, False
    if group == 128:
        return 64, 128, 64, 8, 2, 1, False
    if group == 16:
        return 16, 32 if k == 6144 else 64, 128, 4, 2, 1, False
    bm = 16 if group == 32 else 32
    return bm, 128, 64, 4, 2, 1, False


_prepared_devices = set()


def prepare_projection_tiles(device):
    """Load the static small-row binaries before decode graphs are captured."""
    if device.index in _prepared_devices:
        return
    with torch.cuda.device(device):
        # CoreX determines the matrix-engine mask from the real 2D tensor
        # layouts on the first cache miss, including during a compile-only warmup.
        bf16 = torch.empty((32, 32), device=device, dtype=torch.bfloat16)
        fp32 = torch.empty((32, 32), device=device, dtype=torch.float32)
        for k, n in GEOMETRIES:
            for upper in (16, 32, 48, 64, 128):
                tile = TILES.get((upper, k, n)) or initial_tile(upper, k, n)
                bm, bn, bk, nw, ns, parts, transposed = tile
                dst = bf16 if parts == 1 else fp32
                binary = _projection.warmup(
                    bf16,
                    bf16.t(),
                    dst,
                    upper,
                    n,
                    k,
                    bm,
                    bn,
                    bk,
                    parts,
                    upper <= bm,
                    transposed,
                    grid=(triton.cdiv(upper, bm) * (n // bn), parts),
                    num_warps=nw,
                    num_stages=ns,
                )
                _ = binary.run
                if parts > 1:
                    reduce = _sum_parts.warmup(
                        fp32,
                        bf16,
                        upper,
                        n,
                        parts,
                        1024,
                        grid=(triton.cdiv(upper * n, 1024),),
                        num_warps=4,
                    )
                    _ = reduce.run
    _prepared_devices.add(device.index)


def project(a, b, c, tile=None):
    m, k = a.shape
    n = b.shape[1]
    if tile is None:
        from .mm import _SMALL_BF16_TILES

        known = _SMALL_BF16_TILES.get((m, k, n))
        if known is not None:
            return project_large(a, b, c, known)
        if m >= 129:
            return project_large(a, b, c)
        if n == 130560 and m in (16, 32, 64):
            return _static_vocab(a, b, c)
    if tile is None:
        tile = TILES.get((bucket(m), k, n)) or initial_tile(m, k, n)
    bm, bn, bk, warps, stages, splits, transposed = tile
    # One call owns each partial buffer. During graph capture its addresses
    # belong to that graph's private pool.
    partial = (
        c
        if splits == 1
        else torch.empty((splits, m, n), dtype=torch.float32, device=c.device)
    )
    _projection[(triton.cdiv(m, bm) * (n // bn), splits)](
        a,
        b,
        partial,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        splits,
        bucket(m) <= bm,
        transposed,
        num_warps=warps,
        num_stages=stages,
    )
    if splits > 1:
        _sum_parts[(triton.cdiv(m * n, 1024),)](
            partial, c, m, n, splits, 1024, num_warps=4
        )
    return c


def project_large(a, b, c, known=None):
    # The sealed large-row kernel already slides its last full tile safely.
    # Reuse its measured group ordering for the five model projections.
    from .mm import _mm_large_m_kernel

    m, k = a.shape
    n = b.shape[1]
    bm, bn, bk, warps, stages = known or (
        (256, 256, 32, 16, 2) if m >= 256 else (128, 128, 64, 8, 2)
    )
    _mm_large_m_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        a,
        b,
        c,
        m,
        n,
        k,
        k,
        1,
        1,
        k,
        n,
        1,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=8,
        EVEN_M=m % bm == 0,
        EVEN_N=True,
        EVEN_K=True,
        num_warps=warps,
        num_stages=stages,
    )
    return c


def _static_vocab(a, b, c):
    # These shapes fill every row of the original kernel's input tile.
    # Use its JIT body directly with measured constants and no online tuner.
    from .mm import _single_pass_mm

    m, k = a.shape
    n = b.shape[1]
    bn, bk, nw = (128, 128, 8) if m == 16 else (256, 64, 8 if m == 32 else 16)
    _single_pass_mm[(n // bn, 1)](
        a,
        b,
        c,
        m,
        n,
        k,
        k,
        1,
        1,
        k,
        n,
        1,
        acc_dtype=tl.float32,
        input_precision=None,
        fp8_fast_accum=True,
        GROUP_M=8,
        SPLIT_K=1,
        EVEN_K=True,
        BLOCK_M=m,
        BLOCK_N=bn,
        BLOCK_K=bk,
        AB_DTYPE=tl.bfloat16,
        UPGRADE=False,
        UPGRADE_A_OFFS=False,
        UPGRADE_B_OFFS=False,
        UPGRADE_C_OFFS=False,
        num_warps=nw,
        num_stages=1,
    )
    return c
