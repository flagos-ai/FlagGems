import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


MULTIROW_MAX_N = 8192


def _prev_pow2(x):
    x = max(1, int(x))
    return 1 << (x.bit_length() - 1)


def _multirow_tile_m(N):
    return min(16, _prev_pow2(max(1, MULTIROW_MAX_N // N)))


FWD_MULTIROW_MAX_N = 4096
FWD_CHUNK_BN = 8192
FWD_TAIL_PIECE = 4096
FWD_N_TILE_M = [(16, 64), (64, 32), (256, 16), (1024, 16), (4096, 8)]


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def log_softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    OUTPUT_K: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    outer = pid_m // OUTPUT_K
    inner = pid_m % OUTPUT_K
    input_base = pid_m * N
    output_base = outer * N * OUTPUT_K + inner
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        input_offset = input_base + n_offsets
        output_offset = output_base + n_offsets * OUTPUT_K
        mask = n_offsets < N
        inp = tl.load(input_ptr + input_offset, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        log_z = tl.log(z)
        out = inp - m - log_z
        tl.store(output_ptr + output_offset, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += input_base
        output_ptr += output_base

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets).to(tl.float32)
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced
        log_z = tl.log(z)

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, TILE_N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(
                input_ptr + n_offsets,
                mask=mask,
                other=-float("inf"),
                eviction_policy="evict_first",
            ).to(tl.float32)
            o = inp - m - log_z
            tl.store(output_ptr + n_offsets * OUTPUT_K, o, mask=mask)
        for start_n in range(TILE_N, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first").to(
                tl.float32
            )
            o = inp - m - log_z
            tl.store(output_ptr + n_offsets * OUTPUT_K, o)


FWD_MULTIROW_MAX_N = 4096
FWD_CHUNK_BN = 8192
FWD_TAIL_PIECE = 4096
FWD_N_TILE_M = [(16, 64), (64, 32), (256, 16), (1024, 16), (4096, 8)]


@triton.jit
def _k_fwd_key_u32(bits):
    return bits ^ (0x80000000 | (bits >> 31))


@triton.jit
def _k_fwd_decode_key(m_key):
    return (m_key ^ (0x80000000 | ((m_key >> 31) ^ 1))).to(tl.float32, bitcast=True)


@libentry()
@triton.jit
def log_softmax_kernel_singlepass(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
    NEED_MASK: tl.constexpr,
    USE_KEY: tl.constexpr,
):
    pid_m = ext.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    if NEED_MASK:
        mask = m_offsets[:, None] < M
        inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
            tl.float32
        )
    else:
        inp = tl.load(input_ptr + offsets).to(tl.float32)
    if USE_KEY:
        bits = inp.to(tl.uint32, bitcast=True)
        m = _k_fwd_decode_key(tl.max(_k_fwd_key_u32(bits), 1))
    else:
        m = tl.max(inp, 1)
    e = tl.exp(inp - m[:, None])
    z = tl.sum(e, 1)
    out = inp - m[:, None] - tl.log(z)[:, None]
    if NEED_MASK:
        tl.store(output_ptr + offsets, out, mask=mask)
    else:
        tl.store(output_ptr + offsets, out)


@libentry()
@triton.jit
def log_softmax_kernel_singlepass_tail(
    output_ptr,
    input_ptr,
    M,
    ROW_START,
    N,
    TILE_N: tl.constexpr,
    USE_KEY: tl.constexpr,
):
    """Masked tail rows of the singlepass tile: one program per row, grid =
    M - ROW_START. 1D per-row masked load/store (exact on XPU), unlike the
    2D row-masked tile whose store is not honored."""
    pid = ext.program_id(0)
    n_offsets = tl.arange(0, TILE_N)
    off = (ROW_START + pid) * N + n_offsets
    mask = n_offsets < N
    x = tl.load(input_ptr + off, mask=mask, other=-float("inf")).to(tl.float32)
    if USE_KEY:
        m = _k_fwd_decode_key(tl.max(_k_fwd_key_u32(x.to(tl.uint32, bitcast=True)), 0))
    else:
        m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    out = x - m - tl.log(z)
    tl.store(output_ptr + off, out, mask=mask)


@libentry()
@triton.jit
def log_softmax_kernel_chunk(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
    USE_KEY: tl.constexpr,
):
    """Flat (row*C_FULL + c) grid; offsets = pid*BN (BN constexpr -> the
    [M*C_FULL, BN] read is contiguous, block DMA on XPU). Partial (m_c, z_c)
    stored at row*C + c."""
    pid = ext.program_id(0)
    row = pid // C_FULL
    c = pid % C_FULL
    n_offsets = tl.arange(0, BLOCK_N)
    off = pid * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    if USE_KEY:
        m = _k_fwd_decode_key(tl.max(_k_fwd_key_u32(x.to(tl.uint32, bitcast=True)), 0))
    else:
        m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    tl.store(partial_m_ptr + row * C + c, m)
    tl.store(partial_z_ptr + row * C + c, z)


@libentry()
@triton.jit
def log_softmax_chunk_combine(
    m_ptr,
    log_z_ptr,
    partial_m_ptr,
    partial_z_ptr,
    C,
    C2: tl.constexpr,
):
    """Combine row partials -> (row max, log-sum-exp); row stride = C."""
    pid = ext.program_id(0)
    c_offsets = tl.arange(0, C2)
    cmask = c_offsets < C
    po = pid * C + c_offsets
    mc = tl.load(partial_m_ptr + po, mask=cmask, other=-float("inf"))
    zc = tl.load(partial_z_ptr + po, mask=cmask, other=0.0)
    m = tl.max(mc, 0)
    z = tl.sum(zc * tl.exp(mc - m), 0)
    log_z = tl.log(z)
    tl.store(m_ptr + pid, m)
    tl.store(log_z_ptr + pid, log_z)


@libentry()
@triton.jit
def log_softmax_chunk_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    log_z_ptr,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    """Flat grid (M*C_FULL): second read (offset = pid*BN) -> out."""
    pid = ext.program_id(0)
    row = pid // C_FULL
    m = tl.load(m_ptr + row)
    log_z = tl.load(log_z_ptr + row)
    n_offsets = tl.arange(0, BLOCK_N)
    off = pid * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, x - m - log_z)


@libentry()
@triton.jit
def log_softmax_chunk_strided(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    N,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
    USE_KEY: tl.constexpr,
):
    """Flat (row*C_FULL + c) grid with per-row base offsets (needed when
    N % BN != 0: the flat pid*BN form drifts by the row tail)."""
    pid = ext.program_id(0)
    row = pid // C_FULL
    c = pid % C_FULL
    n_offsets = tl.arange(0, BLOCK_N)
    off = row * N + c * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    if USE_KEY:
        m = _k_fwd_decode_key(tl.max(_k_fwd_key_u32(x.to(tl.uint32, bitcast=True)), 0))
    else:
        m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    tl.store(partial_m_ptr + row * C + c, m)
    tl.store(partial_z_ptr + row * C + c, z)


@libentry()
@triton.jit
def log_softmax_chunk_pass_strided(
    output_ptr,
    input_ptr,
    m_ptr,
    log_z_ptr,
    N,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    row = pid // C_FULL
    m = tl.load(m_ptr + row)
    log_z = tl.load(log_z_ptr + row)
    n_offsets = tl.arange(0, BLOCK_N)
    c = pid % C_FULL
    off = row * N + c * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, x - m - log_z)


@libentry()
@triton.jit
def log_softmax_tail_piece_partial(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    M,
    N,
    C_STRIDE,
    T_SLOT,
    TAIL_BASE,
    PLEN: tl.constexpr,
    USE_KEY: tl.constexpr,
):
    """Partial (m, z) over one exact power-of-2 tail piece of width PLEN<=4096
    (fully inside the row, so loads/stores are UNMASKED). The old masked 1D
    tail tiles (pow2-padded lanes + mask) silently miscompile on XPU for a
    family of widths (probed 2026-08-20: 65/97/99/101/127/129/193/254/255/257/
    511/513/1023/1025 etc.); only unmasked lane sets that exactly match the
    piece are shape-exact."""
    pid = ext.program_id(0)
    n_offsets = TAIL_BASE + tl.arange(0, PLEN)
    off = pid * N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    if USE_KEY:
        m = _k_fwd_decode_key(tl.max(_k_fwd_key_u32(x.to(tl.uint32, bitcast=True)), 0))
    else:
        m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    po = pid * C_STRIDE + T_SLOT
    tl.store(partial_m_ptr + po, m)
    tl.store(partial_z_ptr + po, z)


@libentry()
@triton.jit
def log_softmax_tail_piece_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    log_z_ptr,
    N,
    TAIL_BASE,
    PLEN: tl.constexpr,
):
    """Pass over one exact pow2 tail piece: out = x - row - logsumexp
    (unmasked, piece fully inside the row)."""
    pid = ext.program_id(0)
    m = tl.load(m_ptr + pid)
    log_z = tl.load(log_z_ptr + pid)
    n_offsets = TAIL_BASE + tl.arange(0, PLEN)
    off = pid * N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, x - m - log_z)


@libentry()
@triton.jit
def log_softmax_tail_masked_partial(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    N,
    C_STRIDE,
    T_SLOT,
    TAIL_BASE,
    TAIL_LEN,
    USE_KEY: tl.constexpr,
):
    """Masked 64-lane piece for the <64 column remainder of a row tail.
    A 64-wide masked tile with <64 real lanes is the exact form the previous
    (08-19) implementation used for small tails and is what the official
    (200, 40999, 3) case exercised; wider masks for 1..63 lanes are fine,
    only >= 64-lane padded pieces miscompile."""
    pid = ext.program_id(0)
    n_offsets = tl.arange(0, 64)
    within = n_offsets < TAIL_LEN
    off = pid * N + TAIL_BASE + n_offsets
    x = tl.load(input_ptr + off, mask=within, other=float("-inf")).to(tl.float32)
    if USE_KEY:
        m = _k_fwd_decode_key(tl.max(_k_fwd_key_u32(x.to(tl.uint32, bitcast=True)), 0))
    else:
        m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    po = pid * C_STRIDE + T_SLOT
    tl.store(partial_m_ptr + po, m)
    tl.store(partial_z_ptr + po, z)


@libentry()
@triton.jit
def log_softmax_tail_masked_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    log_z_ptr,
    N,
    TAIL_BASE,
    TAIL_LEN,
):
    """Masked 64-lane tail write for the <64 remainder (see partial)."""
    pid = ext.program_id(0)
    m = tl.load(m_ptr + pid)
    log_z = tl.load(log_z_ptr + pid)
    n_offsets = tl.arange(0, 64)
    within = n_offsets < TAIL_LEN
    off = pid * N + TAIL_BASE + n_offsets
    x = tl.load(input_ptr + off, mask=within, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + off,
        x - m - log_z,
        mask=within,
    )


FWD_N1_BLOCK = 512


@libentry()
@triton.jit
def log_softmax_kernel_n1(
    output_ptr,
    input_ptr,
    n_elem,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + offs, x - x, mask=mask)


def _fwd_n1_flat(out, inp):
    n_elem = inp.numel()
    log_softmax_kernel_n1[(triton.cdiv(n_elem, FWD_N1_BLOCK), 1, 1)](
        out,
        inp,
        n_elem,
        BLOCK=FWD_N1_BLOCK,
        buffer_size_limit=2048,
    )


def _fwd_singlepass(out, inp, M, N):
    use_key = inp.dtype != torch.bfloat16
    if (N & (N - 1)) != 0 and N >= 64:
        grid = (M, 1, 1)
        log_softmax_kernel_inner[grid](
            out,
            inp,
            M,
            N,
            1,
            TILE_N=triton.next_power_of_2(N),
            ONE_TILE_PER_CTA=True,
            buffer_size_limit=2048,
            isCloseVectorization=True,
            is_use_mask_zero=True,
        )
        return
    tile_m = 4
    for n_hi, tm in FWD_N_TILE_M:
        if N <= n_hi:
            tile_m = tm
            break
    if (N & (N - 1)) and N < 64:
        tile_m = 64
    nfull, tail = divmod(M, tile_m)
    log_softmax_kernel_singlepass[(nfull, 1, 1)](
        out,
        inp,
        M,
        N,
        TILE_M=tile_m,
        NEED_MASK=False,
        USE_KEY=use_key,
        buffer_size_limit=2048,
        num_warps=8,
    )
    if tail:
        log_softmax_kernel_singlepass_tail[(tail, 1, 1)](
            out,
            inp,
            M,
            nfull * tile_m,
            N,
            TILE_N=triton.next_power_of_2(N),
            USE_KEY=use_key,
            buffer_size_limit=2048,
            num_warps=8,
        )


def _pow2_tail_pieces(n, cap=FWD_TAIL_PIECE):
    """Split a row tail into (pieces, remainder):
    - pieces: exact power-of-2 lane sets with width >= 64, fully inside the
      row -> unmasked loads/stores (shape-exact on XPU).
    - remainder r = n % 64: handled by the masked 64-lane kernels.
    Pieces narrower than 64 lanes are NEVER emitted: probes (2026-08-20)
    show an unmasked <64-wide lane group written by a wider vectorized
    store corrupts the first (64-w) columns of every row."""
    r = n % 64
    m = n - r
    pieces = []
    while m > 0:
        p = 1 << (m.bit_length() - 1)
        while p > cap:
            p >>= 1
        pieces.append(p)
        m -= p
    return pieces, r


def _fwd_chunk_split(out, inp, M, N):
    use_key = inp.dtype != torch.bfloat16
    c_full = N // FWD_CHUNK_BN
    taillen = N - c_full * FWD_CHUNK_BN
    pieces, rrem = _pow2_tail_pieces(taillen) if taillen else ([], 0)
    have_rem = rrem != 0
    C = c_full + len(pieces) + (1 if have_rem else 0)
    C2 = triton.next_power_of_2(C)
    pm = torch.empty((M * C,), dtype=torch.float32, device=inp.device)
    pz = torch.empty((M * C,), dtype=torch.float32, device=inp.device)
    m_out = torch.empty((M,), dtype=torch.float32, device=inp.device)
    lz = torch.empty((M,), dtype=torch.float32, device=inp.device)
    base = c_full * FWD_CHUNK_BN
    for slot, plen in enumerate(pieces):
        log_softmax_tail_piece_partial[(M, 1, 1)](
            pm,
            pz,
            inp,
            M,
            N,
            C,
            c_full + slot,
            base,
            PLEN=plen,
            USE_KEY=use_key,
            num_warps=8,
        )
        base += plen
    if have_rem:
        log_softmax_tail_masked_partial[(M, 1, 1)](
            pm,
            pz,
            inp,
            N,
            C,
            c_full + len(pieces),
            base,
            rrem,
            USE_KEY=use_key,
            num_warps=8,
        )
    if c_full:
        if pieces or have_rem:
            log_softmax_chunk_strided[(M * c_full, 1, 1)](
                pm,
                pz,
                inp,
                N,
                c_full,
                C,
                BLOCK_N=FWD_CHUNK_BN,
                USE_KEY=use_key,
                buffer_size_limit=2048,
                num_warps=8,
            )
        else:
            log_softmax_kernel_chunk[(M * c_full, 1, 1)](
                pm,
                pz,
                inp,
                c_full,
                C,
                BLOCK_N=FWD_CHUNK_BN,
                USE_KEY=use_key,
                buffer_size_limit=2048,
                num_warps=8,
            )
    log_softmax_chunk_combine[(M, 1, 1)](
        m_out,
        lz,
        pm,
        pz,
        C,
        C2=C2,
        buffer_size_limit=2048,
        num_warps=8,
    )
    if c_full:
        if pieces or have_rem:
            log_softmax_chunk_pass_strided[(M * c_full, 1, 1)](
                out,
                inp,
                m_out,
                lz,
                N,
                c_full,
                C,
                BLOCK_N=FWD_CHUNK_BN,
                buffer_size_limit=2048,
                num_warps=8,
            )
        else:
            log_softmax_chunk_pass[(M * c_full, 1, 1)](
                out,
                inp,
                m_out,
                lz,
                c_full,
                C,
                BLOCK_N=FWD_CHUNK_BN,
                buffer_size_limit=2048,
                num_warps=8,
            )
    base = c_full * FWD_CHUNK_BN
    for plen in pieces:
        log_softmax_tail_piece_pass[(M, 1, 1)](
            out,
            inp,
            m_out,
            lz,
            N,
            base,
            PLEN=plen,
            num_warps=8,
        )
        base += plen
    if have_rem:
        log_softmax_tail_masked_pass[(M, 1, 1)](
            out,
            inp,
            m_out,
            lz,
            N,
            base,
            rrem,
            num_warps=8,
        )


BWD_MULTIROW_MAX_N = 4096
BWD_SINGLE_TILE_MAX_N = 4096
BWD_MT_TILE_N = 8192
BWD_MT_TILE_N_WIDE = 16384


@libentry()
@triton.jit
def log_softmax_backward_kernel_perrow(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    NEED_MASK: tl.constexpr = True,
):
    pid_m = ext.program_id(0)
    if pid_m < M:
        out_ptr += pid_m * N
        out_grad_ptr += pid_m * N
        in_grad_ptr += pid_m * N
        n_offsets = tl.arange(0, TILE_N)
        if NEED_MASK:
            mask = n_offsets < N
            og = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            scale = tl.sum(og, 0)
            o = tl.load(out_ptr + n_offsets, mask=mask).to(tl.float32)
            ig = og - tl.exp(o) * scale
            tl.store(in_grad_ptr + n_offsets, ig, mask=mask)
        else:
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            scale = tl.sum(og, 0)
            o = tl.load(out_ptr + n_offsets).to(tl.float32)
            ig = og - tl.exp(o) * scale
            tl.store(in_grad_ptr + n_offsets, ig)


@libentry()
@triton.jit
def log_softmax_backward_kernel_perrow_mt(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    if pid_m < M:
        out_ptr += pid_m * N
        out_grad_ptr += pid_m * N
        in_grad_ptr += pid_m * N

        scale_acc = tl.zeros([TILE_N], dtype=tl.float32)
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            scale_acc += og
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            og = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            scale_acc += og
        scale = tl.sum(scale_acc, 0)

        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            o = tl.load(out_ptr + n_offsets).to(tl.float32)
            ig = og - tl.exp(o) * scale
            tl.store(in_grad_ptr + n_offsets, ig)
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            og = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            o = tl.load(out_ptr + n_offsets, mask=mask).to(tl.float32)
            ig = og - tl.exp(o) * scale
            tl.store(in_grad_ptr + n_offsets, ig, mask=mask)


@libentry()
@triton.jit
def log_softmax_backward_kernel_flat1(
    x_ptr,
    out_grad_ptr,
    in_grad_ptr,
    n_elem,
    BLOCK: tl.constexpr = 256,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    og = tl.load(out_grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    o = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ig = og - tl.exp(o) * og
    tl.store(in_grad_ptr + offs, ig, mask=mask)


@libentry()
@triton.jit
def log_softmax_backward_kernel_multirow(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
    NEED_MASK: tl.constexpr = True,
):
    pid_m = ext.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    if NEED_MASK:
        mask = m_offsets[:, None] < M
        og = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        o = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.sum(og, 1)
        ig = og - tl.exp(o) * scale[:, None]
        tl.store(in_grad_ptr + offsets, ig, mask=mask)
    else:
        og = tl.load(out_grad_ptr + offsets).to(tl.float32)
        o = tl.load(out_ptr + offsets).to(tl.float32)
        scale = tl.sum(og, 1)
        ig = og - tl.exp(o) * scale[:, None]
        tl.store(in_grad_ptr + offsets, ig)


@libentry()
@triton.jit
def log_softmax_backward_kernel_multirow_tail(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    ROW_START,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid_m = ext.program_id(0)
    m_offsets = ROW_START + pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    mask = m_offsets[:, None] < M
    og = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    o = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.sum(og, 1)
    ig = og - tl.exp(o) * scale[:, None]
    tl.store(in_grad_ptr + offsets, ig, mask=mask)


BWD_STAGED_TILE_N = 8192


@libentry()
@triton.jit
def log_softmax_backward_kernel_stage1(
    out_grad_ptr,
    partial_ptr,
    N,
    N_CHUNKS,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_c = ext.program_id(1)
    offset = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offset < N
    og = tl.load(out_grad_ptr + pid_m * N + offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(partial_ptr + pid_m * N_CHUNKS + pid_c, tl.sum(og, 0))


@libentry()
@triton.jit
def log_softmax_backward_kernel_stage2(
    partial_ptr,
    scale_ptr,
    N_CHUNKS,
    BLOCK_MID: tl.constexpr,
):
    pid_m = ext.program_id(0)
    offset = tl.arange(0, BLOCK_MID)
    p = tl.load(
        partial_ptr + pid_m * N_CHUNKS + offset,
        mask=offset < N_CHUNKS,
        other=0.0,
    )
    tl.store(scale_ptr + pid_m, tl.sum(p, 0))


@libentry()
@triton.jit
def log_softmax_backward_kernel_stage3(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    scale_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_c = ext.program_id(1)
    offset = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offset < N
    scale = tl.load(scale_ptr + pid_m).to(tl.float32)
    og = tl.load(out_grad_ptr + pid_m * N + offset, mask=mask, other=0.0).to(tl.float32)
    o = tl.load(out_ptr + pid_m * N + offset, mask=mask, other=0.0).to(tl.float32)
    ig = og - tl.exp(o) * scale
    tl.store(in_grad_ptr + pid_m * N + offset, ig, mask=mask)


def _backward_launch_staged(output, grad_output, in_grad, M, N):
    n_chunks = triton.cdiv(N, BWD_STAGED_TILE_N)
    scale = torch.empty((M,), dtype=torch.float32, device=grad_output.device)
    if n_chunks == 1:
        log_softmax_backward_kernel_stage1[(M, 1)](
            grad_output,
            scale,
            N,
            1,
            BLOCK_N=BWD_STAGED_TILE_N,
            buffer_size_limit=2048,
            num_warps=8,
        )
    else:
        partial = torch.empty(
            (M, n_chunks), dtype=torch.float32, device=grad_output.device
        )
        log_softmax_backward_kernel_stage1[(M, n_chunks)](
            grad_output,
            partial,
            N,
            n_chunks,
            BLOCK_N=BWD_STAGED_TILE_N,
            buffer_size_limit=2048,
            num_warps=8,
        )
        log_softmax_backward_kernel_stage2[(M,)](
            partial,
            scale,
            n_chunks,
            BLOCK_MID=triton.next_power_of_2(n_chunks),
            buffer_size_limit=2048,
            num_warps=8,
        )
    log_softmax_backward_kernel_stage3[(M, n_chunks)](
        output,
        grad_output,
        in_grad,
        scale,
        N,
        BLOCK_N=BWD_STAGED_TILE_N,
        buffer_size_limit=2048,
        num_warps=8,
    )


def _forward_launch(out, inp, M, N, K=1):
    if K == 1:
        if N <= FWD_MULTIROW_MAX_N:
            _fwd_singlepass(out, inp, M, N)
        else:
            _fwd_chunk_split(out, inp, M, N)
    else:
        grid = (M * K, 1, 1)
        log_softmax_kernel_inner[grid](
            out,
            inp,
            M,
            N,
            K,
            buffer_size_limit=2048,
            isCloseVectorization=True,
            is_use_mask_zero=True,
        )


def _backward_launch(output, grad_output, in_grad, M, N):
    if N == 1:
        grid = (triton.cdiv(M, 256), 1, 1)
        log_softmax_backward_kernel_flat1[grid](
            output,
            grad_output,
            in_grad,
            M,
            buffer_size_limit=2048,
            num_warps=8,
        )
    elif N <= BWD_MULTIROW_MAX_N and (N & (N - 1)) == 0:
        if N <= 64:
            tile_m = 16
        elif N <= 256:
            tile_m = 64
        elif N <= 1024:
            tile_m = 32
        elif N <= 4096:
            tile_m = 16
        else:
            tile_m = 8
        tile_m = min(tile_m, _prev_pow2(max(16, M // 8)))
        nfull, tail = divmod(M, tile_m)
        log_softmax_backward_kernel_multirow[(nfull, 1, 1)](
            output,
            grad_output,
            in_grad,
            M,
            N,
            TILE_M=tile_m,
            NEED_MASK=False,
            buffer_size_limit=2048,
            num_warps=8,
        )
        if tail:
            row_start = nfull * tile_m
            tile_n = max(triton.next_power_of_2(N), 64)
            log_softmax_backward_kernel_perrow[(tail, 1, 1)](
                output[row_start:],
                grad_output[row_start:],
                in_grad[row_start:],
                tail,
                N,
                TILE_N=tile_n,
                NEED_MASK=(N % tile_n) != 0,
                buffer_size_limit=2048,
                num_warps=8,
            )
    elif N <= BWD_SINGLE_TILE_MAX_N:
        grid = (M, 1, 1)
        log_softmax_backward_kernel_perrow[grid](
            output,
            grad_output,
            in_grad,
            M,
            N,
            TILE_N=triton.next_power_of_2(N),
            NEED_MASK=True,
            buffer_size_limit=2048,
            num_warps=8,
        )
    else:
        tile_n = (
            BWD_MT_TILE_N if grad_output.dtype == torch.bfloat16 else BWD_MT_TILE_N_WIDE
        )
        grid = (M, 1, 1)
        log_softmax_backward_kernel_perrow_mt[grid](
            output,
            grad_output,
            in_grad,
            M,
            N,
            TILE_N=tile_n,
            buffer_size_limit=2048,
            num_warps=8,
        )


def log_softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_KUNLUNXIN LOG_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    inp = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    out = torch.empty_like(inp, dtype=dtype)
    K = inp.numel() // M // N

    with torch_device_fn.device(inp.device):
        if K > 1:
            inp_view = inp.view(M, N, K).transpose(1, 2).contiguous()
            inp_reshaped = inp_view.view(M * K, N)
            out_reshaped = torch.empty_like(inp_reshaped, dtype=dtype)

            _forward_launch(out_reshaped, inp_reshaped, M * K, N)
            out = (
                out_reshaped.view(M, K, N).transpose(1, 2).contiguous().view(out.shape)
            )
        else:
            _forward_launch(out, inp, M, N)
    return out


def log_softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS_KUNLUNXIN LOG_SOFTMAX_BACKWARD")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    output = output.contiguous()
    in_grad = torch.empty_like(output, dtype=input_dtype)
    K = output.numel() // M // N

    with torch_device_fn.device(in_grad.device):
        if K > 1:
            out_grad_view = grad_output.view(M, N, K).transpose(1, 2).contiguous()
            out_view = output.view(M, N, K).transpose(1, 2).contiguous()
            out_grad_reshaped = out_grad_view.view(M * K, N)
            out_reshaped = out_view.view(M * K, N)
            in_grad_reshaped = torch.empty_like(out_reshaped, dtype=input_dtype)

            _backward_launch(
                out_reshaped, out_grad_reshaped, in_grad_reshaped, M * K, N
            )
            origin_dim = output.ndim
            if origin_dim == 3:
                m, n, k = output.shape
            elif origin_dim == 2:
                m, n = output.shape
            if M == 1 and origin_dim == 2:
                in_grad = in_grad_reshaped.view(K, N).transpose(0, 1).contiguous()
            elif M == 1 and origin_dim == 3:
                in_grad = in_grad_reshaped.transpose(0, 1).view(m, n, k).contiguous()
            else:
                in_grad = in_grad_reshaped.view(m, k, n).transpose(1, 2).contiguous()
        else:
            _backward_launch(output, grad_output, in_grad, M, N)
    return in_grad


def log_softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS_KUNLUNXIN LOG_SOFTMAX_OUT")
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    dtype = torch.float32 if half_to_float else self.dtype
    if out.dtype != dtype:
        raise RuntimeError(
            f"_log_softmax.out: expected out dtype {dtype}, got {out.dtype}"
        )
    if tuple(out.shape) != tuple(self.shape):
        out.resize_(self.shape)

    if self.numel() == 0:
        return out

    M = 1
    for i in range(dim):
        M *= self.shape[i]
    N = self.shape[dim]
    inp = self.contiguous()
    K = inp.numel() // M // N
    if N == 1 and out.is_contiguous():
        with torch_device_fn.device(inp.device):
            _fwd_n1_flat(out, inp)
        return out
    if K > 1:
        inp_t = torch.empty((M * K, N), dtype=inp.dtype, device=inp.device)
        torch.ops.aten._copy_from(
            inp.view(M, N, K).transpose(1, 2), inp_t.view(M, K, N), False
        )
        tmp = torch.empty((M * K, N), dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            _forward_launch(tmp, inp_t, M * K, N)
        src = tmp.view(M, K, N).transpose(1, 2)
        if out.is_contiguous():
            torch.ops.aten._copy_from(src, out.view(M, N, K), False)
        else:
            scratch = torch.empty((M, N, K), dtype=dtype, device=out.device)
            torch.ops.aten._copy_from(src, scratch, False)
            torch.ops.aten._copy_from(scratch.view(self.shape), out, False)
        return out
    if not out.is_contiguous():
        tmp = torch.empty(self.shape, dtype=dtype, device=self.device)
        with torch_device_fn.device(inp.device):
            _forward_launch(tmp, inp, M, N, K)
        torch.ops.aten._copy_from(tmp, out, False)
        return out
    with torch_device_fn.device(inp.device):
        _forward_launch(out, inp, M, N, K)
    return out


def log_softmax_backward_out(grad_output, output, dim, input_dtype, *, out):
    logger.debug("GEMS_KUNLUNXIN LOG_SOFTMAX_BACKWARD_OUT")
    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]
    if tuple(out.shape) != tuple(output.shape):
        out.resize_(output.shape)
    K = output.numel() // M // N
    grad_output = grad_output.contiguous()
    output = output.contiguous()
    if K == 1 and out.is_contiguous() and out.dtype == input_dtype:
        with torch_device_fn.device(out.device):
            _backward_launch(output, grad_output, out, M, N)
        return out
    res = log_softmax_backward(grad_output, output, dim, input_dtype)
    if tuple(out.shape) != tuple(res.shape):
        out.resize_(res.shape)
    out.copy_(res)
    return out
