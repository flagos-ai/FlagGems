import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)


@triton.jit
def next_multiple_of(a, b):
    return tl.cdiv(a, b) * b


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        input_ptr += pid_m * N
        output_ptr += pid_m * N
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
            output_ptr.dtype.element_ty
        )
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = e / z
        tl.store(output_ptr + n_offsets, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N
        output_ptr += pid_m * N

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets)
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets)
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o)
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o, mask=mask)


_SM_MR_MAX_N = 4096
_SM_N_TILE_M = [(16, 64), (64, 32), (256, 16), (1024, 8), (2048, 4), (4096, 8)]


@triton.jit
def softmax_kernel_multirow(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    mo = pid * TILE_M + tl.arange(0, TILE_M)
    no = tl.arange(0, N)
    off = mo[:, None] * N + no[None, :]
    inp = tl.load(input_ptr + off).to(output_ptr.dtype.element_ty)
    m = tl.max(inp, 1)
    e = tl.exp(inp - m[:, None])
    z = tl.sum(e, 1)
    out = e / z[:, None]
    tl.store(output_ptr + off, out)


_SM_CHUNK_BN = 8192
_SM_TAIL_PIECE = 4096


def _sm_pow2_tail_pieces(n, cap=_SM_TAIL_PIECE):
    """Split a row tail into (pieces, 64-lane remainder) - see log_softmax."""
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


@libentry()
@triton.jit
def softmax_kernel_chunk(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // C_FULL
    c = pid % C_FULL
    n_offsets = tl.arange(0, BLOCK_N)
    off = pid * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    tl.store(partial_m_ptr + row * C + c, m)
    tl.store(partial_z_ptr + row * C + c, z)


_SM_COMBINE_GW = 1024


def _sm_combine_geometry(C):
    """(group width, group count, padded per-row stride) for the partial merge.

    The width is clamped to [64, 1024]: tiles of <= 32 lanes miscompile on this
    XPU, a single tile spanning all partials is untrustworthy, and 64 keeps
    every padded row a whole number of 64-element store units.  The padded
    stride stays <= max(64, 2 * C), so the partial buffers grow at most 2x.
    """
    gw = min(max(64, triton.next_power_of_2(C)), _SM_COMBINE_GW)
    ng = triton.cdiv(C, gw)
    return gw, ng, ng * gw


@libentry()
@triton.jit
def softmax_combine_pad_init(
    partial_m_ptr,
    partial_z_ptr,
    CP: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * CP + tl.arange(0, CP)
    tl.store(partial_m_ptr + off, tl.full([CP], float("-inf"), tl.float32))
    tl.store(partial_z_ptr + off, tl.zeros([CP], tl.float32))


@libentry()
@triton.jit
def softmax_chunk_combine(
    m_ptr,
    z_ptr,
    partial_m_ptr,
    partial_z_ptr,
    NG: tl.constexpr,
    GW: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, GW)
    base = pid * NG * GW
    m = float("-inf")
    for g in tl.range(NG):
        mc = tl.load(partial_m_ptr + base + g * GW + lane)
        m = tl.maximum(m, tl.max(mc, 0))
    z = 0.0
    for g in tl.range(NG):
        off = base + g * GW + lane
        mc = tl.load(partial_m_ptr + off)
        zc = tl.load(partial_z_ptr + off)
        z += tl.sum(zc * tl.exp(mc - m), 0)
    tl.store(m_ptr + pid, m)
    tl.store(z_ptr + pid, z)


@libentry()
@triton.jit
def softmax_chunk_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    z_ptr,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // C_FULL
    m = tl.load(m_ptr + row).to(tl.float32)
    z = tl.load(z_ptr + row).to(tl.float32)
    n_offsets = tl.arange(0, BLOCK_N)
    off = pid * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, tl.exp(x - m) / z)


@libentry()
@triton.jit
def softmax_kernel_chunk_strided(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    N,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // C_FULL
    c = pid % C_FULL
    n_offsets = tl.arange(0, BLOCK_N)
    off = row * N + c * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    tl.store(partial_m_ptr + row * C + c, m)
    tl.store(partial_z_ptr + row * C + c, z)


@libentry()
@triton.jit
def softmax_chunk_pass_strided(
    output_ptr,
    input_ptr,
    m_ptr,
    z_ptr,
    N,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // C_FULL
    m = tl.load(m_ptr + row).to(tl.float32)
    z = tl.load(z_ptr + row).to(tl.float32)
    n_offsets = tl.arange(0, BLOCK_N)
    c = pid % C_FULL
    off = row * N + c * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, tl.exp(x - m) / z)


@libentry()
@triton.jit
def softmax_tail_piece_partial(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    M,
    N,
    C_STRIDE,
    T_SLOT,
    TAIL_BASE,
    PLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    n_offsets = TAIL_BASE + tl.arange(0, PLEN)
    off = pid * N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    po = pid * C_STRIDE + T_SLOT
    tl.store(partial_m_ptr + po, m)
    tl.store(partial_z_ptr + po, z)


@libentry()
@triton.jit
def softmax_tail_piece_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    z_ptr,
    N,
    TAIL_BASE,
    PLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    m = tl.load(m_ptr + pid).to(tl.float32)
    z = tl.load(z_ptr + pid).to(tl.float32)
    n_offsets = TAIL_BASE + tl.arange(0, PLEN)
    off = pid * N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, tl.exp(x - m) / z)


@libentry()
@triton.jit
def softmax_tail_masked_partial(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    N,
    C_STRIDE,
    T_SLOT,
    TAIL_BASE,
    TAIL_LEN,
):
    pid = tl.program_id(0)
    n_offsets = tl.arange(0, 64)
    within = n_offsets < TAIL_LEN
    off = pid * N + TAIL_BASE + n_offsets
    x = tl.load(input_ptr + off, mask=within, other=float("-inf")).to(tl.float32)
    m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - m), 0)
    po = pid * C_STRIDE + T_SLOT
    tl.store(partial_m_ptr + po, m)
    tl.store(partial_z_ptr + po, z)


@libentry()
@triton.jit
def softmax_tail_masked_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    z_ptr,
    N,
    TAIL_BASE,
    TAIL_LEN,
):
    pid = tl.program_id(0)
    m = tl.load(m_ptr + pid).to(tl.float32)
    z = tl.load(z_ptr + pid).to(tl.float32)
    n_offsets = tl.arange(0, 64)
    within = n_offsets < TAIL_LEN
    off = pid * N + TAIL_BASE + n_offsets
    x = tl.load(input_ptr + off, mask=within, other=float("-inf")).to(tl.float32)
    tl.store(output_ptr + off, tl.exp(x - m) / z, mask=within)


def _softmax_chunk_split(output, inp, M, N):
    """Chunked split forward for N > _SM_CHUNK_SPLIT_MIN (see dispatch)."""
    c_full = N // _SM_CHUNK_BN
    taillen = N - c_full * _SM_CHUNK_BN
    pieces, rrem = _sm_pow2_tail_pieces(taillen) if taillen else ([], 0)
    have_rem = rrem != 0
    C = c_full + len(pieces) + (1 if have_rem else 0)
    GW, NG, CP = _sm_combine_geometry(C)
    pm = torch.empty((M * CP,), dtype=torch.float32, device=inp.device)
    pz = torch.empty((M * CP,), dtype=torch.float32, device=inp.device)
    m_out = torch.empty((M,), dtype=torch.float32, device=inp.device)
    z_out = torch.empty((M,), dtype=torch.float32, device=inp.device)
    if CP != C:
        softmax_combine_pad_init[(M, 1, 1)](
            pm,
            pz,
            CP=CP,
            buffer_size_limit=2048,
            num_warps=8,
        )
    base = c_full * _SM_CHUNK_BN
    for slot, plen in enumerate(pieces):
        softmax_tail_piece_partial[(M, 1, 1)](
            pm,
            pz,
            inp,
            M,
            N,
            CP,
            c_full + slot,
            base,
            PLEN=plen,
            buffer_size_limit=2048,
            num_warps=8,
        )
        base += plen
    if have_rem:
        softmax_tail_masked_partial[(M, 1, 1)](
            pm,
            pz,
            inp,
            N,
            CP,
            c_full + len(pieces),
            base,
            rrem,
            buffer_size_limit=2048,
            num_warps=8,
        )
    if c_full:
        if pieces or have_rem:
            if M == 1:
                softmax_kernel_chunk[(c_full, 1, 1)](
                    pm,
                    pz,
                    inp,
                    c_full,
                    CP,
                    BLOCK_N=_SM_CHUNK_BN,
                    buffer_size_limit=2048,
                    num_warps=8,
                )
            else:
                softmax_kernel_chunk_strided[(M * c_full, 1, 1)](
                    pm,
                    pz,
                    inp,
                    N,
                    c_full,
                    CP,
                    BLOCK_N=_SM_CHUNK_BN,
                    buffer_size_limit=2048,
                    num_warps=8,
                )
        else:
            softmax_kernel_chunk[(M * c_full, 1, 1)](
                pm,
                pz,
                inp,
                c_full,
                CP,
                BLOCK_N=_SM_CHUNK_BN,
                buffer_size_limit=2048,
                num_warps=8,
            )
    softmax_chunk_combine[(M, 1, 1)](
        m_out,
        z_out,
        pm,
        pz,
        NG=NG,
        GW=GW,
        buffer_size_limit=2048,
        num_warps=8,
    )
    if c_full:
        if pieces or have_rem:
            if M == 1:
                softmax_chunk_pass[(c_full, 1, 1)](
                    output,
                    inp,
                    m_out,
                    z_out,
                    c_full,
                    C,
                    BLOCK_N=_SM_CHUNK_BN,
                    buffer_size_limit=2048,
                    num_warps=8,
                )
            else:
                softmax_chunk_pass_strided[(M * c_full, 1, 1)](
                    output,
                    inp,
                    m_out,
                    z_out,
                    N,
                    c_full,
                    C,
                    BLOCK_N=_SM_CHUNK_BN,
                    buffer_size_limit=2048,
                    num_warps=8,
                )
        else:
            softmax_chunk_pass[(M * c_full, 1, 1)](
                output,
                inp,
                m_out,
                z_out,
                c_full,
                C,
                BLOCK_N=_SM_CHUNK_BN,
                buffer_size_limit=2048,
                num_warps=8,
            )
    base = c_full * _SM_CHUNK_BN
    for plen in pieces:
        softmax_tail_piece_pass[(M, 1, 1)](
            output,
            inp,
            m_out,
            z_out,
            N,
            base,
            PLEN=plen,
            buffer_size_limit=2048,
            num_warps=8,
        )
        base += plen
    if have_rem:
        softmax_tail_masked_pass[(M, 1, 1)](
            output,
            inp,
            m_out,
            z_out,
            N,
            base,
            rrem,
            buffer_size_limit=2048,
            num_warps=8,
        )


_SM_CHUNK_SPLIT_MAX_N = 8192 * 1024


def _softmax_forward_launch(output, inp, M, N):
    """Inner launch on a contiguous [M, N] view (reduced dim innermost)."""
    use_multirow = N <= _SM_MR_MAX_N and ((N & (N - 1)) == 0)
    if use_multirow:
        tile_m = 1
        for n_hi, tm in _SM_N_TILE_M:
            if N <= n_hi:
                tile_m = tm
                break
        if M % tile_m == 0:
            grid = (M // tile_m,)
            if tile_m * N > 8192:
                softmax_kernel_multirow[grid](
                    output,
                    inp,
                    M,
                    N=N,
                    TILE_M=tile_m,
                    num_warps=4,
                    buffer_size_limit=2048,
                )
            else:
                softmax_kernel_multirow[grid](
                    output, inp, M, N=N, TILE_M=tile_m, num_warps=4
                )
            return
    if N > _SM_CHUNK_SPLIT_MAX_N and M > 1:
        grid = (M, 1, 1)
        softmax_kernel_inner[grid](
            output,
            inp,
            M,
            N,
            buffer_size_limit=2048,
            is_use_mask_zero=True,
        )
        return
    if N > _SM_MR_MAX_N:
        if M * (N // _SM_CHUNK_BN) < 1024 or M == 1:
            _softmax_chunk_split(output, inp, M, N)
        else:
            grid = (M, 1, 1)
            softmax_kernel_inner[grid](
                output,
                inp,
                M,
                N,
                buffer_size_limit=2048,
                is_use_mask_zero=True,
            )
        return
    grid = (M, 1, 1)
    softmax_kernel_inner[grid](
        output,
        inp,
        M,
        N,
        buffer_size_limit=2048,
        is_use_mask_zero=True,
    )


def softmax_backward_kernel_inner_heru_tile_n(args):
    N = args["N"]
    if N <= 32768:
        return triton.next_power_of_2(N)
    return 4096


def softmax_backward_kernel_inner_heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


@libentry()
@triton.heuristics(
    values={
        "TILE_N": softmax_backward_kernel_inner_heru_tile_n,
        "ONE_TILE_PER_CTA": softmax_backward_kernel_inner_heur_one_tile_per_cta,
    },
)
@triton.jit
def softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    out_ptr += pid_m * N
    out_grad_ptr += pid_m * N
    in_grad_ptr += pid_m * N
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        out_tile = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        scale = tl.sum(out_tile * out_grad_tile, 0)
        in_grad_tile = out_tile * (out_grad_tile - scale)
        tl.store(in_grad_ptr + n_offsets, in_grad_tile, mask=mask)
    else:
        scale = tl.zeros([TILE_N], dtype=tl.float32)
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            out_tile = tl.load(out_ptr + n_offsets).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            scale += out_tile * out_grad_tile
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            out_tile = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            scale += out_tile * out_grad_tile
        scale = tl.sum(scale, 0)

        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            out_tile = tl.load(out_ptr + n_offsets).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            in_grad_tile = out_tile * (out_grad_tile - scale)
            tl.store(in_grad_ptr + n_offsets, in_grad_tile)
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            out_tile = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            in_grad_tile = out_tile * (out_grad_tile - scale)
            tl.store(in_grad_ptr + n_offsets, in_grad_tile, mask=mask)


_SB_MR_MAX_N = 4096
_SB_N_TILE_M = [(16, 64), (64, 32), (256, 16), (1024, 8), (2048, 4), (4096, 2)]
_SB_WIDE = 8192


@triton.jit
def softmax_backward_kernel_multirow(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    mo = pid * TILE_M + tl.arange(0, TILE_M)
    no = tl.arange(0, N)
    off = mo[:, None] * N + no[None, :]
    o = tl.load(out_ptr + off).to(tl.float32)
    g = tl.load(out_grad_ptr + off).to(tl.float32)
    s = tl.sum(o * g, 1)
    tl.store(in_grad_ptr + off, o * (g - s[:, None]))


@triton.jit
def softmax_backward_kernel_multirow_pad(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    W: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    mo = tl.minimum(pid * TILE_M + tl.arange(0, TILE_M), M - 1)
    no = tl.arange(0, W)
    nc = tl.minimum(no, N - 1)
    off = mo[:, None] * N + nc[None, :]
    o = tl.load(out_ptr + off).to(tl.float32)
    g = tl.load(out_grad_ptr + off).to(tl.float32)
    s = tl.sum(tl.where(no[None, :] < N, o * g, 0.0), 1)
    tl.store(in_grad_ptr + off, o * (g - s[:, None]))


@triton.jit
def softmax_backward_kernel_perrow_p2(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    W: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < M:
        out_ptr += pid * N
        out_grad_ptr += pid * N
        in_grad_ptr += pid * N
        acc = tl.zeros([W], dtype=tl.float32)
        for start_n in range(0, N, W):
            n_offsets = start_n + tl.arange(0, W)
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            o = tl.load(out_ptr + n_offsets).to(tl.float32)
            acc += o * og
        scale = tl.sum(acc, 0)
        for start_n in range(0, N, W):
            n_offsets = start_n + tl.arange(0, W)
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            o = tl.load(out_ptr + n_offsets).to(tl.float32)
            ig = o * (og - scale)
            tl.store(in_grad_ptr + n_offsets, ig)


@triton.jit
def softmax_backward_kernel_perrow_p2_tail(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    p_tail_ptr,
    scale_ptr,
    N,
    PREV,
):
    pid = tl.program_id(0)
    out_ptr += pid * N
    out_grad_ptr += pid * N
    in_grad_ptr += pid * N
    acc = tl.zeros([4096], dtype=tl.float32)
    for start_n in range(0, PREV, 4096):
        n_offsets = start_n + tl.arange(0, 4096)
        og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
        o = tl.load(out_ptr + n_offsets).to(tl.float32)
        acc += o * og
    scale = tl.sum(acc, 0) + tl.load(p_tail_ptr + pid)
    tl.store(scale_ptr + pid, scale)
    for start_n in range(0, PREV, 4096):
        n_offsets = start_n + tl.arange(0, 4096)
        og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
        o = tl.load(out_ptr + n_offsets).to(tl.float32)
        ig = o * (og - scale)
        tl.store(in_grad_ptr + n_offsets, ig)


@triton.jit
def softmax_backward_kernel_tail_partial(
    p_ptr,
    out_ptr,
    out_grad_ptr,
    N,
    PREV,
):
    pid = tl.program_id(0)
    tno = tl.arange(0, 4096)
    tmask = tno < (N - PREV)
    o = tl.load(out_ptr + pid * N + PREV + tno, mask=tmask, other=0.0).to(tl.float32)
    g = tl.load(out_grad_ptr + pid * N + PREV + tno, mask=tmask, other=0.0).to(
        tl.float32
    )
    tl.store(p_ptr + pid, tl.sum(o * g, 0))


@triton.jit
def softmax_backward_kernel_tail_pass(
    in_grad_ptr,
    scale_ptr,
    out_ptr,
    out_grad_ptr,
    N,
    PREV,
):
    pid = tl.program_id(0)
    scale = tl.load(scale_ptr + pid)
    tno = tl.arange(0, 4096)
    tmask = tno < (N - PREV)
    o = tl.load(out_ptr + pid * N + PREV + tno, mask=tmask, other=0.0).to(tl.float32)
    g = tl.load(out_grad_ptr + pid * N + PREV + tno, mask=tmask, other=0.0).to(
        tl.float32
    )
    tl.store(in_grad_ptr + pid * N + PREV + tno, o * (g - scale), mask=tmask)


def _softmax_backward_launch_k1(output, grad_output, in_grad, M, N, input_dtype):
    if N <= _SB_MR_MAX_N:
        TILE_M = 4
        for n_hi, tm in _SB_N_TILE_M:
            if N <= n_hi:
                TILE_M = tm
                break
        grid = (triton.cdiv(M, TILE_M),)
        if N == triton.next_power_of_2(N) and M % TILE_M == 0:
            softmax_backward_kernel_multirow[grid](
                output,
                grad_output,
                in_grad,
                M,
                N=N,
                TILE_M=TILE_M,
                num_warps=8,
            )
        else:
            softmax_backward_kernel_multirow_pad[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
                W=triton.next_power_of_2(N),
                TILE_M=TILE_M,
                num_warps=8,
            )
    else:
        if N % _SB_WIDE == 0:
            grid = (M,)
            softmax_backward_kernel_perrow_p2[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
                W=_SB_WIDE,
            )
        elif N % 4096 == 0:
            grid = (M,)
            softmax_backward_kernel_perrow_p2[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
                W=4096,
            )
        else:
            prev = (N // 4096) * 4096
            p_tail = torch.empty((M,), dtype=torch.float32, device=in_grad.device)
            scale_buf = torch.empty((M,), dtype=torch.float32, device=in_grad.device)
            grid = (M,)
            softmax_backward_kernel_tail_partial[grid](
                p_tail, output, grad_output, N, prev
            )
            softmax_backward_kernel_perrow_p2_tail[grid](
                output,
                grad_output,
                in_grad,
                p_tail,
                scale_buf,
                N,
                prev,
            )
            softmax_backward_kernel_tail_pass[grid](
                in_grad, scale_buf, output, grad_output, N, prev
            )


def softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX")

    if self.ndim == 0:
        assert dim in (-1, 0), "Invalid dim"
        dtype = torch.float32 if half_to_float else self.dtype
        out = torch.empty_like(self, dtype=dtype)
        with torch_device_fn.device(self.device):
            softmax_kernel_inner[(1, 1, 1)](
                out,
                self,
                1,
                1,
                buffer_size_limit=2048,
                is_use_mask_zero=True,
            )
        return out

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    if self.numel() == 0:
        out_shape = list(self.shape)
        dtype = torch.float32 if half_to_float else self.dtype
        out = torch.empty(out_shape, dtype=dtype, device=self.device)
        zero_(out)
        return out

    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    self = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    K = self.numel() // M // N

    with torch_device_fn.device(self.device):
        if K > 1:
            inp_view = self.view(M, N, K).transpose(1, 2)
            inp_reshaped = torch.empty((M * K, N), dtype=self.dtype, device=self.device)
            if not tle_copy(inp_view, inp_reshaped):
                torch.ops.aten._copy_from(inp_view, inp_reshaped, False)
            out_reshaped = torch.empty((M * K, N), dtype=dtype, device=self.device)

            _softmax_forward_launch(out_reshaped, inp_reshaped, M * K, N)

            out = out_reshaped.view(M, K, N).transpose(1, 2).reshape(self.shape)
        else:
            out = torch.empty_like(self, dtype=dtype)
            _softmax_forward_launch(out, self, M, N)
    return out


_SM_N1_BLOCK = 512


@libentry()
@triton.jit
def softmax_kernel_n1(
    output_ptr,
    input_ptr,
    n_elem,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    e = tl.exp(x - x)
    tl.store(output_ptr + offs, e / e, mask=mask)


def _softmax_n1_flat(out, inp):
    n_elem = inp.numel()
    softmax_kernel_n1[(triton.cdiv(n_elem, _SM_N1_BLOCK), 1, 1)](
        out,
        inp,
        n_elem,
        BLOCK=_SM_N1_BLOCK,
        buffer_size_limit=2048,
    )


def _native_contiguous(t):
    """Materialize `t` contiguously through the native strided copy.

    `Tensor.contiguous()` lowers to aten::contiguous -> aten::copy_, and
    `copy_` IS a gems-registered op, so inside `flag_gems.use_gems()` it turns
    into a gems strided pointwise copy (measured 57-1600x slower than the
    native XPU strided copy). `aten::_copy_from` is never overridden by gems.
    """
    dst = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    if not tle_copy(t, dst):
        torch.ops.aten._copy_from(t, dst, False)
    return dst


def softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX_OUT")

    if self.ndim == 0:
        assert dim in (-1, 0), "Invalid dim"
        dtype = torch.float32 if half_to_float else self.dtype
        if out.dtype != dtype:
            raise RuntimeError(
                f"_softmax.out: expected out dtype {dtype}, got {out.dtype}"
            )
        out.copy_(softmax(self, dim, half_to_float))
        return out

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    if self.numel() == 0:
        if tuple(out.shape) != tuple(self.shape):
            out.resize_(self.shape)
        zero_(out)
        return out

    dtype = torch.float32 if half_to_float else self.dtype
    if tuple(out.shape) != tuple(self.shape):
        out.resize_(self.shape)
    if out.dtype != dtype:
        raise RuntimeError(f"_softmax.out: expected out dtype {dtype}, got {out.dtype}")

    dim = dim % self.ndim
    M = 1
    for i in range(dim):
        M *= self.shape[i]
    N = self.shape[dim]
    inp = self if self.is_contiguous() else _native_contiguous(self)
    K = inp.numel() // M // N

    if N == 1 and out.is_contiguous():
        with torch_device_fn.device(inp.device):
            _softmax_n1_flat(out, inp)
        return out

    with torch_device_fn.device(inp.device):
        if K > 1:
            inp_t = torch.empty((M * K, N), dtype=inp.dtype, device=inp.device)
            inp_view = inp.view(M, N, K).transpose(1, 2)
            if not tle_copy(inp_view, inp_t.view(M, K, N)):
                torch.ops.aten._copy_from(inp_view, inp_t.view(M, K, N), False)
            tmp = torch.empty((M * K, N), dtype=dtype, device=inp.device)
            _softmax_forward_launch(tmp, inp_t, M * K, N)
            src = tmp.view(M, K, N).transpose(1, 2)
            if out.is_contiguous():
                if not tle_copy(src, out.view(M, N, K)):
                    torch.ops.aten._copy_from(src, out.view(M, N, K), False)
            else:
                scratch = torch.empty((M, N, K), dtype=dtype, device=out.device)
                if not tle_copy(src, scratch):
                    torch.ops.aten._copy_from(src, scratch, False)
                if not tle_copy(scratch.view(self.shape), out):
                    torch.ops.aten._copy_from(scratch.view(self.shape), out, False)
        elif not out.is_contiguous():
            tmp = torch.empty(self.shape, dtype=dtype, device=self.device)
            _softmax_forward_launch(tmp, inp, M, N)
            if not tle_copy(tmp, out):
                torch.ops.aten._copy_from(tmp, out, False)
        else:
            _softmax_forward_launch(out, inp, M, N)
    return out


def softmax_backward(grad_output, output, dim, input_dtype, grad_input=None):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX_VJP")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = (
        grad_output if grad_output.is_contiguous() else _native_contiguous(grad_output)
    )
    output = output if output.is_contiguous() else _native_contiguous(output)
    K = output.numel() // M // N
    if grad_input is not None and K == 1:
        in_grad = grad_input
    else:
        in_grad = torch.empty_like(output, dtype=input_dtype)

    with torch_device_fn.device(in_grad.device):
        if K > 1:
            out_grad_view = grad_output.view(M, N, K).transpose(1, 2)
            out_view = output.view(M, N, K).transpose(1, 2)
            out_grad_reshaped = torch.empty(
                (M * K, N), dtype=grad_output.dtype, device=grad_output.device
            )
            out_reshaped = torch.empty(
                (M * K, N), dtype=output.dtype, device=output.device
            )
            if not tle_copy(out_grad_view, out_grad_reshaped):
                torch.ops.aten._copy_from(out_grad_view, out_grad_reshaped, False)
            if not tle_copy(out_view, out_reshaped):
                torch.ops.aten._copy_from(out_view, out_reshaped, False)
            in_grad_reshaped = torch.empty(
                (M * K, N), dtype=in_grad.dtype, device=in_grad.device
            )
            _softmax_backward_launch_k1(
                out_reshaped, out_grad_reshaped, in_grad_reshaped, M * K, N, input_dtype
            )
            in_grad = in_grad_reshaped.view(M, K, N).transpose(1, 2).view(output.shape)
        else:
            _softmax_backward_launch_k1(output, grad_output, in_grad, M, N, input_dtype)
    return in_grad


def softmax_backward_out(grad_output, output, dim, input_dtype, *, grad_input):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX_VJP_OUT")
    if tuple(grad_input.shape) != tuple(output.shape):
        grad_input.resize_(output.shape)
    if grad_input.dtype != input_dtype:
        raise RuntimeError(
            f"_softmax_backward_data.out: expected out dtype {input_dtype}, "
            f"got {grad_input.dtype}"
        )
    result = softmax_backward(
        grad_output,
        output,
        dim,
        input_dtype,
        grad_input=grad_input if grad_input.is_contiguous() else None,
    )
    if result is not grad_input:
        if not tle_copy(result, grad_input):
            torch.ops.aten._copy_from(result, grad_input, False)
    return grad_input
