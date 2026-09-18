import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

exp2 = tl_extra_shim.exp2
log2 = tl_extra_shim.log2
logger = logging.getLogger(__name__)

_BLOCK_D = 2048
_MAX_PIECES = 4
_MID_BLOCK = 2048


@triton.jit
def _pd_mode_reduce(diff, p_scalar, MODE: tl.constexpr):
    if MODE == 0:
        return tl.sum(diff * diff)
    elif MODE == 1:
        return tl.sum(diff)
    elif MODE == 2:
        return tl.sum((diff != 0).to(tl.float32))
    elif MODE == 3:
        return tl.max(diff)
    elif MODE == 4:
        return tl.min(diff)
    else:
        return tl.sum(exp2(p_scalar * log2(diff)))


@triton.jit
def _pd_combine(acc, part, MODE: tl.constexpr):
    if MODE == 3:
        return tl.maximum(acc, part)
    elif MODE == 4:
        return tl.minimum(acc, part)
    else:
        return acc + part


@triton.jit
def _pd_finalize(acc, p_scalar, MODE: tl.constexpr):
    if MODE == 0:
        return tl.sqrt(acc)
    elif MODE == 5:
        return exp2((1.0 / p_scalar) * log2(acc))
    else:
        return acc


@triton.jit
def _pd_piece_sum(
    x1_ptr,
    x2_ptr,
    base,
    eps,
    p_scalar,
    MODE: tl.constexpr,
    S: tl.constexpr,
    NP: tl.constexpr,
    NSCALAR: tl.constexpr,
):
    acc = 0.0
    if NP >= 1:
        a = tl.load(x1_ptr + base + tl.arange(0, S)).to(tl.float32)
        b = tl.load(x2_ptr + base + tl.arange(0, S)).to(tl.float32)
        acc = _pd_mode_reduce(tl.abs(a - b + eps), p_scalar, MODE)
    if NP >= 2:
        a = tl.load(x1_ptr + base + S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(x2_ptr + base + S + tl.arange(0, S)).to(tl.float32)
        acc = _pd_combine(
            acc, _pd_mode_reduce(tl.abs(a - b + eps), p_scalar, MODE), MODE
        )
    if NP >= 3:
        a = tl.load(x1_ptr + base + 2 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(x2_ptr + base + 2 * S + tl.arange(0, S)).to(tl.float32)
        acc = _pd_combine(
            acc, _pd_mode_reduce(tl.abs(a - b + eps), p_scalar, MODE), MODE
        )
    if NP >= 4:
        a = tl.load(x1_ptr + base + 3 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(x2_ptr + base + 3 * S + tl.arange(0, S)).to(tl.float32)
        acc = _pd_combine(
            acc, _pd_mode_reduce(tl.abs(a - b + eps), p_scalar, MODE), MODE
        )
    if NP >= 5:
        a = tl.load(x1_ptr + base + 4 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(x2_ptr + base + 4 * S + tl.arange(0, S)).to(tl.float32)
        acc = _pd_combine(
            acc, _pd_mode_reduce(tl.abs(a - b + eps), p_scalar, MODE), MODE
        )
    if NP >= 6:
        a = tl.load(x1_ptr + base + 5 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(x2_ptr + base + 5 * S + tl.arange(0, S)).to(tl.float32)
        acc = _pd_combine(
            acc, _pd_mode_reduce(tl.abs(a - b + eps), p_scalar, MODE), MODE
        )
    if NP >= 7:
        a = tl.load(x1_ptr + base + 6 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(x2_ptr + base + 6 * S + tl.arange(0, S)).to(tl.float32)
        acc = _pd_combine(
            acc, _pd_mode_reduce(tl.abs(a - b + eps), p_scalar, MODE), MODE
        )
    if NP >= 8:
        a = tl.load(x1_ptr + base + 7 * S + tl.arange(0, S)).to(tl.float32)
        b = tl.load(x2_ptr + base + 7 * S + tl.arange(0, S)).to(tl.float32)
        acc = _pd_combine(
            acc, _pd_mode_reduce(tl.abs(a - b + eps), p_scalar, MODE), MODE
        )
    if NSCALAR > 0:
        for j in tl.range(NSCALAR):
            sa = tl.load(x1_ptr + base + NP * S + j).to(tl.float32)
            sb = tl.load(x2_ptr + base + NP * S + j).to(tl.float32)
            sdiff = tl.abs(sa - sb + eps)
            if MODE == 0:
                spart = sdiff * sdiff
            elif MODE == 2:
                spart = (sdiff != 0).to(tl.float32)
            elif MODE == 5:
                spart = exp2(p_scalar * log2(sdiff))
            else:
                spart = sdiff
            acc = _pd_combine(acc, spart, MODE)
    return acc


@libentry()
@triton.jit
def _pd_small_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    D,
    eps,
    p_scalar,
    MODE: tl.constexpr,
    S: tl.constexpr,
    NP: tl.constexpr,
    NSCALAR: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * D
    acc = _pd_piece_sum(
        x1_ptr,
        x2_ptr,
        base,
        eps,
        p_scalar,
        MODE,
        S,
        NP,
        NSCALAR,
    )
    tl.store(out_ptr + pid, _pd_finalize(acc, p_scalar, MODE))


_ROWS = 8
_MULTI_MIN_N = 1024
_D1_BLOCK = 1024


@libentry()
@triton.jit
def _pd_small_multi_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    D,
    eps,
    p_scalar,
    MODE: tl.constexpr,
    S: tl.constexpr,
    NP: tl.constexpr,
    NSCALAR: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    for r in tl.static_range(ROWS):
        row = pid * ROWS + r
        if row < N:
            base = row * D
            acc = _pd_piece_sum(
                x1_ptr,
                x2_ptr,
                base,
                eps,
                p_scalar,
                MODE,
                S,
                NP,
                NSCALAR,
            )
            tl.store(out_ptr + row, _pd_finalize(acc, p_scalar, MODE))


@libentry()
@triton.jit
def _pd_d1_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    eps,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    safe = tl.minimum(off, N - 1)
    a = tl.load(x1_ptr + safe).to(tl.float32)
    b = tl.load(x2_ptr + safe).to(tl.float32)
    d = tl.abs(a - b + eps)
    if MODE == 2:
        d = (d != 0).to(tl.float32)
    tl.store(out_ptr + off, d, mask=off < N)


@libentry()
@triton.jit
def _pd_chunk_kernel(
    x1_ptr,
    x2_ptr,
    mid_ptr,
    D,
    eps,
    p_scalar,
    MID_STRIDE,
    MID_SIZE,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    base = pid_n * D + pid_c * BLOCK
    a = tl.load(x1_ptr + base + tl.arange(0, BLOCK)).to(tl.float32)
    b = tl.load(x2_ptr + base + tl.arange(0, BLOCK)).to(tl.float32)
    diff = tl.abs(a - b + eps)
    m = _pd_mode_reduce(diff, p_scalar, MODE)
    tl.store(mid_ptr + pid_n * MID_STRIDE + pid_c, m)


@libentry()
@triton.jit
def _pd_tail_kernel(
    x1_ptr,
    x2_ptr,
    mid_ptr,
    N,
    D,
    T,
    eps,
    p_scalar,
    MID_SIZE,
    MID_STRIDE,
    MODE: tl.constexpr,
    S: tl.constexpr,
    NP: tl.constexpr,
    NSCALAR: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * D + D - T
    acc = _pd_piece_sum(
        x1_ptr,
        x2_ptr,
        base,
        eps,
        p_scalar,
        MODE,
        S,
        NP,
        NSCALAR,
    )
    tl.store(mid_ptr + pid * MID_STRIDE + MID_SIZE, acc)


@libentry()
@triton.jit
def _pd_mid_reduce_kernel(
    mid_ptr,
    out_ptr,
    MID,
    STRIDE_IN,
    STRIDE_OUT,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    off = pid_c * BLOCK + tl.arange(0, BLOCK)
    m = tl.load(mid_ptr + pid_n * STRIDE_IN + off).to(tl.float32)
    if MODE == 3:
        acc = tl.max(m)
    elif MODE == 4:
        acc = tl.min(m)
    else:
        acc = tl.sum(m)
    tl.store(out_ptr + pid_n * STRIDE_OUT + pid_c, acc)


@libentry()
@triton.jit
def _pd_final_kernel(
    mid_ptr,
    out_ptr,
    p_scalar,
    MID_STRIDE,
    MODE: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    pid = tl.program_id(0)
    off = tl.arange(0, BLOCK_MID)
    m = tl.load(mid_ptr + pid * MID_STRIDE + off).to(tl.float32)
    if MODE == 3:
        acc = tl.max(m)
    elif MODE == 4:
        acc = tl.min(m)
    else:
        acc = tl.sum(m)
    tl.store(out_ptr + pid, _pd_finalize(acc, p_scalar, MODE))


def _mode_of(p):
    if p == 0.0:
        return 2
    if p == 1.0:
        return 1
    if p == 2.0:
        return 0
    if math.isinf(p):
        return 3 if p > 0 else 4
    return 5


def _piece_args(t):
    """Uniform-width piece decomposition of t.

    Returns (S, NP, NSCALAR): NP tiles of uniform width S (power of two) plus
    NSCALAR trailing lanes covered by a scalar loop. Widths are kept uniform
    because mixed-width live tiles blow the XPU uni_sram budget; S maximises
    the covered width (min(NP, 8) * S) over powers of two <= 512.
    """
    if t <= 0:
        return 0, 0, 0
    best = (0, 0)
    S = 512
    while S > 0:
        n = t // S
        if n > 0:
            n_used = min(n, 8)
            cov = n_used * S
            if cov > best[0]:
                best = (cov, S)
        S //= 2
    _, S = best
    np = min(t // S, 8)
    return S, np, t - np * S


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN PAIRWISE_DISTANCE")
    if x1.shape != x2.shape:
        x1, x2 = torch.broadcast_tensors(x1, x2)
    if not x1.is_contiguous():
        x1 = x1.contiguous()
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    D = x1.shape[-1]

    if D == 0:
        if p == float("inf") or p == float("-inf"):
            raise RuntimeError(
                "pairwise_distance cannot compute the inf/-inf norm on an empty "
                "reduction dimension (no identity element)"
            )
        out = torch.zeros(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
        if keepdim:
            out = out.unsqueeze(-1)
        return out

    N = x1.numel() // D
    out = torch.empty(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
    if keepdim:
        out = out.unsqueeze(-1)

    mode = _mode_of(p)
    p_scalar = float(p) if mode == 5 else 1.0

    with torch_device_fn.device(x1.device):
        if D <= _BLOCK_D:
            PS, PNP, PNSC = _piece_args(D)
            _pd_small_kernel[(N,)](
                x1,
                x2,
                out,
                N,
                D,
                eps,
                p_scalar,
                MODE=mode,
                S=PS,
                NP=PNP,
                NSCALAR=PNSC,
            )
        else:
            MID = D // _BLOCK_D
            T = D - MID * _BLOCK_D
            if mode in (3, 4):
                MID = D // 4096
                T = D - MID * 4096
            P = MID + (1 if T > 0 else 0)
            if mode == 3:
                pad = -float("inf")
            elif mode == 4:
                pad = float("inf")
            else:
                pad = 0.0
            stride = triton.next_power_of_2(P)
            if stride > _MID_BLOCK:
                stride = triton.cdiv(P, _MID_BLOCK) * _MID_BLOCK
            mid = torch.full((N * stride,), pad, device=x1.device, dtype=torch.float32)
            chunk_block = 4096 if mode in (3, 4) else _BLOCK_D
            _pd_chunk_kernel[(N, MID)](
                x1,
                x2,
                mid,
                D,
                eps,
                p_scalar,
                stride,
                MID,
                MODE=mode,
                BLOCK=chunk_block,
            )
            if T > 0:
                PS, PNP, NCSC = _piece_args(T)
                _pd_tail_kernel[(N,)](
                    x1,
                    x2,
                    mid,
                    N,
                    D,
                    T,
                    eps,
                    p_scalar,
                    MID,
                    stride,
                    MODE=mode,
                    S=PS,
                    NP=PNP,
                    NSCALAR=NCSC,
                )
            cur_mid, cur_stride, cur_n = mid, stride, P
            while triton.next_power_of_2(cur_n) > _MID_BLOCK:
                g = triton.cdiv(cur_n, _MID_BLOCK)
                nstride = triton.next_power_of_2(g)
                cur_out = torch.full(
                    (N * nstride,), pad, device=x1.device, dtype=torch.float32
                )
                _pd_mid_reduce_kernel[(N, g)](
                    cur_mid,
                    cur_out,
                    MID=cur_n,
                    STRIDE_IN=cur_stride,
                    STRIDE_OUT=nstride,
                    MODE=mode,
                    BLOCK=_MID_BLOCK,
                )
                cur_mid, cur_stride, cur_n = cur_out, nstride, g
            _pd_final_kernel[(N,)](
                cur_mid,
                out,
                p_scalar,
                cur_stride,
                MODE=mode,
                BLOCK_MID=triton.next_power_of_2(cur_n),
            )

    return out
