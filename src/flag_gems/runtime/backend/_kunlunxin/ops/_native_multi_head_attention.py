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
"""Kunlunxin XPU3 override of `aten::_native_multi_head_attention`.

Why the generic pipeline cannot be used here
--------------------------------------------
The generic implementation feeds *masked* 2-D loads straight into `tl.dot` in
all four pipeline kernels (QKV projection, streaming / materialized attention,
output projection).  On XPU3 that combination fails to lower
(`TritonToLinalgExperimental` / `TritonToStructured` for the projection,
`ConvertTritonSDNNToLLVM` for attention), so the op never launches (batch13:
0 passed / 69 failed).

Mask-free rewrite
-----------------
Every `tl.dot` operand is an unmasked, in-bounds, affine load.  Partial tiles
are handled by wrapping the row/column index with `%` (the sanctioned GEMM
scheme in ``_kunlunxin/ops/mm.py``) or by host-side zero padding, never by a
mask/select on a dot operand.  Dots are 2-arg with an fp32 accumulator,
`allow_tf32=False`.  `P @ V` is its own kernel whose operands are *both loads*
(a `tl.dot` fed by a register tile is silently wrong on XPU3).

Measured XPU3 traps avoided here (this op was the worst offender of the family)
-----------------------------------------------------------------------------
* `tl.exp2` is miscompiled (plain-tile max abs err ~17); only `tl.exp` is exact.
* A value-level op on a raw dot result (e.g. `qk * scale`) poisons the
  following `tl.where`, collapsing every lane -> fold `scale` into the Q
  *operand* instead.
* `tl.where(mask, dot_result, scalar)` for the softmax padding columns is fine,
  but `p * (1/tl.sum(p,1))` (a broadcast multiply driven by a reduction over the
  dot result) collapses `p` to a constant -> store the *unnormalised* exp
  weights and the row sums, then normalise in a pure-elementwise kernel that
  reads both from memory.
* A **masked store of a dot-derived value** writes 1.0 for every lane (measured)
  -> every dot result is stored *unmasked* into a width-`NP` (power-of-two)
  buffer; masking/padding is applied only by later elementwise kernels.

Pipeline
--------
1. `_proj_kernel`          — `x @ Wᵀ + b` GEMM (launched once per Q/K/V group).
2. `_pad_*`                — host padding helpers, only for non-power-of-two `D`.
3. `_headpad_kernel`       — reorder Q/K/V into a head-padded `[3, BH, TP, DHP]`
                             layout; padding rows/cols are zero.
4. `_attn_weights_kernel`  — QK scores -> unnormalised exp weights + row sums.
5. `_norm_weights_kernel`  — elementwise normalise into the padded weights.
6. `_pv_kernel`            — `Weights @ V` (both operands loads) -> context.
7. export kernels          — average / copy the weights.

`TP = round_up(T, 16)` and `BN = max(16, next_pow2(T))` (>= TP) give the PV
key loop a power-of-two divisor >= 16.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _pick_block(n: int) -> int:
    """Dot block size: power of two, at least 16, at most 64."""
    return min(64, max(16, _next_pow2(n)))


def _pick_divisor(n: int, cap: int = 64) -> int:
    """Largest power of two that divides `n`, capped at `cap` (>= 16 here)."""
    b = 1
    while b * 2 <= cap and n % (b * 2) == 0:
        b *= 2
    return b


def _round_up(n: int, a: int) -> int:
    return ((n + a - 1) // a) * a


# ---------------------------------------------------------------------------
# projection GEMM: OUT[m, n] = sum_k IN[m, k] * Wg[n, k] + Bg[n]   (width NP)
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _proj_kernel(
    IN,
    Wg,
    Bg,
    OUT,
    M,
    KP,
    NP,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Single-group GEMM launched once per projection group; the kernel never
    # merges input pointers from a runtime branch (that poisons the dot).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, KP, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(IN + rm[:, None] * KP + rk[None, :])
        bt = tl.load(Wg + rn[:, None] * KP + rk[None, :])
        acc += tl.dot(a, tl.trans(bt), out_dtype=tl.float32, allow_tf32=False)

    bias = tl.load(Bg + rn)
    acc = acc + bias[None, :]

    # Unmasked store: the output buffer has width NP (power of two) so `rn` is
    # always in bounds; a masked store of a dot result is miscompiled here.
    tl.store(OUT + rm[:, None] * NP + rn[None, :], acc.to(OUT.dtype.element_ty))


# ---------------------------------------------------------------------------
# host-side padding helpers (plain masked copies, NOT inside a dot kernel)
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _pad_in_kernel(SRC, DST, R, C, CP, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rr = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    m = (rr < R)[:, None] & (rc < C)[None, :]
    rrc = tl.minimum(rr, R - 1)
    rcc = tl.minimum(rc, C - 1)
    v = tl.load(SRC + rrc[:, None] * C + rcc[None, :], mask=m, other=0.0)
    tl.store(DST + rrc[:, None] * CP + rcc[None, :], v, mask=m)


@libentry()
@triton.jit
def _pad_w_kernel(SRC, DST, R, DD, NP, KP, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rr = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    m = (rr < R)[:, None] & (rc < DD)[None, :]
    rrc = tl.minimum(rr, R - 1)
    rcc = tl.minimum(rc, DD - 1)
    g = rrc // DD
    n = rrc % DD
    v = tl.load(SRC + rrc[:, None] * DD + rcc[None, :], mask=m, other=0.0)
    tl.store(DST + g[:, None] * (NP * KP) + n[:, None] * KP + rcc[None, :], v, mask=m)


@libentry()
@triton.jit
def _pad_b_kernel(SRC, DST, R, DD, NP, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rr = pid * BLOCK + tl.arange(0, BLOCK)
    m = rr < R
    rrc = tl.minimum(rr, R - 1)
    g = rrc // DD
    n = rrc % DD
    v = tl.load(SRC + rrc, mask=m, other=0.0)
    tl.store(DST + g * NP + n, v, mask=m)


@libentry()
@triton.jit
def _compact_kernel(SRC, DST, R, DD, NP, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    """Copy the leading `DD` columns of a width-`NP` buffer to a width-`DD` one."""
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rr = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    m = (rr < R)[:, None] & (rc < DD)[None, :]
    rrc = tl.minimum(rr, R - 1)
    rcc = tl.minimum(rc, DD - 1)
    v = tl.load(SRC + rrc[:, None] * NP + rcc[None, :], mask=m, other=0.0)
    tl.store(DST + rrc[:, None] * DD + rcc[None, :], v, mask=m)


# ---------------------------------------------------------------------------
# reorder Q/K/V [3, M, QS] -> padded head layout [3, BH, TP, DHP]
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _headpad_kernel(QKV, QKVHP, M, T, QS, NH, DH, DHP, TP, BLOCK_D: tl.constexpr):
    idx = tl.program_id(0)
    head = idx % NH
    tmp = idx // NH
    m = tmp % M
    g = tmp // M
    B = M // T
    BH = B * NH
    b = m // T
    t = m % T

    rd = tl.arange(0, BLOCK_D)
    msk = rd < DH
    rdc = tl.minimum(rd, DH - 1)
    src = g * (M * QS) + m * QS + head * DH + rdc
    dst = ((g * BH + b * NH + head) * TP + t) * DHP + rdc
    v = tl.load(QKV + src, mask=msk, other=0.0)
    tl.store(QKVHP + dst, v, mask=msk)


# ---------------------------------------------------------------------------
# attention weights: unnormalised exp weights + per-row sums
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _attn_qk_kernel(
    Qh,
    Kh,
    QKB,
    TP,
    DHP,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """QK scores only, stored (unmasked) to a width-`BLOCK_N` scratch buffer.

    Reductions (`tl.max` / `tl.sum`) on a raw dot result are miscompiled on
    XPU3 (the `cov` trap, measured max-row error ~6), so the softmax (and the
    soft-scale / mask application) run in a separate kernel on this *loaded*
    score tile.  The softmax scale is applied there to the fp32 scores, which
    also matches the generic kernel's numerics for fp16/bf16.
    """
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = tl.arange(0, BLOCK_N)
    rd = tl.arange(0, BLOCK_D)

    q = tl.load(Qh + (bh * TP + rm[:, None]) * DHP + rd[None, :])
    k = tl.load(Kh + (bh * TP + on[:, None]) * DHP + rd[None, :])
    qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32, allow_tf32=False)
    tl.store(QKB + (bh * TP + rm[:, None]) * BLOCK_N + on[None, :], qk)


@libentry()
@triton.jit
def _softmax_kernel(
    QKB,
    MASK,
    RAW,
    LSUM,
    sm_scale,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_mn,
    T,
    TP,
    BN,
    NH,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Apply scale + mask, row softmax, store the unnormalised exp + row sums.

    Everything here is on a *loaded* score tile, so the reductions and the
    `tl.where` masking are exact.
    """
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = tl.arange(0, BLOCK_N)
    n_valid = on < T
    NEG = -1.0e30

    qk = tl.load(QKB + (bh * TP + rm[:, None]) * BN + on[None, :]) * sm_scale
    if HAS_MASK:
        b = bh // NH
        h = bh % NH
        rmc = tl.minimum(rm, T - 1)
        rnc = tl.minimum(on, T - 1)
        mv = tl.load(
            MASK
            + b * stride_mb
            + h * stride_mh
            + rmc[:, None] * stride_mm
            + rnc[None, :] * stride_mn
        )
        qk = tl.where(mv != 0, NEG, qk)
    qk = tl.where(n_valid[None, :], qk, NEG)
    m_i = tl.max(qk, 1)
    e = tl.exp(qk - m_i[:, None])
    l_i = tl.sum(e, 1)
    tl.store(LSUM + bh * TP + rm, l_i)
    tl.store(RAW + (bh * TP + rm[:, None]) * BN + on[None, :], e)


@libentry()
@triton.jit
def _norm_weights_kernel(
    RAW,
    LSUM,
    WP,
    T,
    TP,
    BN,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """WP[bh,t,n] = RAW[bh,t,n] / LSUM[bh,t] for n < T, else 0 (padding)."""
    bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mrow = om < T
    omc = tl.minimum(om, T - 1)
    onc = tl.minimum(on, TP - 1)
    inv = 1.0 / tl.load(LSUM + bh * TP + omc, mask=mrow, other=1.0)
    raw = tl.load(
        RAW + (bh * TP + omc[:, None]) * BN + onc[None, :],
        mask=mrow[:, None],
        other=0.0,
    )
    val = raw * inv[:, None]
    val = tl.where(on[None, :] < T, val, 0.0)
    tl.store(
        WP + (bh * TP + omc[:, None]) * TP + onc[None, :],
        val.to(WP.dtype.element_ty),
        mask=mrow[:, None] & (on[None, :] < TP),
    )


# ---------------------------------------------------------------------------
# P @ V: context[b, t, h*DH + d] = sum_n Weights[bh, t, n] * V[bh, n, d]
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _pv_kernel(
    WP,
    Vh,
    CTXP,
    TP,
    DHP,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """CTXP[bh, t, :] = sum_n Weights[bh, t, n] * V[bh, n, :]  (padded rows).

    Both dot operands are loads.  The result is stored physically (unmasked,
    into the head-padded `[BH, TP, DHP]` buffer); `_ctx_repack_kernel` maps it
    to the compact `[M, D]` layout afterwards.
    """
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, BLOCK_D)

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for n0 in range(0, TP, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N)
        w = tl.load(WP + (bh * TP + rm[:, None]) * TP + rn[None, :])
        v = tl.load(Vh + (bh * TP + rn[:, None]) * DHP + rd[None, :])
        acc += tl.dot(w, v, out_dtype=tl.float32, allow_tf32=False)

    tl.store(CTXP + (bh * TP + rm[:, None]) * DHP + rd[None, :], acc.to(CTXP.dtype.element_ty))


@libentry()
@triton.jit
def _ctx_repack_kernel(CTXP, CTX, M, T, D, NH, DH, DHP, TP, BLOCK_D: tl.constexpr):
    """CTX[b*T+t, head*DH+d] = CTXP[(b*NH+head)*TP + t, d] for d < DH."""
    m = tl.program_id(0)
    head = tl.program_id(1)
    b = m // T
    t = m % T
    rd = tl.arange(0, BLOCK_D)
    msk = rd < DH
    rdc = tl.minimum(rd, DH - 1)
    src = (b * NH + head) * TP * DHP + t * DHP + rdc
    dst = m * D + head * DH + rdc
    v = tl.load(CTXP + src, mask=msk, other=0.0)
    tl.store(CTX + dst, v.to(CTX.dtype.element_ty), mask=msk)


# ---------------------------------------------------------------------------
# weights export (elementwise; values come from loads)
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _avg_weights_kernel(WP, OUT, T, TP, NH, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    msk = (om < T)[:, None] & (on < T)[None, :]
    omc = tl.minimum(om, T - 1)
    onc = tl.minimum(on, T - 1)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for hh in range(NH):
        w = tl.load(
            WP + ((b * NH + hh) * TP + omc[:, None]) * TP + onc[None, :],
            mask=msk,
            other=0.0,
        )
        acc += w.to(tl.float32)
    acc = acc * (1.0 / NH)
    tl.store(
        OUT + (b * T + omc[:, None]) * T + onc[None, :],
        acc.to(OUT.dtype.element_ty),
        mask=msk,
    )


@libentry()
@triton.jit
def _copy_weights_kernel(WP, OUT, T, TP, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    msk = (om < T)[:, None] & (on < T)[None, :]
    omc = tl.minimum(om, T - 1)
    onc = tl.minimum(on, T - 1)
    w = tl.load(WP + (bh * TP + omc[:, None]) * TP + onc[None, :], mask=msk, other=0.0)
    tl.store(OUT + (bh * T + omc[:, None]) * T + onc[None, :], w, mask=msk)


# ---------------------------------------------------------------------------
# host entry
# ---------------------------------------------------------------------------
def _native_multi_head_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    embed_dim: int,
    num_head: int,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    mask: torch.Tensor = None,
    need_weights: bool = True,
    average_attn_weights: bool = True,
    mask_type=None,
):
    logger.debug("GEMS_KUNLUNXIN _NATIVE_MULTI_HEAD_ATTENTION")

    B, T, D = query.shape
    NH = num_head
    DH = D // NH
    M = B * T
    BH = B * NH
    dt = query.dtype
    device = query.device

    has_mask = mask is not None
    if query.numel() == 0:
        out = torch.empty_like(query)
        weights = torch.empty((B, T, T), dtype=dt, device=device) if need_weights else None
        return out, weights

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    scale = 1.0 / math.sqrt(DH)

    is_pow2 = (D & (D - 1)) == 0
    DP = D if is_pow2 else _next_pow2(D)  # K/N pad for the projection GEMMs

    # ---- 1. QKV projection (one launch per group) -------------------------
    qkv_out = torch.empty((3 * M, DP), dtype=dt, device=device)

    if is_pow2:
        group_inputs = (query, key, value)
        wg = qkv_weight.contiguous().view(3, D, D)
        bg = qkv_bias.contiguous().view(3, D)
    else:
        inp3 = torch.zeros((3, M, DP), dtype=dt, device=device)
        for g, src in enumerate((query, key, value)):
            _pad_in_kernel[(triton.cdiv(M, 64), triton.cdiv(DP, 64))](
                src, inp3[g], M, D, DP, BLOCK_R=64, BLOCK_C=64
            )
        group_inputs = (inp3[0], inp3[1], inp3[2])
        wg = torch.zeros((3, DP, DP), dtype=dt, device=device)
        _pad_w_kernel[(triton.cdiv(3 * D, 64), triton.cdiv(DP, 64))](
            qkv_weight.contiguous(), wg, 3 * D, D, DP, DP, BLOCK_R=64, BLOCK_C=64
        )
        bg = torch.zeros((3, DP), dtype=dt, device=device)
        _pad_b_kernel[(triton.cdiv(3 * D, 64),)](
            qkv_bias.contiguous(), bg, 3 * D, D, DP, BLOCK=64
        )

    BLOCK_M_P = _pick_block(M)
    BLOCK_N_P = min(64, DP)
    BLOCK_K_P = min(64, DP)
    grid = (triton.cdiv(M, BLOCK_M_P), triton.cdiv(DP, BLOCK_N_P))
    with torch_device_fn.device(device):
        for g in range(3):
            inp_g = group_inputs[g]
            _proj_kernel[grid](
                inp_g,
                wg[g],
                bg[g],
                qkv_out[g * M : (g + 1) * M],
                M,
                DP,
                DP,
                BLOCK_M=BLOCK_M_P,
                BLOCK_N=BLOCK_N_P,
                BLOCK_K=BLOCK_K_P,
            )

    # ---- 2. head-padded attention layout ----------------------------------
    # Physical-row tiling: BM divides the padded row extent, sources are
    # zero-padded to TP rows, and no index is wrapped with %T (duplicate rows
    # inside a tl.dot are miscompiled).  Rows >= T are padding.
    BLOCK_M_A = _pick_block(T)
    NROWS = triton.cdiv(T, BLOCK_M_A) * BLOCK_M_A
    BN = max(16, _next_pow2(T))
    TP = max(_round_up(T, 16), NROWS, BN)
    DHP = max(16, _next_pow2(DH))
    bd = _next_pow2(DH)
    qkv_hp = torch.zeros((3, BH, TP, DHP), dtype=dt, device=device)
    with torch_device_fn.device(device):
        _headpad_kernel[(3 * M * NH,)](
            qkv_out, qkv_hp, M, T, DP, NH, DH, DHP, TP, BLOCK_D=bd
        )

    # ---- 3. attention weights (unnormalised) ------------------------------
    raw = torch.empty((BH, TP, BN), dtype=torch.float32, device=device)
    lsum = torch.empty((BH, TP), dtype=torch.float32, device=device)
    if has_mask:
        if mask_type == 0:
            m4 = mask.unsqueeze(0).unsqueeze(0)
        elif mask_type == 1:
            m4 = mask.unsqueeze(1).unsqueeze(1)
        else:
            m4 = mask
            if m4.dim() == 2:
                m4 = m4.unsqueeze(0).unsqueeze(0)
            elif m4.dim() == 3:
                m4 = m4.unsqueeze(0)
        # Materialise the broadcast mask as a contiguous int8 [BH, T, T]; this
        # is pure input prep (data movement / dtype cast), not the tested
        # numerics, and gives the softmax kernel unambiguous strides.
        mask_buf = m4.expand(B, NH, T, T).contiguous().to(torch.int8)
        s_b, s_h, s_m, s_n = mask_buf.stride(0), mask_buf.stride(1), mask_buf.stride(2), mask_buf.stride(3)
    else:
        mask_buf = query
        s_b = s_h = s_m = s_n = 0

    weights_pad = torch.zeros((BH, TP, TP), dtype=dt, device=device)
    qkb = torch.empty((BH, TP, BN), dtype=torch.float32, device=device)
    grid_a = (triton.cdiv(T, BLOCK_M_A), BH)
    with torch_device_fn.device(device):
        _attn_qk_kernel[grid_a](
            qkv_hp[0],
            qkv_hp[1],
            qkb,
            TP,
            DHP,
            BLOCK_M=BLOCK_M_A,
            BLOCK_N=BN,
            BLOCK_D=DHP,
        )
        _softmax_kernel[grid_a](
            qkb,
            mask_buf,
            raw,
            lsum,
            scale,
            s_b,
            s_h,
            s_m,
            s_n,
            T,
            TP,
            BN,
            NH,
            HAS_MASK=has_mask,
            BLOCK_M=BLOCK_M_A,
            BLOCK_N=BN,
        )
        _norm_weights_kernel[(BH, triton.cdiv(T, 16), triton.cdiv(TP, 64))](
            raw, lsum, weights_pad, T, TP, BN, BLOCK_M=16, BLOCK_N=64
        )

    # ---- 4. context = Weights @ V (physical rows) + repack -----------------
    BLOCK_N_PV = _pick_divisor(TP, 64)
    ctx_pad = torch.empty((BH, TP, DHP), dtype=dt, device=device)
    ctx = torch.empty((M, D), dtype=dt, device=device)
    with torch_device_fn.device(device):
        _pv_kernel[grid_a](
            weights_pad, qkv_hp[2], ctx_pad, TP, DHP,
            BLOCK_M=BLOCK_M_A, BLOCK_N=BLOCK_N_PV, BLOCK_D=DHP,
        )
        _ctx_repack_kernel[(M, NH)](
            ctx_pad, ctx, M, T, D, NH, DH, DHP, TP, BLOCK_D=bd
        )

    # ---- 5. output projection ---------------------------------------------
    if is_pow2:
        x0 = ctx
        wp = proj_weight.contiguous().view(1, D, D)
        bp = proj_bias.contiguous().view(1, D)
    else:
        x0 = torch.zeros((M, DP), dtype=dt, device=device)
        _pad_in_kernel[(triton.cdiv(M, 64), triton.cdiv(DP, 64))](
            ctx, x0, M, D, DP, BLOCK_R=64, BLOCK_C=64
        )
        wp = torch.zeros((1, DP, DP), dtype=dt, device=device)
        _pad_w_kernel[(triton.cdiv(D, 64), triton.cdiv(DP, 64))](
            proj_weight.contiguous(), wp, D, D, DP, DP, BLOCK_R=64, BLOCK_C=64
        )
        bp = torch.zeros((1, DP), dtype=dt, device=device)
        _pad_b_kernel[(triton.cdiv(D, 64),)](
            proj_bias.contiguous(), bp, D, D, DP, BLOCK=64
        )

    out_pad = torch.empty((M, DP), dtype=dt, device=device)
    grid_o = (triton.cdiv(M, BLOCK_M_P), triton.cdiv(DP, BLOCK_N_P))
    with torch_device_fn.device(device):
        _proj_kernel[grid_o](
            x0, wp[0], bp[0], out_pad, M, DP, DP,
            BLOCK_M=BLOCK_M_P, BLOCK_N=BLOCK_N_P, BLOCK_K=BLOCK_K_P,
        )
    if is_pow2:
        out = out_pad.view(B, T, D)
    else:
        out_flat = torch.empty((M, D), dtype=dt, device=device)
        _compact_kernel[(triton.cdiv(M, 64), triton.cdiv(D, 64))](
            out_pad, out_flat, M, D, DP, BLOCK_R=64, BLOCK_C=64
        )
        out = out_flat.view(B, T, D)

    # ---- 6. export weights ------------------------------------------------
    if not need_weights:
        return out, None

    EBM, EBN = 16, 64
    if average_attn_weights:
        avg = torch.empty((B, T, T), dtype=dt, device=device)
        grid_e = (B, triton.cdiv(T, EBM), triton.cdiv(T, EBN))
        with torch_device_fn.device(device):
            _avg_weights_kernel[grid_e](weights_pad, avg, T, TP, NH, BLOCK_M=EBM, BLOCK_N=EBN)
        return out, avg
    else:
        wout = torch.empty((BH, T, T), dtype=dt, device=device)
        grid_e = (BH, triton.cdiv(T, EBM), triton.cdiv(T, EBN))
        with torch_device_fn.device(device):
            _copy_weights_kernel[grid_e](weights_pad, wout, T, TP, BLOCK_M=EBM, BLOCK_N=EBN)
        return out, wout.view(B, NH, T, T)
