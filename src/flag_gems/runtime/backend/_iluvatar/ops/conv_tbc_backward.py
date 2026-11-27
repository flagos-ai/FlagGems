import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _cdiv(a, b):
    return -(-a // b)


@triton.jit
def _ctb_kernel(
    DO,
    X,
    W,
    GI,
    GW,
    GB,
    T,
    B,
    CIN,
    COUT,
    OLEN,
    K,
    PAD,
    sdo_t,
    sdo_b,
    sdo_c,
    sx_t,
    sx_b,
    sx_ci,
    sw_k,
    sw_ci,
    sw_co,
    N_GI,
    NT_GI,
    M_GW,
    N_GW,
    BLOCK_T: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_TB: tl.constexpr,
    BLOCK_CIW: tl.constexpr,
    BLOCK_COW: tl.constexpr,
    IEEE: tl.constexpr,
):
    """One launch for conv_tbc_backward. Programs with pid < N_GI compute a
    grad_input tile, the rest compute a grad_weight tile; the (k == 0, ci-block 0)
    grad_weight programs also reduce grad_bias.

    grad_input[s, b, ci] = sum_k sum_co dO[s + pad - k, b, co] * W[k, ci, co]
    grad_weight[k, ci, co] = sum_{t,b} X[t + k - pad, b, ci] * dO[t, b, co]
    grad_bias[co] = sum_{t,b} dO[t, b, co]

    GI and GW are freshly allocated and contiguous, so their strides are derived
    from the dimensions instead of being passed in. The two branches use disjoint
    variable names because Triton merges same-named variables across if/else arms
    and requires their types to agree.
    """
    pid = tl.program_id(0)
    if pid < N_GI:
        # ---- grad_input tile ----
        a_nt_b = NT_GI * B
        a_ci_blk = pid // a_nt_b
        a_rem = pid - a_ci_blk * a_nt_b
        a_b = a_rem // NT_GI
        a_it = a_rem - a_b * NT_GI

        a_s = a_it * BLOCK_T + tl.arange(0, BLOCK_T)
        a_ci = a_ci_blk * BLOCK_CI + tl.arange(0, BLOCK_CI)
        a_ms = a_s < T
        a_mci = a_ci < CIN

        a_acc = tl.zeros((BLOCK_T, BLOCK_CI), dtype=tl.float32)
        a_dobase = DO + a_b * sdo_b
        a_wbase = W + a_ci * sw_ci
        for a_k in range(0, K):
            a_r = a_s + PAD - a_k
            a_rok = (a_r >= 0) & (a_r < OLEN)
            a_rc = tl.where(a_rok, a_r, 0)
            a_rowoff = a_rc * sdo_t
            a_dmask = a_rok[:, None]
            a_wkbase = a_wbase + a_k * sw_k
            for a_c0 in range(0, COUT, BLOCK_CO):
                a_co = a_c0 + tl.arange(0, BLOCK_CO)
                a_mco = a_co < COUT
                a_d = tl.load(
                    a_dobase + a_rowoff[:, None] + a_co[None, :] * sdo_c,
                    mask=a_dmask & a_mco[None, :],
                    other=0.0,
                )
                a_w = tl.load(
                    a_wkbase + a_co[:, None] * sw_co,
                    mask=a_mco[:, None] & a_mci[None, :],
                    other=0.0,
                )
                if IEEE:
                    a_acc = tl.dot(a_d, a_w, a_acc, input_precision="ieee")
                else:
                    a_acc = tl.dot(a_d, a_w, a_acc)

        tl.store(
            GI + a_s[:, None] * (B * CIN) + a_b * CIN + a_ci[None, :],
            a_acc.to(GI.dtype.element_ty),
            mask=a_ms[:, None] & a_mci[None, :],
        )
    else:
        # ---- grad_weight tile (+ grad_bias) ----
        g_local = pid - N_GI
        g_mn = M_GW * N_GW
        g_k = g_local // g_mn
        g_rem = g_local - g_k * g_mn
        g_m = g_rem // N_GW
        g_n = g_rem - g_m * N_GW

        g_ci = g_m * BLOCK_CIW + tl.arange(0, BLOCK_CIW)
        g_co = g_n * BLOCK_COW + tl.arange(0, BLOCK_COW)
        g_mci = g_ci < CIN
        g_mco = g_co < COUT
        g_bias_ok = (g_k == 0) & (g_m == 0)

        g_acc = tl.zeros((BLOCK_CIW, BLOCK_COW), dtype=tl.float32)
        g_shift = g_k - PAD
        g_xcol = g_ci[:, None] * sx_ci
        g_dcol = g_co[None, :] * sdo_c
        for g_t0 in range(0, OLEN, BLOCK_TB):
            g_t = g_t0 + tl.arange(0, BLOCK_TB)
            g_mt = g_t < OLEN
            g_r = g_t + g_shift
            g_rok = (g_r >= 0) & (g_r < T) & g_mt
            g_rc = tl.where(g_rok, g_r, 0)
            g_xrow = g_rc[None, :] * sx_t
            g_drow = g_t[:, None] * sdo_t
            g_xmask = g_mci[:, None] & g_rok[None, :]
            g_dmask = g_mt[:, None] & g_mco[None, :]
            for g_b in range(0, B):
                g_x = tl.load(X + g_b * sx_b + g_xcol + g_xrow, mask=g_xmask, other=0.0)
                g_d = tl.load(
                    DO + g_b * sdo_b + g_drow + g_dcol, mask=g_dmask, other=0.0
                )
                if IEEE:
                    g_acc = tl.dot(g_x, g_d, g_acc, input_precision="ieee")
                else:
                    g_acc = tl.dot(g_x, g_d, g_acc)

        tl.store(
            GW + g_k * (CIN * COUT) + g_ci[:, None] * COUT + g_co[None, :],
            g_acc.to(GW.dtype.element_ty),
            mask=g_mci[:, None] & g_mco[None, :],
        )

        if g_bias_ok:
            g_bias = tl.zeros((BLOCK_COW,), dtype=tl.float32)
            for g_bt0 in range(0, OLEN, BLOCK_TB):
                g_bt = g_bt0 + tl.arange(0, BLOCK_TB)
                g_bmt = g_bt < OLEN
                g_bdrow = g_bt[:, None] * sdo_t
                g_bdmask = g_bmt[:, None] & g_mco[None, :]
                for g_bb in range(0, B):
                    g_bd = tl.load(
                        DO + g_bb * sdo_b + g_bdrow + g_dcol,
                        mask=g_bdmask,
                        other=0.0,
                    )
                    g_bias += tl.sum(g_bd.to(tl.float32), 0)
            tl.store(GB + g_co, g_bias.to(GB.dtype.element_ty), mask=g_mco)


def _pow2_at_least(n, lo=16, hi=64):
    v = lo
    while v < n and v < hi:
        v *= 2
    return min(v, hi)


_GW_TILES = [(64, 64), (64, 32), (32, 64), (32, 32), (32, 16), (16, 32), (16, 16)]


def conv_tbc_backward(self, input, weight, bias, pad):
    logger.debug("GEMS_ILUVATAR CONV_TBC_BACKWARD")
    grad_output = self
    pad = int(pad)
    if grad_output.dim() != 3 or input.dim() != 3 or weight.dim() != 3:
        raise RuntimeError(
            "expected 3D grad_output, input and weight for conv_tbc_backward"
        )
    ilen, batch, in_c = input.shape
    kw, _, out_c = weight.shape
    expected_olen = ilen - kw + 1 + 2 * pad
    if (
        weight.shape[1] != in_c
        or grad_output.shape[0] != expected_olen
        or grad_output.shape[1] != batch
        or grad_output.shape[2] != out_c
    ):
        raise RuntimeError(
            "invalid gradient shape for conv_tbc_backward: expected grad_output of "
            "shape ({}, {}, {}) but got {}".format(
                expected_olen, batch, out_c, tuple(grad_output.shape)
            )
        )
    olen = grad_output.shape[0]

    device = input.device
    grad_input = torch.empty((ilen, batch, in_c), dtype=input.dtype, device=device)
    grad_weight = torch.empty((kw, in_c, out_c), dtype=weight.dtype, device=device)
    grad_bias = torch.empty(
        (out_c,),
        dtype=bias.dtype if bias is not None else input.dtype,
        device=device,
    )
    if ilen == 0 or batch == 0 or in_c == 0 or out_c == 0 or olen == 0:
        return grad_input, grad_weight, grad_bias

    bt = _pow2_at_least(ilen, 16, 64)
    bci = _pow2_at_least(in_c, 16, 64)
    bco = _pow2_at_least(out_c, 16, 32)
    nt_gi = _cdiv(ilen, bt)
    n_gi = _cdiv(in_c, bci) * batch * nt_gi

    wci, wco = _GW_TILES[-1]
    for cand in _GW_TILES:
        cp, cq = cand
        if kw * _cdiv(in_c, cp) * _cdiv(out_c, cq) >= 32:
            wci, wco = cand
            break
    btb = _pow2_at_least(olen, 16, 64)
    m_gw = _cdiv(in_c, wci)
    n_gw = _cdiv(out_c, wco)

    # Resident blocks are pinned by the block's shared memory (the larger arm's
    # operand staging), so extra warps per block add warps per CU for free. Only
    # do it when every dot operand dimension is at least 32 lanes wide; the
    # 16-wide tile shapes keep the narrower block.
    is_fp32 = input.dtype == torch.float32
    if min(bt, bci, bco, wci, btb, wco) >= 32:
        # fp32 blocks are shared-memory pinned at half the resident blocks of the
        # half-precision ones, so their warps per CU is still 32 against 64; for
        # them only, double warps per block again to close that gap.
        warps = 16 if is_fp32 else 8
        # Widen the grad_input arm's co strip: it halves the number of co-blocks
        # per k tap (so half the iterations and half the address arithmetic for
        # the arm that holds most programs), and its operand staging then equals
        # rather than exceeds the grad_weight arm's, so residency is unchanged.
        bco = min(64, _pow2_at_least(out_c, 16, 64))
    else:
        warps = 4

    _ctb_kernel[(n_gi + kw * m_gw * n_gw,)](
        grad_output,
        input,
        weight,
        grad_input,
        grad_weight,
        grad_bias,
        ilen,
        batch,
        in_c,
        out_c,
        olen,
        kw,
        pad,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        input.stride(0),
        input.stride(1),
        input.stride(2),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        n_gi,
        nt_gi,
        m_gw,
        n_gw,
        bt,
        bci,
        bco,
        btb,
        wci,
        wco,
        input.dtype == torch.float32,
        num_warps=warps,
        num_stages=2,
    )

    return grad_input, grad_weight, grad_bias
