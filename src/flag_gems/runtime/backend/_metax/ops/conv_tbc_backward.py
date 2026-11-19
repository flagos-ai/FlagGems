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

"""conv_tbc_backward (Triton).

grad_output = self : (T_out, B, C_out)
input              : (T, B, C_in)
weight             : (K, C_in, C_out)
bias               : (C_out,)
pad                : int

grad_input[s,b,c]  = sum_k sum_o g[s+pad-k, b, o] * w[k, c, o]     (0 <= s+pad-k < T_out)
grad_weight[k,c,o] = sum_{t,b} g[t,b,o] * x[t-pad+k, b, c]         (0 <= t-pad+k < T)
grad_bias[o]       = sum_{t,b} g[t,b,o]

Two launches per call:
  * _ctbc_bwd_weight reduces over the (t,b) rows; when the (K, c-block, o-block)
    grid leaves the device partly idle the row range is split into S chunks that
    write fp32 partials (plus per-chunk grad_bias partials).
  * _ctbc_bwd_input_reduce computes the grad_input tile for each of its first
    A_N programs and uses the remaining programs to reduce the fp32 partials
    written by the first kernel into grad_weight/grad_bias, so the reduction
    costs no additional launch.
"""

import torch
import triton
import triton.language as tl


def _block_for(n, lo=16, hi=64):
    v = lo
    while v < n and v < hi:
        v *= 2
    return v


# One reduction-loop iteration moves ~2*BLOCK_M*BLOCK_C elements of halves; the
# measured reduce-kernel bandwidth and per-iteration cost make the partial-buffer
# round trip of the split reduction comparable to this many bytes per iteration.
_BYTES_PER_ITER_EQ = 1.3e6
_LAUNCH_ITER_EQ = 2.5


def _choose_split(n_grid, n_iter, out_elems, sm, max_s=16):
    """Pick the (t,b) split factor S minimizing a chain-vs-traffic cost model."""
    best_s, best_cost = 1, float(n_iter)
    for s in range(2, min(max_s, n_iter) + 1):
        chain = -(-n_grid * s // sm)
        per_prog = -(-n_iter // s)
        traffic = (2.0 * s * out_elems * 4) / _BYTES_PER_ITER_EQ
        cost = max(chain, per_prog) + traffic + _LAUNCH_ITER_EQ
        if cost < best_cost - 1e-9:
            best_s, best_cost = s, cost
    return best_s


@triton.jit
def _ctbc_bwd_weight(
    x_ptr,
    g_ptr,
    gw_ptr,
    gb_ptr,
    p_ptr,
    pb_ptr,
    B,
    T,
    T_OUT,
    C_IN,
    C_OUT,
    PAD,
    M_OUT,
    S,
    s_xt,
    s_xb,
    s_xc,
    s_gt,
    s_gb,
    s_go,
    s_wk,
    s_wc,
    s_wo,
    s_ps,
    s_pk,
    s_pc,
    s_po,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    USE_DOT: tl.constexpr,
    IEEE: tl.constexpr,
    PARTIAL: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_o = tl.program_id(2)

    if PARTIAL:
        k = pid0 // S
        s = pid0 - k * S
    else:
        k = pid0
        s = 0

    offs_c = pid_c * BLOCK_CI + tl.arange(0, BLOCK_CI)
    offs_o = pid_o * BLOCK_CO + tl.arange(0, BLOCK_CO)
    c_mask = offs_c < C_IN
    o_mask = offs_o < C_OUT

    if USE_DOT:
        acc = tl.zeros((BLOCK_CI, BLOCK_CO), dtype=tl.float32)
        bias_acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_CI, BLOCK_CO), dtype=gw_ptr.dtype.element_ty)
        bias_acc = tl.zeros((BLOCK_CO,), dtype=tl.float64)

    do_bias = (k == 0) & (pid_c == 0)

    if PARTIAL:
        chunk = (M_OUT + S - 1) // S
        m_lo = s * chunk
        m_hi = tl.minimum(m_lo + chunk, M_OUT)
    else:
        m_lo = 0
        m_hi = M_OUT

    for m0 in range(m_lo, m_hi, BLOCK_M):
        offs = m0 + tl.arange(0, BLOCK_M)
        m_mask = offs < m_hi
        t = offs // B
        b = offs - t * B
        tx = t - PAD + k
        x_mask = m_mask & (tx >= 0) & (tx < T)
        t_safe = tl.where(m_mask, t, 0)
        tx_safe = tl.where(x_mask, tx, 0)

        xt = tl.load(
            x_ptr
            + tx_safe[:, None] * s_xt
            + b[:, None] * s_xb
            + offs_c[None, :] * s_xc,
            mask=x_mask[:, None] & c_mask[None, :],
            other=0.0,
        )
        gt = tl.load(
            g_ptr + t_safe[:, None] * s_gt + b[:, None] * s_gb + offs_o[None, :] * s_go,
            mask=m_mask[:, None] & o_mask[None, :],
            other=0.0,
        )

        if USE_DOT:
            if IEEE:
                acc = tl.dot(tl.trans(xt), gt, acc, input_precision="ieee")
            else:
                acc = tl.dot(tl.trans(xt), gt, acc)
        else:
            acc += tl.sum(xt[:, :, None] * gt[:, None, :], axis=0)

        if do_bias:
            bias_acc += tl.sum(gt.to(bias_acc.dtype), axis=0)

    if PARTIAL:
        p_ptrs = (
            p_ptr
            + s * s_ps
            + k * s_pk
            + offs_c[:, None] * s_pc
            + offs_o[None, :] * s_po
        )
        tl.store(p_ptrs, acc, mask=c_mask[:, None] & o_mask[None, :])
    else:
        w_ptrs = gw_ptr + k * s_wk + offs_c[:, None] * s_wc + offs_o[None, :] * s_wo
        tl.store(
            w_ptrs,
            acc.to(gw_ptr.dtype.element_ty),
            mask=c_mask[:, None] & o_mask[None, :],
        )

    if do_bias:
        if PARTIAL:
            tl.store(pb_ptr + s * C_OUT + offs_o, bias_acc, mask=o_mask)
        else:
            tl.store(gb_ptr + offs_o, bias_acc.to(gb_ptr.dtype.element_ty), mask=o_mask)


@triton.jit
def _ctbc_bwd_input_reduce(
    g_ptr,
    w_ptr,
    gi_ptr,
    p_ptr,
    pb_ptr,
    gw_ptr,
    gb_ptr,
    B,
    T,
    T_OUT,
    C_IN,
    C_OUT,
    PAD,
    S,
    A_N,
    N_CI,
    N_CO,
    s_gt,
    s_gb,
    s_go,
    s_wk,
    s_wc,
    s_wo,
    s_it,
    s_ib,
    s_ic,
    s_ps,
    s_pk,
    s_pc,
    s_po,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    USE_DOT: tl.constexpr,
    IEEE: tl.constexpr,
    DO_REDUCE: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid < A_N:
        # ---- grad_input tile ----
        pid_m = pid // N_CI
        pid_c = pid - pid_m * N_CI

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_c = pid_c * BLOCK_CI + tl.arange(0, BLOCK_CI)
        offs_o = tl.arange(0, BLOCK_CO)

        m_mask = offs_m < T * B
        c_mask = offs_c < C_IN
        o_mask = offs_o < C_OUT

        t = offs_m // B
        b = offs_m - t * B
        t_safe = tl.where(m_mask, t, 0)

        if USE_DOT:
            acc_i = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
        else:
            acc_i = tl.zeros((BLOCK_M, BLOCK_CI), dtype=gi_ptr.dtype.element_ty)

        for k in tl.static_range(K):
            tp = t + PAD - k
            row_mask = m_mask & (tp >= 0) & (tp < T_OUT)
            tp_safe = tl.where(row_mask, tp, 0)
            g_ptrs = (
                g_ptr
                + tp_safe[:, None] * s_gt
                + b[:, None] * s_gb
                + offs_o[None, :] * s_go
            )
            gt = tl.load(g_ptrs, mask=row_mask[:, None] & o_mask[None, :], other=0.0)
            w_ptrs = w_ptr + k * s_wk + offs_c[:, None] * s_wc + offs_o[None, :] * s_wo
            wt = tl.load(w_ptrs, mask=c_mask[:, None] & o_mask[None, :], other=0.0)
            if USE_DOT:
                if IEEE:
                    acc_i = tl.dot(gt, tl.trans(wt), acc_i, input_precision="ieee")
                else:
                    acc_i = tl.dot(gt, tl.trans(wt), acc_i)
            else:
                acc_i += tl.sum(gt[:, None, :] * wt[None, :, :], axis=2)

        out_ptrs = (
            gi_ptr + t_safe[:, None] * s_it + b[:, None] * s_ib + offs_c[None, :] * s_ic
        )
        tl.store(
            out_ptrs,
            acc_i.to(gi_ptr.dtype.element_ty),
            mask=m_mask[:, None] & c_mask[None, :],
        )
    else:
        if DO_REDUCE:
            # ---- grad_weight / grad_bias partial reduction ----
            q = pid - A_N
            kk = q // (N_CI * N_CO)
            rem = q - kk * (N_CI * N_CO)
            pid_c = rem // N_CO
            pid_o = rem - pid_c * N_CO

            offs_c = pid_c * BLOCK_CI + tl.arange(0, BLOCK_CI)
            offs_o = pid_o * BLOCK_CO + tl.arange(0, BLOCK_CO)
            c_mask = offs_c < C_IN
            o_mask = offs_o < C_OUT

            acc_w = tl.zeros((BLOCK_CI, BLOCK_CO), dtype=tl.float32)
            base = kk * s_pk + offs_c[:, None] * s_pc + offs_o[None, :] * s_po
            wmask = c_mask[:, None] & o_mask[None, :]
            for s in range(S):
                acc_w += tl.load(p_ptr + s * s_ps + base, mask=wmask, other=0.0)

            w_ptrs = (
                gw_ptr + kk * s_wk + offs_c[:, None] * s_wc + offs_o[None, :] * s_wo
            )
            tl.store(w_ptrs, acc_w.to(gw_ptr.dtype.element_ty), mask=wmask)

            if (kk == 0) & (pid_c == 0):
                bacc = tl.zeros((BLOCK_CO,), dtype=tl.float32)
                for s in range(S):
                    bacc += tl.load(pb_ptr + s * C_OUT + offs_o, mask=o_mask, other=0.0)
                tl.store(gb_ptr + offs_o, bacc.to(gb_ptr.dtype.element_ty), mask=o_mask)


def conv_tbc_backward(self, input, weight, bias, pad):
    grad_out = self
    x = input
    w = weight
    bs = bias
    pad = int(pad)

    if grad_out.dim() != 3 or x.dim() != 3 or w.dim() != 3 or bs.dim() != 1:
        raise RuntimeError(
            "conv_tbc_backward: grad_output, input and weight must be 3D and bias must be 1D"
        )

    T, Bn, C_in = x.shape
    K, C_in_w, C_out = w.shape

    if C_in_w != C_in:
        raise RuntimeError("conv_tbc_backward: weight.size(1) must equal input.size(2)")
    if bs.shape[0] != C_out:
        raise RuntimeError("conv_tbc_backward: bias.size(0) must equal weight.size(2)")
    if pad < 0:
        raise RuntimeError("conv_tbc_backward: pad must be non-negative")

    T_out = T + 2 * pad - K + 1
    if T_out < 1 or grad_out.shape != (T_out, Bn, C_out):
        raise RuntimeError(
            "conv_tbc_backward: grad_output must have shape (T + 2*pad - K + 1, B, C_out)"
        )

    gi = torch.empty((T, Bn, C_in), dtype=x.dtype, device=x.device)
    gw = torch.empty((K, C_in, C_out), dtype=w.dtype, device=w.device)
    gb = torch.empty((C_out,), dtype=bs.dtype, device=bs.device)

    if T == 0 or Bn == 0 or C_in == 0 or C_out == 0:
        return gi, gw, gb

    f64 = x.dtype == torch.float64

    bci = _block_for(C_in)
    bco = _block_for(C_out)

    if f64:
        use_dot = False
        ieee = False
        bm_a = 16
        bm_b = 16
    else:
        use_dot = True
        ieee = x.dtype == torch.float32
        bm_a = 64
        # The reduction loop pays a barrier-serialised staging round trip per
        # iteration, so a wider row tile halves the number of barrier pairs per
        # row.  The emulated IEEE fp32 dot does not scale with that tile, so the
        # fp32 path keeps 64 rows.
        bm_b = 64 if ieee else 128

    M_in = T * Bn
    M_out = T_out * Bn

    n_ci = triton.cdiv(C_in, bci)
    n_co = triton.cdiv(C_out, bco)
    n_grid_b = K * n_ci * n_co

    # Split the (t,b) reduction across S chunks when the natural grid leaves most
    # of the device idle, or when one program's chain is much longer than the
    # per-SM work implied by the grid.  The SM count is queried at runtime
    # (portable to C500); the cost model trades chain length per program against
    # fp32 partial-buffer traffic and the extra reduction work.
    S = 1
    if not f64 and M_out > bm_b:
        S = _choose_split(
            n_grid_b,
            triton.cdiv(M_out, bm_b),
            K * C_in * C_out,
            torch.cuda.get_device_properties(x.device).multi_processor_count,
        )

    if S == 1:
        _ctbc_bwd_weight[(K, n_ci, n_co)](
            x,
            grad_out,
            gw,
            gb,
            gw,
            gb,
            Bn,
            T,
            T_out,
            C_in,
            C_out,
            pad,
            M_out,
            1,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            gw.stride(0),
            gw.stride(1),
            gw.stride(2),
            0,
            0,
            0,
            0,
            BLOCK_M=bm_b,
            BLOCK_CI=bci,
            BLOCK_CO=bco,
            USE_DOT=use_dot,
            IEEE=ieee,
            PARTIAL=False,
            num_warps=4,
            num_stages=2,
        )
        part = gw
        partb = gb
        ps = (0, 0, 0, 0)
    else:
        part = torch.empty((S, K, C_in, C_out), dtype=torch.float32, device=x.device)
        partb = torch.empty((S, C_out), dtype=torch.float32, device=x.device)
        _ctbc_bwd_weight[(K * S, n_ci, n_co)](
            x,
            grad_out,
            gw,
            gb,
            part,
            partb,
            Bn,
            T,
            T_out,
            C_in,
            C_out,
            pad,
            M_out,
            S,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            gw.stride(0),
            gw.stride(1),
            gw.stride(2),
            part.stride(0),
            part.stride(1),
            part.stride(2),
            part.stride(3),
            BLOCK_M=bm_b,
            BLOCK_CI=bci,
            BLOCK_CO=bco,
            USE_DOT=use_dot,
            IEEE=ieee,
            PARTIAL=True,
            num_warps=4,
            num_stages=2,
        )
        ps = (part.stride(0), part.stride(1), part.stride(2), part.stride(3))

    a_count = triton.cdiv(M_in, bm_a) * n_ci
    r_count = n_grid_b if S > 1 else 0
    _ctbc_bwd_input_reduce[(a_count + r_count,)](
        grad_out,
        w,
        gi,
        part,
        partb,
        gw,
        gb,
        Bn,
        T,
        T_out,
        C_in,
        C_out,
        pad,
        S,
        a_count,
        n_ci,
        n_co,
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        gi.stride(0),
        gi.stride(1),
        gi.stride(2),
        ps[0],
        ps[1],
        ps[2],
        ps[3],
        K=K,
        BLOCK_M=bm_a,
        BLOCK_CI=bci,
        BLOCK_CO=bco,
        USE_DOT=use_dot,
        IEEE=ieee,
        DO_REDUCE=S > 1,
        num_warps=4,
        num_stages=2,
    )

    return gi, gw, gb
