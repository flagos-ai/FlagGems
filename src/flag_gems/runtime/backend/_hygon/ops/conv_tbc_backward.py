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

"""Triton implementation of aten::conv_tbc_backward.

Semantics pinned on the target with aten::conv_tbc_backward:
  forward (cross-correlation, no kernel flip):
      Y[t,b,m] = bias[m] + sum_{k,c} X[t+k-pad, b, c] * W[k,c,m]
  with T_out = T + 2*pad - K + 1.
  backward:
      dX[t,b,c]   = sum_{k,m} G[t-k+pad, b, m] * W[k,c,m]      (0 <= t-k+pad < T_out)
      dW[k,c,m]   = sum_{t,b} G[t,b,m] * X[t+k-pad, b, c]      (0 <= t+k-pad < T)
      dB[m]       = sum_{t,b} G[t,b,m]
Outputs mirror the reference dtypes (G/X/W may be non-contiguous).

Three host-side dispatch tiers (shape only, never values):
  * heavy (estimated MACs > 1e9): dedicated dX and dW kernels plus the reduce
    kernel. GPU-bound, so the extra launches are hidden and each kernel keeps
    its own register allocation.
  * light direct (T_out*B small): one fused launch whose first grid segment
    computes dX and whose remaining segment computes dW (with its own 32x32
    channel tiling for parallelism and BRN=64) and folds dB in from the
    k==0 / c-tile-0 programs, all storing straight into the outputs.
    1 launch, 3 allocations.
  * light split (large T_out*B at small MACs): the same fused launch but each dW
    program reduces only CHUNK rows and writes deterministic partials into one
    scratch buffer; a second segmented launch sums them. 2 launches, 4 allocations.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _conv_tbc_fused_kernel(
    g_ptr,
    x_ptr,
    w_ptr,
    dwpo_ptr,
    dbpo_ptr,
    dx_ptr,
    T,
    B,
    C,
    M,
    T_out,
    PAD,
    ROWS,
    ROWS_OUT,
    CHUNK,
    NUMEL,
    NDX,
    sGt,
    sGb,
    sGm,
    sXt,
    sXb,
    sXc,
    sWk,
    sWc,
    sWm,
    sDt,
    sDb,
    sDc,
    K: tl.constexpr,
    NC: tl.constexpr,
    NM: tl.constexpr,
    NCW: tl.constexpr,
    NMW: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    BM: tl.constexpr,
    BCW: tl.constexpr,
    BMW: tl.constexpr,
    BRN: tl.constexpr,
    IS_F64: tl.constexpr,
    DIRECT: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < NDX:
        # ---------------- dX segment ----------------
        pid_r = pid // NC
        pid_c = pid - pid_r * NC
        rowsx = pid_r * BR + tl.arange(0, BR)
        rmaskx = rowsx < ROWS
        tx = rowsx // B
        bx = rowsx - tx * B
        colsx = pid_c * BC + tl.arange(0, BC)
        cmaskx = colsx < C
        if IS_F64:
            accx = tl.zeros((BR, BC), dtype=tl.float64)
        else:
            accx = tl.zeros((BR, BC), dtype=tl.float32)
        nM = tl.cdiv(M, BM)
        for kx in tl.static_range(K):
            tsrc = tx - kx + PAD
            rowok = rmaskx & (tsrc >= 0) & (tsrc < T_out)
            g_off = tsrc * sGt + bx * sGb
            w_off = kx * sWk + colsx * sWc
            for mb in range(0, nM):
                msx = mb * BM + tl.arange(0, BM)
                mmaskx = msx < M
                gx = tl.load(
                    g_ptr + g_off[:, None] + msx[None, :] * sGm,
                    mask=rowok[:, None] & mmaskx[None, :],
                    other=0.0,
                )
                wx = tl.load(
                    w_ptr + w_off[None, :] + msx[:, None] * sWm,
                    mask=mmaskx[:, None] & cmaskx[None, :],
                    other=0.0,
                )
                if IS_F64:
                    accx += tl.sum(gx[:, :, None] * wx[None, :, :], axis=1)
                else:
                    accx = tl.dot(gx, wx, accx, input_precision="ieee")
        tl.store(
            dx_ptr + tx[:, None] * sDt + bx[:, None] * sDb + colsx[None, :] * sDc,
            accx.to(dx_ptr.dtype.element_ty),
            mask=rmaskx[:, None] & cmaskx[None, :],
        )
    else:
        # ---------------- dW (+dB) segment ----------------
        nbase = K * NCW * NMW
        q = pid - NDX
        split = q // nbase
        p = q - split * nbase
        ncm = NCW * NMW
        k = p // ncm
        rem = p - k * ncm
        pid_cw = rem // NMW
        pid_m = rem - pid_cw * NMW

        cols = pid_cw * BCW + tl.arange(0, BCW)
        cmask = cols < C
        ms = pid_m * BMW + tl.arange(0, BMW)
        mmask = ms < M
        write_db = (k == 0) & (pid_cw == 0)

        if IS_F64:
            accw = tl.zeros((BCW, BMW), dtype=tl.float64)
            dbacc = tl.zeros((BMW,), dtype=tl.float64)
        else:
            accw = tl.zeros((BCW, BMW), dtype=tl.float32)
            dbacc = tl.zeros((BMW,), dtype=tl.float32)

        start = split * CHUNK
        end = start + CHUNK
        x_base = x_ptr + cols[None, :] * sXc
        g_base = g_ptr + ms[None, :] * sGm
        for r0 in range(start, end, BRN):
            rows = r0 + tl.arange(0, BRN)
            rmask = rows < ROWS_OUT
            t = rows // B
            b = rows - t * B
            tin = t + k - PAD
            rok = rmask & (tin >= 0) & (tin < T)
            x = tl.load(
                x_base + tin[:, None] * sXt + b[:, None] * sXb,
                mask=rok[:, None] & cmask[None, :],
                other=0.0,
            )
            g = tl.load(
                g_base + t[:, None] * sGt + b[:, None] * sGb,
                mask=rmask[:, None] & mmask[None, :],
                other=0.0,
            )
            if IS_F64:
                accw += tl.sum(x[:, :, None] * g[:, None, :], axis=0)
                dbacc += tl.where(write_db, tl.sum(g, axis=0), 0.0)
            else:
                accw = tl.dot(tl.trans(x), g, accw, input_precision="ieee")
                dbacc += tl.where(write_db, tl.sum(g.to(tl.float32), axis=0), 0.0)

        offs = k * (C * M) + cols[:, None] * M + ms[None, :]
        if DIRECT:
            tl.store(
                dwpo_ptr + offs,
                accw.to(dwpo_ptr.dtype.element_ty),
                mask=cmask[:, None] & mmask[None, :],
            )
            if write_db:
                tl.store(
                    dbpo_ptr + ms,
                    dbacc.to(dbpo_ptr.dtype.element_ty),
                    mask=mmask,
                )
        else:
            tl.store(
                dwpo_ptr + split * NUMEL + offs,
                accw.to(dwpo_ptr.dtype.element_ty),
                mask=cmask[:, None] & mmask[None, :],
            )
            if write_db:
                tl.store(
                    dbpo_ptr + split * M + ms,
                    dbacc.to(dbpo_ptr.dtype.element_ty),
                    mask=mmask,
                )


@triton.jit
def _conv_tbc_dx_kernel(
    g_ptr,
    w_ptr,
    dx_ptr,
    T,
    B,
    C,
    M,
    T_out,
    PAD,
    ROWS,
    sGt,
    sGb,
    sGm,
    sWk,
    sWc,
    sWm,
    sDt,
    sDb,
    sDc,
    K: tl.constexpr,
    NC: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    BM: tl.constexpr,
    IS_F64: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_r = pid // NC
    pid_c = pid - pid_r * NC
    rows = pid_r * BR + tl.arange(0, BR)
    rmask = rows < ROWS
    t = rows // B
    b = rows - t * B
    cols = pid_c * BC + tl.arange(0, BC)
    cmask = cols < C
    if IS_F64:
        accx = tl.zeros((BR, BC), dtype=tl.float64)
    else:
        accx = tl.zeros((BR, BC), dtype=tl.float32)
    nM = tl.cdiv(M, BM)
    for kx in tl.static_range(K):
        tsrc = t - kx + PAD
        rowok = rmask & (tsrc >= 0) & (tsrc < T_out)
        g_off = tsrc * sGt + b * sGb
        w_off = kx * sWk + cols * sWc
        for mb in range(0, nM):
            ms = mb * BM + tl.arange(0, BM)
            mmask = ms < M
            gx = tl.load(
                g_ptr + g_off[:, None] + ms[None, :] * sGm,
                mask=rowok[:, None] & mmask[None, :],
                other=0.0,
            )
            wx = tl.load(
                w_ptr + w_off[None, :] + ms[:, None] * sWm,
                mask=mmask[:, None] & cmask[None, :],
                other=0.0,
            )
            if IS_F64:
                accx += tl.sum(gx[:, :, None] * wx[None, :, :], axis=1)
            else:
                accx = tl.dot(gx, wx, accx, input_precision="ieee")
    tl.store(
        dx_ptr + t[:, None] * sDt + b[:, None] * sDb + cols[None, :] * sDc,
        accx.to(dx_ptr.dtype.element_ty),
        mask=rmask[:, None] & cmask[None, :],
    )


@triton.jit
def _conv_tbc_dw_kernel(
    g_ptr,
    x_ptr,
    dwp_ptr,
    dbp_ptr,
    T,
    B,
    C,
    M,
    T_out,
    PAD,
    ROWS_OUT,
    CHUNK,
    NUMEL,
    sGt,
    sGb,
    sGm,
    sXt,
    sXb,
    sXc,
    K: tl.constexpr,
    NC: tl.constexpr,
    NM: tl.constexpr,
    BC: tl.constexpr,
    BM: tl.constexpr,
    BRN: tl.constexpr,
    IS_F64: tl.constexpr,
    DIRECT: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    ncm = NC * NM
    k = pid // ncm
    rem = pid - k * ncm
    pid_cw = rem // NM
    pid_m = rem - pid_cw * NM

    cols = pid_cw * BC + tl.arange(0, BC)
    cmask = cols < C
    ms = pid_m * BM + tl.arange(0, BM)
    mmask = ms < M
    write_db = (k == 0) & (pid_cw == 0)

    if IS_F64:
        accw = tl.zeros((BC, BM), dtype=tl.float64)
        dbacc = tl.zeros((BM,), dtype=tl.float64)
    else:
        accw = tl.zeros((BC, BM), dtype=tl.float32)
        dbacc = tl.zeros((BM,), dtype=tl.float32)

    start = split * CHUNK
    end = start + CHUNK
    x_base = x_ptr + cols[None, :] * sXc
    g_base = g_ptr + ms[None, :] * sGm
    for r0 in range(start, end, BRN):
        rows = r0 + tl.arange(0, BRN)
        rmask = rows < ROWS_OUT
        t = rows // B
        b = rows - t * B
        tin = t + k - PAD
        rok = rmask & (tin >= 0) & (tin < T)
        x = tl.load(
            x_base + tin[:, None] * sXt + b[:, None] * sXb,
            mask=rok[:, None] & cmask[None, :],
            other=0.0,
        )
        g = tl.load(
            g_base + t[:, None] * sGt + b[:, None] * sGb,
            mask=rmask[:, None] & mmask[None, :],
            other=0.0,
        )
        if IS_F64:
            accw += tl.sum(x[:, :, None] * g[:, None, :], axis=0)
            dbacc += tl.where(write_db, tl.sum(g, axis=0), 0.0)
        else:
            accw = tl.dot(tl.trans(x), g, accw, input_precision="ieee")
            dbacc += tl.where(write_db, tl.sum(g.to(tl.float32), axis=0), 0.0)

    offs = k * (C * M) + cols[:, None] * M + ms[None, :]
    if DIRECT:
        tl.store(
            dwp_ptr + offs,
            accw.to(dwp_ptr.dtype.element_ty),
            mask=cmask[:, None] & mmask[None, :],
        )
        if write_db:
            tl.store(
                dbp_ptr + ms,
                dbacc.to(dbp_ptr.dtype.element_ty),
                mask=mmask,
            )
    else:
        tl.store(
            dwp_ptr + split * NUMEL + offs,
            accw.to(dwp_ptr.dtype.element_ty),
            mask=cmask[:, None] & mmask[None, :],
        )
        if write_db:
            tl.store(
                dbp_ptr + split * M + ms,
                dbacc.to(dbp_ptr.dtype.element_ty),
                mask=mmask,
            )


@triton.jit
def _conv_tbc_reduce_kernel(
    dwp_ptr,
    dbp_ptr,
    dw_ptr,
    db_ptr,
    NUMEL,
    M,
    NSPLIT,
    BLOCK: tl.constexpr,
    BM: tl.constexpr,
    NDW: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < NDW:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < NUMEL
        accw = tl.zeros((BLOCK,), dtype=dwp_ptr.dtype.element_ty)
        for i in range(0, NSPLIT):
            accw += tl.load(dwp_ptr + i * NUMEL + offs, mask=mask, other=0.0)
        tl.store(dw_ptr + offs, accw.to(dw_ptr.dtype.element_ty), mask=mask)
    else:
        j = pid - NDW
        offs_b = j * BM + tl.arange(0, BM)
        mask_b = offs_b < M
        accb = tl.zeros((BM,), dtype=dbp_ptr.dtype.element_ty)
        for i in range(0, NSPLIT):
            accb += tl.load(dbp_ptr + i * M + offs_b, mask=mask_b, other=0.0)
        tl.store(db_ptr + offs_b, accb.to(db_ptr.dtype.element_ty), mask=mask_b)


def conv_tbc_backward(self, input, weight, bias, pad=0):
    logger.debug("GEMS_HYGON CONV_TBC_BACKWARD")
    grad = self
    if grad.dim() != 3 or input.dim() != 3 or weight.dim() != 3:
        raise RuntimeError(
            "conv_tbc_backward: expected 3-D grad_output, input and weight tensors"
        )
    T, B, C = input.shape
    K, Cin, M = weight.shape
    T_out, Bg, Mg = grad.shape
    if Cin != C:
        raise RuntimeError(
            "conv_tbc_backward: input and weight must agree on in_channels"
        )
    if Bg != B or Mg != M:
        raise RuntimeError(
            "conv_tbc_backward: grad_output must match batch and out_channels"
        )
    if bias.numel() != M:
        raise RuntimeError("conv_tbc_backward: bias size must match out_channels")
    if T_out < T + 2 * pad - K + 1:
        raise RuntimeError(
            "conv_tbc_backward: grad_output time length is too small for input/weight/pad"
        )

    dev = input.device
    grad_input = torch.empty((T, B, C), dtype=input.dtype, device=dev)
    grad_weight = torch.empty((K, C, M), dtype=weight.dtype, device=dev)
    grad_bias = torch.empty((M,), dtype=bias.dtype, device=dev)

    rows = T * B
    rows_out = T_out * B
    if rows > 0 and rows_out > 0 and K > 0:
        is_f64 = (
            grad.dtype == torch.float64
            or input.dtype == torch.float64
            or weight.dtype == torch.float64
        )
        pdt = torch.float64 if is_f64 else torch.float32
        sGt, sGb, sGm = grad.stride()
        sXt, sXb, sXc = input.stride()
        sWk, sWc, sWm = weight.stride()

        BC = max(16, min(64, triton.next_power_of_2(C)))
        BM = max(16, min(64, triton.next_power_of_2(M)))
        BR = 16
        BRN = 32
        NC = triton.cdiv(C, BC)
        NM = triton.cdiv(M, BM)
        numel = K * C * M
        ndx = triton.cdiv(rows, BR) * NC
        sDt = B * C
        sDb = C
        macs = K * C * M * (rows + rows_out)

        if macs > 1_000_000_000:
            # GPU-bound: dedicated kernels + wide row split.
            nbase = K * NC * NM
            total_iter = triton.cdiv(rows_out, BRN)
            _conv_tbc_dx_kernel[(ndx,)](
                grad,
                weight,
                grad_input,
                T,
                B,
                C,
                M,
                T_out,
                pad,
                rows,
                sGt,
                sGb,
                sGm,
                sWk,
                sWc,
                sWm,
                sDt,
                sDb,
                1,
                K=K,
                NC=NC,
                BR=BR,
                BC=BC,
                BM=BM,
                IS_F64=is_f64,
            )
            nsplit = max(1, min(total_iter, triton.cdiv(320, nbase), 32))
            nsplit = min(nsplit, max(1, 8_000_000 // (4 * numel)))
            chunk_iter = triton.cdiv(total_iter, nsplit)
            nsplit = triton.cdiv(total_iter, chunk_iter)
            chunk = chunk_iter * BRN
            buf = torch.empty((nsplit * numel + nsplit * M,), dtype=pdt, device=dev)
            dbp = buf.narrow(0, nsplit * numel, nsplit * M)
            _conv_tbc_dw_kernel[(nbase, nsplit)](
                grad,
                input,
                buf,
                dbp,
                T,
                B,
                C,
                M,
                T_out,
                pad,
                rows_out,
                chunk,
                numel,
                sGt,
                sGb,
                sGm,
                sXt,
                sXb,
                sXc,
                K=K,
                NC=NC,
                NM=NM,
                BC=BC,
                BM=BM,
                BRN=BRN,
                IS_F64=is_f64,
                DIRECT=False,
            )
            BLOCK = 1024
            BMr = max(16, min(64, triton.next_power_of_2(M)))
            ndw = triton.cdiv(numel, BLOCK)
            _conv_tbc_reduce_kernel[(ndw + triton.cdiv(M, BMr),)](
                buf,
                dbp,
                grad_weight,
                grad_bias,
                numel,
                M,
                nsplit,
                BLOCK=BLOCK,
                BM=BMr,
                NDW=ndw,
            )
        else:
            # Light: one fused launch, direct store when the whole dW
            # reduction needs only a few row steps, otherwise split.
            BCW = max(16, min(32, triton.next_power_of_2(C)))
            BMW = max(16, min(32, triton.next_power_of_2(M)))
            NCW = triton.cdiv(C, BCW)
            NMW = triton.cdiv(M, BMW)
            nbase = K * NCW * NMW
            if rows_out <= 4096:
                direct = True
                # Two-byte operands keep a wide iteration cheap; fp32 staging
                # is throughput bound at the smaller row block.
                if grad.element_size() <= 2:
                    brnw = 256
                else:
                    brnw = 128
                chunk = triton.cdiv(rows_out, brnw) * brnw
                nsplit = 1
                out_w = grad_weight
                out_b = grad_bias
                buf = None
                dbp = None
            else:
                direct = False
                brnw = 32
                total_iter = triton.cdiv(rows_out, 32)
                nsplit = max(1, min(total_iter, triton.cdiv(256, nbase), 32))
                nsplit = min(nsplit, max(1, 3_000_000 // (4 * numel)))
                chunk_iter = triton.cdiv(total_iter, nsplit)
                nsplit = triton.cdiv(total_iter, chunk_iter)
                chunk = chunk_iter * 32
                buf = torch.empty((nsplit * numel + nsplit * M,), dtype=pdt, device=dev)
                dbp = buf.narrow(0, nsplit * numel, nsplit * M)
                out_w = buf
                out_b = dbp
            _conv_tbc_fused_kernel[(ndx + nbase * nsplit,)](
                grad,
                input,
                weight,
                out_w,
                out_b,
                grad_input,
                T,
                B,
                C,
                M,
                T_out,
                pad,
                rows,
                rows_out,
                chunk,
                numel,
                ndx,
                sGt,
                sGb,
                sGm,
                sXt,
                sXb,
                sXc,
                sWk,
                sWc,
                sWm,
                sDt,
                sDb,
                1,
                K=K,
                NC=NC,
                NM=NM,
                NCW=NCW,
                NMW=NMW,
                BR=BR,
                BC=BC,
                BM=BM,
                BCW=BCW,
                BMW=BMW,
                BRN=brnw,
                IS_F64=is_f64,
                DIRECT=direct,
            )
            if not direct:
                BLOCK = 1024
                BMr = max(16, min(64, triton.next_power_of_2(M)))
                ndw = triton.cdiv(numel, BLOCK)
                _conv_tbc_reduce_kernel[(ndw + triton.cdiv(M, BMr),)](
                    buf,
                    dbp,
                    grad_weight,
                    grad_bias,
                    numel,
                    M,
                    nsplit,
                    BLOCK=BLOCK,
                    BM=BMr,
                    NDW=ndw,
                )
    else:
        grad_input.zero_()
        grad_weight.zero_()
        grad_bias.zero_()

    return grad_input, grad_weight, grad_bias
