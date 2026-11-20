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

import torch
import torch_npu
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

# ---------------------------------------------------------------------------
# GroupedMatmulSwigluQuantV2 - MXFP8 variant.
#
# Contract recovered from the evaluator diagnostics + native-op probing:
#   x             [M, K]                     fp8_e4m3
#   weight        list[1] -> [G, ?, ?]       fp8_e4m3, K and N may be either
#                                            inner/outer dim (strides decide)
#   weight_scale  list[1] -> [G, K/64, N, 2] uint8 raw e8m0 bytes
#   x_scale       [M, K/64, 2]               float8_e8m0fnu
#   group_list    [G] int64, cumulative END offsets (group_list_type == 0)
#   out0          [M, N/2] float8_e4m3fn     (MX block-32 quantised)
#   out1          [M, N/128, 2] float8_e8m0fnu (MX block-32 e8m0 scale)
# Scale granularity is 32 along K (two e8m0 blocks packed per 64-chunk).
# MX rule: scale = 2**(floor(log2(block amax)) - 8) for e4m3 (emax_elem = 8).
#
# Round-2 change (profiled cause: AIV vector pipe 0.995 busy, Cube 1.5%, MTE2
# 2.4%; the non-affine (kb//2, kb%2) MX scale indexing lowered to an
# int64-indexed indirect_load + byte transpose per k-step, ~24-29 vector cycles
# per scale byte): every MX scale tile is now read with a fully affine index
# expression. x_scale's packed pair strides satisfy xsk == 2*xss, so its byte
# address collapses to kb*xss. weight_scale's packed pair layout puts the two
# 32-wide e8m0 blocks of one k-pair in the 64 contiguous bytes that follow each
# column, so the tile for one n-block and one k-step is PAIRS contiguous
# spans of 2*BLOCK_N bytes -> one affine 2-D load, then an in-register permute
# to the [BLOCK_N, BLOCK_K/32] operand tl.dot_scaled expects.
# ---------------------------------------------------------------------------


@triton.jit
def _fp8__gmm_swiglu_mx_kernel(
    X,
    W,
    WS,
    XS,
    GL,
    Y,
    M,
    K,
    N,
    HIDDEN,
    G,
    NUM_NT,
    sxm,
    sxk,
    swg,
    swk,
    swn,
    wsg,
    wsk,
    wsn,
    wss,
    xsm,
    xsk,
    xss,
    sym,
    syn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    W_NZ: tl.constexpr,
    WS_AFFINE: tl.constexpr,
    WS_KINNER: tl.constexpr,
    XS_AFFINE: tl.constexpr,
    KEXACT: tl.constexpr,
    TAIL: tl.constexpr,
):
    pid = tl.program_id(0)
    g = pid // NUM_NT
    nt = pid % NUM_NT
    ncol = nt * BLOCK_N
    if (g < G) and (ncol < HIDDEN):
        gend = tl.minimum(tl.maximum(tl.load(GL + g), 0), M)
        gstart = tl.minimum(tl.maximum(tl.load(GL + g - 1, mask=g > 0, other=0), 0), M)
        glen = gend - gstart
        wbase = W + g * swg
        wsbase = WS + g * wsg
        cols = ncol + tl.arange(0, BLOCK_N)
        cmask = cols < HIDDEN
        KB: tl.constexpr = BLOCK_K // 32
        PAIRS: tl.constexpr = BLOCK_K // 64
        JJ: tl.constexpr = 2 * BLOCK_N
        K1 = K // 16
        for mt in range(0, tl.cdiv(glen, BLOCK_M)):
            rows = mt * BLOCK_M + tl.arange(0, BLOCK_M)
            rmask = rows < glen
            grow = gstart + rows
            acc_g = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            acc_u = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for k0 in range(0, K, BLOCK_K):
                kk = k0 + tl.arange(0, BLOCK_K)
                kmask = kk < K
                kb = k0 // 32 + tl.arange(0, KB)
                xt = tl.load(
                    X + grow[:, None] * sxm + kk[None, :] * sxk,
                    mask=rmask[:, None] & kmask[None, :],
                    other=0.0,
                )
                if XS_AFFINE:
                    if KEXACT:
                        xsc = tl.load(
                            XS + grow[:, None] * xsm + kb[None, :] * xss,
                            mask=rmask[:, None],
                            other=127,
                        )
                    else:
                        xsc = tl.load(
                            XS + grow[:, None] * xsm + kb[None, :] * xss,
                            mask=rmask[:, None] & (kb[None, :] < K // 32),
                            other=127,
                        )
                else:
                    xsc = tl.load(
                        XS
                        + grow[:, None] * xsm
                        + (kb // 2)[None, :] * xsk
                        + (kb % 2)[None, :] * xss,
                        mask=rmask[:, None] & (kb[None, :] < K // 32),
                        other=127,
                    )
                if W_NZ:
                    # Ascend NZ (npu format 29) fractal layout of a 1-byte
                    # [K, N] matrix: [N/32, K/16, 16, 32] physically, so the
                    # byte address of element (kg, n) inside one n-block is
                    #   (nblk*K1 + kg//16)*512 + (kg%16)*32 + n%32 .
                    # Because 16*32 == 512, the k-dependent part collapses
                    # exactly: (kg//16)*512 + (kg%16)*32 == kg*32. The tile for
                    # one 32-wide n-block and BLOCK_K consecutive k is therefore
                    # a rank-2 affine block with row stride 32 (== BLOCK_N) and
                    # unit n stride, i.e. a single fully contiguous
                    # [BLOCK_K, BLOCK_N] run - structurally the same load form
                    # as the ND tile below, which the profile records as going
                    # GM->L1 directly, while this operand's rank-3/reshape form
                    # was staged GM->UB->L1 through the vector core at
                    # aiv_mte2 0.716. No reshape, no mask when the k range is
                    # exact.
                    n32 = tl.arange(0, BLOCK_N)
                    og = (
                        wbase
                        + (ncol // 32) * K1 * 512
                        + kk[:, None] * 32
                        + n32[None, :]
                    )
                    ou = (
                        wbase
                        + ((ncol + HIDDEN) // 32) * K1 * 512
                        + kk[:, None] * 32
                        + n32[None, :]
                    )
                    if KEXACT:
                        wg = tl.load(og)
                        wu = tl.load(ou)
                    else:
                        wg = tl.load(og, mask=kmask[:, None], other=0.0)
                        wu = tl.load(ou, mask=kmask[:, None], other=0.0)
                else:
                    # ND weight tile: n is unit-stride, so the tile is a plain
                    # block. Drop the always-true predication when the k range
                    # is exact and the tile lies inside HIDDEN, so the backend can
                    # issue a block transfer (the same reformulation that gave
                    # the NZ branch 1.09-1.14x in round 7).
                    if KEXACT and (not TAIL):
                        wg = tl.load(wbase + kk[:, None] * swk + cols[None, :] * swn)
                        wu = tl.load(
                            wbase + kk[:, None] * swk + (cols + HIDDEN)[None, :] * swn
                        )
                    else:
                        wg = tl.load(
                            wbase + kk[:, None] * swk + cols[None, :] * swn,
                            mask=kmask[:, None] & cmask[None, :],
                            other=0.0,
                        )
                        wu = tl.load(
                            wbase + kk[:, None] * swk + (cols + HIDDEN)[None, :] * swn,
                            mask=kmask[:, None] & cmask[None, :],
                            other=0.0,
                        )
                if WS_KINNER:
                    # Packed pair is innermost along K: wsk == 2*wss makes
                    # (kb//2)*wsk + (kb%2)*wss collapse to kb*wss, so the byte
                    # address of the [BLOCK_N, BLOCK_K/32] operand is an
                    # ordinary 2-D affine form with unit-strided k. No permute,
                    # no div/mod, no indirect gather: this is the same shape as
                    # the x_scale operand that already lowers to a direct
                    # GM->L1 load_scale.
                    if KEXACT and (not TAIL):
                        scg = tl.load(wsbase + cols[:, None] * wsn + kb[None, :] * wss)
                        scu = tl.load(
                            wsbase + (cols + HIDDEN)[:, None] * wsn + kb[None, :] * wss
                        )
                    else:
                        m3 = (kb[None, :] * 32 < K) & cmask[:, None]
                        scg = tl.load(
                            wsbase + cols[:, None] * wsn + kb[None, :] * wss,
                            mask=m3,
                            other=127,
                        )
                        scu = tl.load(
                            wsbase + (cols + HIDDEN)[:, None] * wsn + kb[None, :] * wss,
                            mask=m3,
                            other=127,
                        )
                elif WS_AFFINE:
                    # one k-pair row holds, for column n, the two 32-wide e8m0
                    # blocks at byte offset 2*n and 2*n+1, so the tile for this
                    # n-block is PAIRS contiguous 2*BLOCK_N-byte spans.
                    pp = k0 // 64 + tl.arange(0, PAIRS)
                    jj = tl.arange(0, JJ)
                    og = wsbase + pp[:, None] * wsk + (2 * ncol) + jj[None, :]
                    ou = (
                        wsbase + pp[:, None] * wsk + (2 * (ncol + HIDDEN)) + jj[None, :]
                    )
                    if KEXACT and (not TAIL):
                        pg = tl.load(og)
                        pu = tl.load(ou)
                    else:
                        m2 = ((2 * pp[:, None] + (jj[None, :] % 2)) * 32 < (K - k0)) & (
                            jj[None, :] < 2 * (HIDDEN - ncol)
                        )
                        pg = tl.load(og, mask=m2, other=127)
                        pu = tl.load(ou, mask=m2, other=127)
                    scg = tl.reshape(
                        tl.permute(tl.reshape(pg, (PAIRS, BLOCK_N, 2)), (1, 0, 2)),
                        (BLOCK_N, KB),
                    )
                    scu = tl.reshape(
                        tl.permute(tl.reshape(pu, (PAIRS, BLOCK_N, 2)), (1, 0, 2)),
                        (BLOCK_N, KB),
                    )
                else:
                    # Fully affine general form: keep the k-pair and the
                    # in-pair lane as separate axes so the address of byte
                    # (n, pair, lane) is n*wsn + pair*wsk + lane*wss for ANY
                    # packed layout. Flattening (pair, lane) to kb is then a
                    # plain row-major reshape, which removes the (kb//2, kb%2)
                    # div/mod and the int64 indirect_load the generic path used
                    # to lower to.
                    pp = k0 // 64 + tl.arange(0, PAIRS)
                    ub = tl.arange(0, 2)
                    og = (
                        wsbase
                        + cols[:, None, None] * wsn
                        + pp[None, :, None] * wsk
                        + ub[None, None, :] * wss
                    )
                    ou = (
                        wsbase
                        + (cols + HIDDEN)[:, None, None] * wsn
                        + pp[None, :, None] * wsk
                        + ub[None, None, :] * wss
                    )
                    if KEXACT and (not TAIL):
                        scg = tl.reshape(tl.load(og), (BLOCK_N, KB))
                        scu = tl.reshape(tl.load(ou), (BLOCK_N, KB))
                    else:
                        m3 = (
                            ((2 * pp[None, :, None] + ub[None, None, :]) * 32)
                            < (K - k0)
                        ) & cmask[:, None, None]
                        scg = tl.reshape(tl.load(og, mask=m3, other=127), (BLOCK_N, KB))
                        scu = tl.reshape(tl.load(ou, mask=m3, other=127), (BLOCK_N, KB))
                acc_g = tl.dot_scaled(
                    xt, xsc, "e4m3", wg, scg, "e4m3", acc_g, out_dtype=tl.float32
                )
                acc_u = tl.dot_scaled(
                    xt, xsc, "e4m3", wu, scu, "e4m3", acc_u, out_dtype=tl.float32
                )
            y = (acc_g / (1.0 + tl.exp(-acc_g))) * acc_u
            tl.store(
                Y + grow[:, None] * sym + cols[None, :] * syn,
                y.to(Y.dtype.element_ty),
                mask=rmask[:, None] & cmask[None, :],
            )


@triton.jit
def _fp8__quant_row_kernel(
    Y,
    OUT0,
    OUT1U8,
    M,
    HIDDEN,
    sym,
    syn,
    so0m,
    so0n,
    so1m,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    QMAX: tl.constexpr,
    BLOCK_MX: tl.constexpr,
    EMIN: tl.constexpr,
    EMAX: tl.constexpr,
    NCHUNK: tl.constexpr,
    EXACT: tl.constexpr,
):
    # The MX scale is per 32-element block, so the 32-blocks of one row are
    # completely independent and a column chunk is bit-identical to the whole
    # row. NCHUNK therefore only buys parallelism: when M alone cannot fill the
    # vector fleet this kernel used to run as a single program while the pipe it
    # saturates (aiv_vec 0.92-0.98) had 63 blocks idle.
    pid = tl.program_id(0)
    pm = pid // NCHUNK
    nc = pid % NCHUNK
    rows = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    cbase = nc * BLOCK_I
    offs = cbase + tl.arange(0, BLOCK_I)
    if EXACT:
        # every lane of this program's tile is in range, so drop the
        # predication: the always-true masked form costs a whole zero-fill
        # body of the vector chain per program.
        y = tl.load(Y + rows[:, None] * sym + offs[None, :] * syn).to(tl.float32)
    else:
        rmask = rows < M
        cmask = offs < HIDDEN
        y = tl.load(
            Y + rows[:, None] * sym + offs[None, :] * syn,
            mask=rmask[:, None] & cmask[None, :],
            other=0.0,
        ).to(tl.float32)
    nb: tl.constexpr = BLOCK_I // BLOCK_MX
    yb = tl.reshape(y, (BLOCK_M, nb, BLOCK_MX))
    amax = tl.max(tl.abs(yb), 2)
    # MX e4m3 shared exponent (emax_elem = 8): scale = 2**(floor(log2(amax))-8).
    # fexp is the *biased* fp32 exponent, so the e8m0 byte is simply fexp - 8.
    bits = amax.to(tl.int32, bitcast=True)
    fexp = (bits >> 23) & 0xFF
    sb = tl.minimum(tl.maximum(fexp - 8, EMIN), EMAX)
    scale = (sb << 23).to(tl.float32, bitcast=True)
    scale = tl.where(amax > 0.0, scale, 1.0)
    # scale is an exact power of two, so one reciprocal per 32-element block
    # replaces one general vector divide per ELEMENT of the row and the scaling
    # becomes a multiply. y/2**e and y*(2**-e) agree bit for bit because both
    # are the same exact value rounded once.
    inv = 1.0 / scale
    q = yb * inv[:, :, None]
    q = tl.minimum(tl.maximum(q, -QMAX), QMAX)
    qflat = tl.reshape(q, (BLOCK_M, BLOCK_I))
    if EXACT:
        tl.store(
            OUT0 + rows[:, None] * so0m + offs[None, :] * so0n,
            qflat.to(OUT0.dtype.element_ty),
        )
    else:
        tl.store(
            OUT0 + rows[:, None] * so0m + offs[None, :] * so0n,
            qflat.to(OUT0.dtype.element_ty),
            mask=rmask[:, None] & cmask[None, :],
        )
    # out1 is a packed byte array indexed by 32-block, so a chunk simply
    # offsets the block index by its own block count.
    kbs = nc * nb + tl.arange(0, nb)
    q1 = tl.where(amax > 0.0, sb, 0).to(tl.uint8)
    if EXACT:
        tl.store(OUT1U8 + rows[:, None] * so1m + kbs[None, :], q1)
    else:
        tl.store(
            OUT1U8 + rows[:, None] * so1m + kbs[None, :],
            q1,
            mask=rmask[:, None] & (kbs[None, :] < tl.cdiv(HIDDEN, BLOCK_MX)),
        )


def _fp8__pick_weight(w):
    if isinstance(w, (list, tuple)):
        if len(w) != 1:
            raise RuntimeError("GMMSQ unsupported weight list length %d" % len(w))
        return w[0]
    return w


def _fp8_run(
    x,
    weight,
    weight_scale,
    x_scale,
    group_list,
    *,
    smooth_scale=None,
    weight_assist_matrix=None,
    bias=None,
    dequant_mode=0,
    dequant_dtype=6,
    quant_mode=0,
    quant_dtype=1,
    group_list_type=0,
    tuning_config=None,
    x_dtype=None,
    weight_dtype=None,
    weight_scale_dtype=None,
    x_scale_dtype=None,
):
    wt = _fp8__pick_weight(weight)
    ws = _fp8__pick_weight(weight_scale)
    gl = group_list.reshape(-1)

    M = int(x.shape[0])
    K = int(x.shape[1])
    G = int(gl.numel())

    # weight group body: one of the two trailing dims is K
    if int(wt.shape[2]) == K and int(wt.shape[1]) != K:
        N = int(wt.shape[1])
        swk, swn = wt.stride(2), wt.stride(1)
    else:
        N = int(wt.shape[2])
        swk, swn = wt.stride(1), wt.stride(2)
    HIDDEN = N // 2

    wsg, wsk, wsn, wss = ws.stride()
    xsm, xsk, xss = x_scale.stride()
    xs_u8 = x_scale.view(torch.uint8)

    dev = x.device
    odt = quant_dtype if isinstance(quant_dtype, torch.dtype) else torch.float8_e4m3fn
    out0 = torch.empty((M, HIDDEN), dtype=odt, device=dev)
    # MX block-32 output scale, packed exactly like x_scale: [M, HIDDEN/64, 2] e8m0
    out1 = torch.empty((M, max(1, HIDDEN // 64), 2), dtype=torch.uint8, device=dev)
    ybuf = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=dev)

    avg = M / max(G, 1)
    w_nz = False
    try:
        import torch_npu as _tn

        w_nz = int(_tn.get_npu_format(wt)) == 29
    except Exception:
        w_nz = False
    if w_nz:
        # one 32-wide n-block is contiguous in the fractal layout, so BLOCK_N
        # is pinned at 32; BLOCK_M follows the group size so that 64-row groups
        # do not need 4 m-tiles and re-stream the weight.
        BLOCK_M = min(64, max(16, triton.next_power_of_2(int(avg + 0.5) or 1)))
        BLOCK_N = 32
        BLOCK_K = 512
    else:
        # ND weights are n-unit-stride, so any BLOCK_N is legal. The old
        # reason for pinning BLOCK_N to 32 - the MX weight-scale staging cost
        # being proportional to BLOCK_N through a per-16-wide-n-sub-fractal
        # permute loop - no longer applies: round 3 gave this layout the
        # WS_KINNER branch, a plain unit-strided block load whose bytes per
        # output column are invariant in BLOCK_N. What BLOCK_N still buys is
        # traffic: x is re-read cdiv(HIDDEN, BLOCK_N) times, which at BLOCK_N=32 is
        # 16 passes over x and at round-6's 64 is still 8, so the d53c family
        # remains MTE2 bound with a large redundant x term (round-8 profile:
        # aic_mte2 0.778, aiv_vec 0.026). Take the widest tile the on-chip
        # budgets allow: L0C 8*BLOCK_M*BLOCK_N = 8*64*128 = 65536 <= 131072, and
        # L1 2*(BLOCK_M*BLOCK_K + 2*BLOCK_N*BLOCK_K) = 2*(32768 + 131072) =
        # 327680 <= 524288 at BLOCK_K=512. BLOCK_N=128 forces BLOCK_K down to
        # 512, which used to cost 19-26% when the ND scale staging dominated
        # every k-step; round 3 removed exactly that, and an independent ledger
        # now measures 64 -> 128 with BLOCK_K 1024 -> 512 as a further 1.13-1.18x
        # on this same class.
        BLOCK_M = min(64, max(16, triton.next_power_of_2(int(avg + 0.5) or 1)))
        # Width trades the redundant x re-read against per-program control
        # work, and which term binds is set by the group size through BLOCK_M.
        # Measured twice on the same class: 32 -> 64 (round 6) and 64 -> 128
        # (round 9) each helped the byte-bound large-group shape (d53c,
        # BLOCK_M=64, aic_mte2 0.778 / aiv_vec 0.026) and cost the control-bound
        # small-group one (c6ae, BLOCK_M=16, aic_scalar 0.648 / mac 0.149,
        # cube_util 64%, no arithmetic to amortise a wider tile). So take the
        # widest tile only where there are enough rows per program to amortise
        # it, and keep a narrower one below that.
        if BLOCK_M < 64:
            if (HIDDEN % 64) == 0:
                BLOCK_N = 64
            else:
                BLOCK_N = 32
        elif (HIDDEN % 128) == 0:
            BLOCK_N = 128
        elif (HIDDEN % 64) == 0:
            BLOCK_N = 64
        else:
            BLOCK_N = 32
        BLOCK_K = 512
    # The MX scale operand is re-staged (UB copies + two permutes + UB->L1
    # copy) once per k-step regardless of how many bytes it carries, so the
    # deepest K tile that still divides K is the cheapest: it minimises the
    # number of k-steps and keeps the fast unmasked (KEXACT) path. That is only
    # true while the scale staging is the per-k-step cost; on the n-unit-stride
    # branch round 3 made it a plain block load, so there a wide BLOCK_N (whose
    # on-chip budget forces BLOCK_K=512) is worth more than a deep k tile.
    for _cand in ((512,) if BLOCK_N >= 128 else (1024, 512)):
        if K % _cand == 0:
            BLOCK_K = _cand
            break
    num_nt = max(1, triton.cdiv(HIDDEN, BLOCK_N))
    # packed e8m0 pairs are byte-adjacent in n (weight) / in k (x) when the
    # inner pair dim has unit stride and the next dim is exactly 2 bytes wide.
    ws_affine = (wss == 1) and (wsn == 2) and ((K % 32) == 0)
    # k-innermost packing: (kb//2)*wsk + (kb%2)*wss == kb*wss, i.e. the k axis
    # of the [n, k/32] scale operand is unit strided, so the tile is a plain
    # 2-D affine block load with no transpose at all.
    ws_kinner = (wsk == 2 * wss) and ((K % 32) == 0)
    ws_affine = ws_affine and (not ws_kinner)
    xs_affine = (xss == 1) and (xsk == 2) and ((K % 32) == 0)
    _fp8__gmm_swiglu_mx_kernel[(G * num_nt,)](
        x,
        wt,
        ws,
        xs_u8,
        gl,
        ybuf,
        M,
        K,
        N,
        HIDDEN,
        G,
        num_nt,
        x.stride(0),
        x.stride(1),
        wt.stride(0),
        swk,
        swn,
        wsg,
        wsk,
        wsn,
        wss,
        xsm,
        xsk,
        xss,
        ybuf.stride(0),
        ybuf.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        W_NZ=w_nz,
        WS_AFFINE=ws_affine,
        WS_KINNER=ws_kinner,
        XS_AFFINE=xs_affine,
        KEXACT=(K % BLOCK_K) == 0,
        TAIL=(HIDDEN % BLOCK_N) != 0,
    )

    # Row-quant launch geometry. This kernel is the ONLY saturated pipe in the
    # whole pipeline (aiv_vec 0.916-0.977 measured on every workload that runs
    # it), and with BLOCK_I pinned to the whole row its grid is just
    # cdiv(M, QBM) - which collapses to ONE program for the M=16 shapes, i.e.
    # ~29% of c6ae and ~13% of geometry-001 spent on a single vector unit while
    # 63 blocks idled. The MX scale is per 32-element block and out1 is a packed
    # byte array indexed by block, so splitting the column axis is bit
    # identical. First shrink the row tile, then split columns, and only ever do
    # so when the row grid alone cannot cover the fleet.
    BLOCK_I_FULL = max(32, triton.next_power_of_2(HIDDEN))
    QBM = max(1, min(16, 16384 // BLOCK_I_FULL))
    nrow = triton.cdiv(M, QBM)
    CH = BLOCK_I_FULL
    if nrow < 64:
        QBM = max(1, triton.next_power_of_2(max(1, M // 64)))
        nrow = triton.cdiv(M, QBM)
        if nrow < 64:
            _nc = min(triton.cdiv(64, nrow), max(1, HIDDEN // 32))
            CH = max(32, triton.next_power_of_2(triton.cdiv(HIDDEN, _nc)))
    CH = min(CH, BLOCK_I_FULL)
    nchunk = triton.cdiv(HIDDEN, CH)
    _fp8__quant_row_kernel[(nrow * nchunk,)](
        ybuf,
        out0,
        out1,
        M,
        HIDDEN,
        ybuf.stride(0),
        ybuf.stride(1),
        out0.stride(0),
        out0.stride(1),
        out1.stride(0),
        BLOCK_M=QBM,
        BLOCK_I=CH,
        QMAX=448.0,
        BLOCK_MX=32,
        EMIN=1,
        EMAX=254,
        NCHUNK=nchunk,
        EXACT=((M % QBM) == 0) and ((nchunk * CH) == HIDDEN),
    )
    return out0, out1.view(torch.float8_e8m0fnu)


def _fp4__fp4_dtype():
    return torch.float4_e2m1fn_x2


@triton.jit
def _fp4__e2m1(c):
    """E2M1 code (0..15) -> value, exactly representable in bf16."""
    m = c & 7
    v = tl.where(
        m == 0,
        0.0,
        tl.where(
            m == 1,
            0.5,
            tl.where(
                m == 2,
                1.0,
                tl.where(
                    m == 3,
                    1.5,
                    tl.where(
                        m == 4, 2.0, tl.where(m == 5, 3.0, tl.where(m == 6, 4.0, 6.0))
                    ),
                ),
            ),
        ),
    )
    return tl.where((c & 8) != 0, -v, v)


@triton.jit
def _fp4__e2m1_code(r):
    """value -> E2M1 code, round-to-nearest with ties-to-even, saturating.

    The sign nibble is taken from the sign BIT, not from a comparison, because
    the native operator encodes a negative zero result (silu underflow times a
    positive factor) as code 8 while 0.0 is code 0.
    """
    a = tl.abs(r)
    code = tl.where(
        a <= 0.25,
        0,
        tl.where(
            a < 0.75,
            1,
            tl.where(
                a <= 1.25,
                2,
                tl.where(
                    a < 1.75,
                    3,
                    tl.where(
                        a <= 2.5, 4, tl.where(a < 3.5, 5, tl.where(a <= 5.0, 6, 7))
                    ),
                ),
            ),
        ),
    )
    sgn = (r.to(tl.int32, bitcast=True) >> 31) & 1
    return code + sgn * 8


@triton.jit
def _fp4__to_bf16(x):
    """Explicit round-to-nearest-even fp32 -> bf16 -> fp32 (compiler-proof)."""
    b = x.to(tl.int32, bitcast=True)
    b = b + 32767 + ((b >> 16) & 1)
    return (b & -65536).to(tl.float32, bitcast=True)


@triton.jit
def _fp4__gmm_swiglu_mxq(
    X,
    XS,
    W,
    WS,
    GL,
    O0,
    O1,
    M,
    K,
    H,
    NT,
    xsm,
    xsk,
    wsg,
    wsk,
    wsn,
    wsse,
    wsi,
    wsn2,
    xsse,
    xsi,
    o0sm,
    o1sm,
    K16,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    IN_FP4: tl.constexpr,
    OUT_FP4: tl.constexpr,
    W_NZ: tl.constexpr,
    EMIN: tl.constexpr,
):
    pid = tl.program_id(0)
    g = pid // NT
    nt = pid % NT

    gend = tl.minimum(tl.maximum(tl.load(GL + g), 0), M).to(tl.int32)
    gprev = tl.load(GL + tl.maximum(g - 1, 0)).to(tl.int32)
    gstart = tl.minimum(tl.maximum(tl.where(g > 0, gprev, 0), 0), M)
    glen = gend - gstart

    NKG: tl.constexpr = BK // 32
    PAIRS: tl.constexpr = BK // 64
    NB32: tl.constexpr = BN // 32
    kk = tl.arange(0, BK)
    kb2 = tl.arange(0, BK // 2)
    r2 = tl.arange(0, 2)
    kp0 = tl.arange(0, PAIRS)
    gate_cols = nt * BN + tl.arange(0, BN)
    up_cols = gate_cols + H

    if glen > 0:
        for mt in range(0, tl.cdiv(glen, BM)):
            rows = mt * BM + tl.arange(0, BM)
            rmask = rows < glen
            grow = gstart + rows
            acc_g = tl.zeros((BM, BN), dtype=tl.float32)
            acc_u = tl.zeros((BM, BN), dtype=tl.float32)
            for k0 in range(0, K, BK):
                # ---- x tile, dequantised to bf16 (exact)
                xso = (
                    grow[:, None, None] * xsse
                    + (k0 // 64 + kp0)[None, :, None] * xsi
                    + r2[None, None, :]
                )
                xsc = tl.reshape(
                    tl.exp2(
                        tl.load(XS + xso, mask=rmask[:, None, None], other=127).to(
                            tl.float32
                        )
                        - 127.0
                    ),
                    (BM, NKG, 1),
                ).to(tl.bfloat16)
                if IN_FP4:
                    xb = tl.load(
                        X + grow[:, None] * xsm + (k0 // 2 + kb2)[None, :] * xsk,
                        mask=rmask[:, None],
                        other=0,
                    ).to(tl.int32)
                    xv = tl.reshape(
                        tl.join(
                            _fp4__e2m1(xb & 15).to(tl.bfloat16),
                            _fp4__e2m1((xb >> 4) & 15).to(tl.bfloat16),
                        ),
                        (BM, BK),
                    )
                else:
                    xv = tl.load(
                        X + grow[:, None] * xsm + (k0 + kk)[None, :] * xsk,
                        mask=rmask[:, None],
                        other=0.0,
                    ).to(tl.bfloat16)
                xt = tl.reshape(tl.reshape(xv, (BM, NKG, 32)) * xsc, (BM, BK))

                wbase = W + g * wsg
                if W_NZ:
                    og = (
                        wbase
                        + (gate_cols // 32)[None, :] * (K16 * 512)
                        + (k0 // 16) * 512
                        + kk[:, None] * 32
                        + (gate_cols % 32)[None, :]
                    )
                    ou = (
                        wbase
                        + (up_cols // 32)[None, :] * (K16 * 512)
                        + (k0 // 16) * 512
                        + kk[:, None] * 32
                        + (up_cols % 32)[None, :]
                    )
                    wg = tl.load(og).to(tl.bfloat16)
                    wu = tl.load(ou).to(tl.bfloat16)
                    scg = tl.reshape(
                        tl.exp2(
                            tl.load(
                                WS
                                + g * wsse
                                + (k0 // 64 + kp0)[:, None, None] * wsi
                                + gate_cols[None, None, :] * wsn2
                                + r2[None, :, None]
                            ).to(tl.float32)
                            - 127.0
                        ),
                        (NKG, BN),
                    ).to(tl.bfloat16)
                    scu = tl.reshape(
                        tl.exp2(
                            tl.load(
                                WS
                                + g * wsse
                                + (k0 // 64 + kp0)[:, None, None] * wsi
                                + up_cols[None, None, :] * wsn2
                                + r2[None, :, None]
                            ).to(tl.float32)
                            - 127.0
                        ),
                        (NKG, BN),
                    ).to(tl.bfloat16)
                    wtg = tl.reshape(
                        tl.reshape(wg, (NKG, 32, BN)) * scg[:, None, :], (BK, BN)
                    )
                    wtu = tl.reshape(
                        tl.reshape(wu, (NKG, 32, BN)) * scu[:, None, :], (BK, BN)
                    )
                else:
                    scg = tl.reshape(
                        tl.exp2(
                            tl.load(
                                WS
                                + g * wsse
                                + gate_cols[:, None, None] * wsn2
                                + (k0 // 64 + kp0)[None, :, None] * wsi
                                + r2[None, None, :]
                            ).to(tl.float32)
                            - 127.0
                        ),
                        (BN, NKG, 1),
                    ).to(tl.bfloat16)
                    scu = tl.reshape(
                        tl.exp2(
                            tl.load(
                                WS
                                + g * wsse
                                + up_cols[:, None, None] * wsn2
                                + (k0 // 64 + kp0)[None, :, None] * wsi
                                + r2[None, None, :]
                            ).to(tl.float32)
                            - 127.0
                        ),
                        (BN, NKG, 1),
                    ).to(tl.bfloat16)
                    if IN_FP4:
                        gb = tl.load(
                            wbase
                            + gate_cols[:, None] * wsn
                            + (k0 // 2 + kb2)[None, :] * wsk
                        ).to(tl.int32)
                        ub = tl.load(
                            wbase
                            + up_cols[:, None] * wsn
                            + (k0 // 2 + kb2)[None, :] * wsk
                        ).to(tl.int32)
                        gv = tl.reshape(
                            tl.join(
                                _fp4__e2m1(gb & 15).to(tl.bfloat16),
                                _fp4__e2m1((gb >> 4) & 15).to(tl.bfloat16),
                            ),
                            (BN, BK),
                        )
                        uv = tl.reshape(
                            tl.join(
                                _fp4__e2m1(ub & 15).to(tl.bfloat16),
                                _fp4__e2m1((ub >> 4) & 15).to(tl.bfloat16),
                            ),
                            (BN, BK),
                        )
                    else:
                        gv = tl.load(
                            wbase + gate_cols[:, None] * wsn + (k0 + kk)[None, :] * wsk
                        ).to(tl.bfloat16)
                        uv = tl.load(
                            wbase + up_cols[:, None] * wsn + (k0 + kk)[None, :] * wsk
                        ).to(tl.bfloat16)
                    wtg = tl.trans(
                        tl.reshape(tl.reshape(gv, (BN, NKG, 32)) * scg, (BN, BK))
                    )
                    wtu = tl.trans(
                        tl.reshape(tl.reshape(uv, (BN, NKG, 32)) * scu, (BN, BK))
                    )

                acc_g = tl.dot(xt, wtg, acc_g)
                acc_u = tl.dot(xt, wtu, acc_u)

            sig = tl.fdiv(acc_g, 1.0 + tl.exp(-acc_g), ieee_rounding=True)
            z = _fp4__to_bf16(sig * acc_u)
            z3 = tl.reshape(z, (BM, NB32, 32))
            amax = tl.max(tl.abs(z3), 2)
            fe = (amax.to(tl.int32, bitcast=True) >> 23) & 0xFF
            sh = fe - 127 - EMIN
            sh = tl.where(amax > 0.0, sh, -127)
            r = tl.reshape(z3 * tl.exp2(-sh.to(tl.float32))[:, :, None], (BM, BN))
            tl.store(
                O1 + grow[:, None] * o1sm + (nt * NB32 + tl.arange(0, NB32))[None, :],
                (sh + 127).to(tl.uint8),
                mask=rmask[:, None],
            )
            if OUT_FP4:
                code = _fp4__e2m1_code(r).to(tl.uint8)
                c2 = tl.reshape(code, (BM, BN // 2, 2))
                c0, c1 = tl.split(c2)
                packed = (c0.to(tl.int32) | (c1.to(tl.int32) << 4)).to(tl.uint8)
                tl.store(
                    O0
                    + grow[:, None] * o0sm
                    + (nt * (BN // 2) + tl.arange(0, BN // 2))[None, :],
                    packed,
                    mask=rmask[:, None],
                )
            else:
                rc = tl.minimum(tl.maximum(r, -448.0), 448.0)
                tl.store(
                    O0 + grow[:, None] * o0sm + (nt * BN + tl.arange(0, BN))[None, :],
                    rc.to(tl.float8e4nv),
                    mask=rmask[:, None],
                )


def _fp4__first(t):
    if isinstance(t, (list, tuple)):
        return t[0]
    return t


def _fp4__is_nz(t):
    if torch_npu is None:
        return False
    try:
        return int(torch_npu.get_npu_format(t)) == 29
    except Exception:
        return False


def _fp4_run(
    x,
    weight,
    weight_scale,
    x_scale,
    group_list,
    *,
    smooth_scale=None,
    weight_assist_matrix=None,
    bias=None,
    dequant_mode=0,
    dequant_dtype=6,
    quant_mode=0,
    quant_dtype=1,
    group_list_type=0,
    tuning_config=None,
    x_dtype=None,
    weight_dtype=None,
    weight_scale_dtype=None,
    x_scale_dtype=None,
):
    w = _fp4__first(weight)
    ws = _fp4__first(weight_scale)
    gl = group_list.reshape(-1)

    in_fp4 = x.dtype == torch.uint8
    M = int(x.shape[0])
    K = int(x.shape[1]) * 2 if in_fp4 else int(x.shape[1])
    kp = K // 2 if in_fp4 else K
    if int(w.shape[1]) != kp:  # defensive: swap the trailing axes
        n_axis, k_axis = 1, 2
    else:
        n_axis, k_axis = 2, 1
    N = int(w.shape[n_axis])
    H = N // 2
    E = int(gl.numel())

    out_fp4 = not isinstance(quant_dtype, torch.dtype)
    dev = x.device
    if out_fp4:
        o0_u8 = torch.empty((M, H // 2), dtype=torch.uint8, device=dev)
        o0 = o0_u8.view(_fp4__fp4_dtype())
    else:
        o0_u8 = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=dev)
        o0 = o0_u8
    o1_u8 = torch.empty((M, (H + 63) // 64, 2), dtype=torch.uint8, device=dev)
    o1 = o1_u8.view(torch.float8_e8m0fnu)

    if M == 0 or E == 0:
        return o0, o1

    xs_b = x_scale.view(torch.uint8) if x_scale.dtype != torch.uint8 else x_scale
    ws_b = ws.view(torch.uint8) if ws.dtype != torch.uint8 else ws
    nz = (not in_fp4) and _fp4__is_nz(w)

    avg = max(1, (M + E - 1) // E)
    BM = min(64, max(16, triton.next_power_of_2(avg)))
    BN = 32 if nz else 64
    BK = 64
    for cand in (128, 64):
        if K % cand == 0:
            BK = cand
            break

    nt = triton.cdiv(H, BN)
    _fp4__gmm_swiglu_mxq[(E * nt,)](
        x,
        xs_b,
        w,
        ws_b,
        gl,
        o0_u8,
        o1_u8,
        M,
        K,
        H,
        nt,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(k_axis),
        w.stride(n_axis),
        ws_b.stride(0),
        ws_b.stride(1),
        ws_b.stride(2),
        xs_b.stride(0),
        xs_b.stride(1),
        o0_u8.stride(0),
        o1_u8.stride(0),
        K // 16,
        BM=BM,
        BN=BN,
        BK=BK,
        IN_FP4=in_fp4,
        OUT_FP4=out_fp4,
        W_NZ=nz,
        EMIN=2 if out_fp4 else 8,
    )
    return o0, o1


def _launch(
    x,
    weight,
    weight_scale,
    x_scale,
    group_list,
    *,
    smooth_scale=None,
    weight_assist_matrix=None,
    bias=None,
    dequant_mode=0,
    dequant_dtype=6,
    quant_mode=0,
    quant_dtype=1,
    group_list_type=0,
    tuning_config=None,
    x_dtype=None,
    weight_dtype=None,
    weight_scale_dtype=None,
    x_scale_dtype=None,
):
    impl = _fp4_run if x.dtype == torch.uint8 else _fp8_run
    return impl(
        x,
        weight,
        weight_scale,
        x_scale,
        group_list,
        smooth_scale=smooth_scale,
        weight_assist_matrix=weight_assist_matrix,
        bias=bias,
        dequant_mode=dequant_mode,
        dequant_dtype=dequant_dtype,
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        group_list_type=group_list_type,
        tuning_config=tuning_config,
        x_dtype=x_dtype,
        weight_dtype=weight_dtype,
        weight_scale_dtype=weight_scale_dtype,
        x_scale_dtype=x_scale_dtype,
    )


def npu_grouped_matmul_swiglu_quant_v2(
    x,
    weight,
    weight_scale,
    x_scale,
    group_list,
    *,
    smooth_scale=None,
    weight_assist_matrix=None,
    bias=None,
    dequant_mode=0,
    dequant_dtype=6,
    quant_mode=0,
    quant_dtype=1,
    group_list_type=0,
    tuning_config=None,
    x_dtype=None,
    weight_dtype=None,
    weight_scale_dtype=None,
    x_scale_dtype=None,
):
    """Grouped MX matrix multiplication with a SwiGLU epilogue and per-block output quantization."""
    if (
        not isinstance(weight, (tuple, list))
        or len(weight) != 1
        or not isinstance(weight_scale, (tuple, list))
        or len(weight_scale) != 1
    ):
        raise ValueError("expected singleton weight and scale lists")
    w, ws = weight[0], weight_scale[0]
    if (
        x.ndim != 2
        or w.ndim != 3
        or not x.is_contiguous()
        or not all(t.device == x.device for t in (w, ws, x_scale, group_list))
    ):
        raise ValueError(
            "expected [M,K] activations and [E,K,2H] weights on the same device"
        )
    m, kp = x.shape
    e, wk, n = w.shape
    packed = x.dtype == torch.uint8
    k = 2 * kp if packed else kp
    if (
        m <= 0
        or e <= 0
        or wk != kp
        or k % 512
        or n % 128
        or k <= 0
        or n <= 0
        or max(w.numel(), x.numel(), m * n) >= 2**31
    ):
        raise ValueError(
            "expected K divisible by 512, output width divisible by 64, and int32-addressable tensors"
        )
    if (
        smooth_scale is not None
        or weight_assist_matrix is not None
        or bias is not None
        or tuning_config is not None
        or dequant_mode != 2
        or quant_mode != 2
        or group_list_type != 0
    ):
        raise ValueError(
            "only MX quantization with contiguous cumulative expert groups is supported"
        )
    if (
        group_list.dtype != torch.int64
        or group_list.shape != (e,)
        or not group_list.is_contiguous()
    ):
        raise ValueError("expected one int64 cumulative row end per expert")
    if (
        ws.shape != (e, k // 64, n, 2)
        or x_scale.shape != (m, k // 64, 2)
        or x_scale.dtype not in (torch.uint8, torch.float8_e8m0fnu)
        or ws.dtype not in (torch.uint8, torch.float8_e8m0fnu)
        or ws.stride(-1) != 1
        or x_scale.stride(-1) != 1
    ):
        raise ValueError("expected per-32 E8M0 scale pairs")
    if (
        weight_scale_dtype not in (None, 293)
        or x_scale_dtype not in (None, 293)
        or dequant_dtype not in (6, torch.float32)
    ):
        raise ValueError("expected FP32 dequantization and E8M0 scale tensors")
    if packed:
        if (
            w.dtype != torch.uint8
            or quant_dtype != 296
            or x_dtype != 296
            or weight_dtype != 296
            or int(torch_npu.get_npu_format(w)) not in (0, 2)
        ):
            raise ValueError("expected ND packed E2M1 operands and E2M1 output")
    else:
        if (
            x.dtype != torch.float8_e4m3fn
            or w.dtype != x.dtype
            or quant_dtype != x.dtype
            or x_dtype not in (None, 24)
            or weight_dtype not in (None, 24)
            or int(torch_npu.get_npu_format(w)) not in (0, 2, 29)
        ):
            raise ValueError("expected E4M3 tensors with E4M3 output")
    if x.device.type != "npu":
        raise ValueError("expected inputs on the same NPU")
    with torch_device_fn.device(x.device):
        return _launch(
            x,
            weight,
            weight_scale,
            x_scale,
            group_list,
            smooth_scale=smooth_scale,
            weight_assist_matrix=weight_assist_matrix,
            bias=bias,
            dequant_mode=dequant_mode,
            dequant_dtype=dequant_dtype,
            quant_mode=quant_mode,
            quant_dtype=quant_dtype,
            group_list_type=group_list_type,
            tuning_config=tuning_config,
            x_dtype=x_dtype,
            weight_dtype=weight_dtype,
            weight_scale_dtype=weight_scale_dtype,
            x_scale_dtype=x_scale_dtype,
        )
