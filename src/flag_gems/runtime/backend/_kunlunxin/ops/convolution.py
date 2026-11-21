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

"""Kunlunxin override of ``aten::convolution``.

The generic implicit-GEMM kernel does a masked, div/mod (non-affine) gather of
the ``a`` operand inside a ``tl.dot`` kernel. On XPU3 ``tl.dot`` silently
miscompiles for both masked and non-affine (div/mod) loads (see
``patterns/xpu3-dot-partial-tile-miscompile``), which is why the generic op
returns wrong values here.

This override reformulates the forward / transposed convolution as an explicit
im2col gather (a non-dot kernel, where div/mod and masks are fine) followed by a
clean affine ``tl.dot`` matmul whose K/N dimensions are memory-padded to tile
multiples. The matmul operands are fully affine and unmasked.
"""

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _im2col_forward_kernel(
    x_ptr,
    col_ptr,
    KDIM,
    OUT0,
    OUT1,
    OUT2,
    IN0,
    IN1,
    IN2,
    S0,
    S1,
    S2,
    P0,
    P1,
    P2,
    DL0,
    DL1,
    DL2,
    KVOL,
    K12,
    K2,
    x_ns,
    x_cs,
    x_0s,
    x_1s,
    x_2s,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off // KDIM
    k = off % KDIM
    o2 = m % OUT2
    t1 = m // OUT2
    o1 = t1 % OUT1
    t0 = t1 // OUT1
    o0 = t0 % OUT0
    bidx = t0 // OUT0
    ci = k // KVOL
    rem = k % KVOL
    ka = rem // K12
    rem2 = rem % K12
    kb = rem2 // K2
    kc = rem2 % K2
    i0 = o0 * S0 - P0 + ka * DL0
    i1 = o1 * S1 - P1 + kb * DL1
    i2 = o2 * S2 - P2 + kc * DL2
    valid = (i0 >= 0) & (i0 < IN0) & (i1 >= 0) & (i1 < IN1) & (i2 >= 0) & (i2 < IN2)
    aoff = bidx * x_ns + ci * x_cs + i0 * x_0s + i1 * x_1s + i2 * x_2s
    val = tl.load(x_ptr + aoff, mask=valid, other=0.0)
    tl.store(col_ptr + off, val)


@libentry()
@triton.jit
def _im2col_transposed_kernel(
    x_ptr,
    col_ptr,
    KDIM,
    OUT0,
    OUT1,
    OUT2,
    IN0,
    IN1,
    IN2,
    S0,
    S1,
    S2,
    P0,
    P1,
    P2,
    DL0,
    DL1,
    DL2,
    KVOL,
    K12,
    K2,
    x_ns,
    x_cs,
    x_0s,
    x_1s,
    x_2s,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off // KDIM
    k = off % KDIM
    o2 = m % OUT2
    t1 = m // OUT2
    o1 = t1 % OUT1
    t0 = t1 // OUT1
    o0 = t0 % OUT0
    bidx = t0 // OUT0
    ci = k // KVOL
    rem = k % KVOL
    ka = rem // K12
    rem2 = rem % K12
    kb = rem2 // K2
    kc = rem2 % K2
    # transposed: input position i satisfies o = i*S - P + k*DL
    n0 = o0 + P0 - ka * DL0
    n1 = o1 + P1 - kb * DL1
    n2 = o2 + P2 - kc * DL2
    i0 = n0 // S0
    i1 = n1 // S1
    i2 = n2 // S2
    valid = (
        (n0 % S0 == 0)
        & (n1 % S1 == 0)
        & (n2 % S2 == 0)
        & (i0 >= 0)
        & (i0 < IN0)
        & (i1 >= 0)
        & (i1 < IN1)
        & (i2 >= 0)
        & (i2 < IN2)
    )
    aoff = bidx * x_ns + ci * x_cs + i0 * x_0s + i1 * x_1s + i2 * x_2s
    val = tl.load(x_ptr + aoff, mask=valid, other=0.0)
    tl.store(col_ptr + off, val)


@libentry()
@triton.jit
def _mm_kernel(A, B, C, K, USE_FP16: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ks = k + tl.arange(0, BK)
        a = tl.load(A + om[:, None] * K + ks[None, :])
        b = tl.load(B + ks[:, None] * BN + on[None, :])
        if USE_FP16:
            # fp16 tensor-core (FMA) path: faster, and fp16 has enough mantissa
            # to match the fp16 reference in the convolution accuracy tests.
            acc = tl.dot(a, b, acc)
        else:
            acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="ieee")
    tl.store(C + om[:, None] * BN + on[None, :], acc)


def _gv(v, i, d):
    if v is None:
        return d
    if isinstance(v, (int, float, bool)):
        return int(v)
    v = list(v)
    if i >= len(v):
        return d
    if len(v) == 1:
        return int(v[0]) if i == 0 else d
    return int(v[i])


def convolution(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    x = input
    w = weight
    nd = x.dim() - 2
    is_transposed = bool(transposed)
    g = int(groups)

    # ---- validation (mirrors the generic op, matching its error messages) ----
    if x.dim() != w.dim():
        raise RuntimeError(
            "Expected %d-dimensional input for %d-dimensional weight %s, but "
            "got %d-dimensional input of size %s instead"
            % (w.dim(), w.dim(), list(w.shape), x.dim(), list(x.shape))
        )
    if nd < 1 or nd > 3:
        raise RuntimeError(
            "Expected 3D, 4D or 5D input to convolution, but got input of size %s"
            % (list(x.shape),)
        )
    if g <= 0:
        raise RuntimeError("non-positive groups is not supported")
    if int(w.shape[0]) < 1:
        raise RuntimeError(
            "Given groups=%d, expected weight to be at least 1 at dimension 0, "
            "but got weight of size %s instead" % (g, list(w.shape))
        )
    if is_transposed:
        expected_in = int(w.shape[0])
    else:
        expected_in = int(w.shape[1]) * g
    if int(x.shape[1]) != expected_in:
        raise RuntimeError(
            "Given %sgroups=%d, weight of size %s, expected input%s to have %d "
            "channels, but got %d channels instead"
            % (
                "transposed " if is_transposed else "",
                g,
                list(w.shape),
                list(x.shape),
                expected_in,
                int(x.shape[1]),
            )
        )
    out_channels = int(w.shape[1]) * g if is_transposed else int(w.shape[0])
    if out_channels % g != 0:
        raise RuntimeError(
            "Given groups=%d, expected weight to be divisible by %d, but got "
            "output channels of %d" % (g, g, out_channels)
        )
    if bias is not None and bias.numel() > 0:
        if bias.dim() != 1 or int(bias.shape[0]) != out_channels:
            raise RuntimeError(
                "Given weight of size %s, expected bias to be 1-dimensional "
                "with %d elements, but got bias of size %s instead"
                % (list(w.shape), out_channels, list(bias.shape))
            )

    for i in range(nd):
        if _gv(stride, i, 1) <= 0:
            raise RuntimeError("non-positive stride is not supported")
        if _gv(dilation, i, 1) <= 0:
            raise ValueError("Expected positive dilation in convolution.")
        if _gv(padding, i, 0) < 0:
            raise RuntimeError("negative padding is not supported")
        if _gv(output_padding, i, 0) < 0:
            raise RuntimeError("negative output_padding is not supported")
    if is_transposed:
        for i in range(nd):
            op = _gv(output_padding, i, 0)
            if op >= _gv(stride, i, 1) and op >= _gv(dilation, i, 1):
                raise RuntimeError(
                    "output padding must be smaller than either stride or "
                    "dilation, but got output_padding=%d, stride=%d, dilation=%d"
                    % (op, _gv(stride, i, 1), _gv(dilation, i, 1))
                )

    C_in = int(x.shape[1])
    C_in_pg = C_in // g
    C_out = out_channels
    C_out_pg = C_out // g

    S = [_gv(stride, i, 1) for i in range(3)]
    P = [_gv(padding, i, 0) for i in range(3)]
    DL = [_gv(dilation, i, 1) for i in range(3)]
    IN = [int(x.shape[2 + i]) if i < nd else 1 for i in range(3)]
    K = [int(w.shape[2 + i]) if i < nd else 1 for i in range(3)]
    OP = [_gv(output_padding, i, 0) for i in range(3)]

    OUT = []
    for i in range(3):
        if is_transposed:
            o = (IN[i] - 1) * S[i] - 2 * P[i] + DL[i] * (K[i] - 1) + 1 + OP[i]
        else:
            o = (IN[i] + 2 * P[i] - DL[i] * (K[i] - 1) - 1) // S[i] + 1
        OUT.append(int(o))
    if not is_transposed and min(OUT[:nd]) <= 0:
        padded = [IN[i] + 2 * P[i] for i in range(nd)]
        eff_k = [DL[i] * (K[i] - 1) + 1 for i in range(nd)]
        raise RuntimeError(
            "Calculated padded input size per channel: (%s). Kernel size: (%s). "
            "Kernel size can't be greater than actual input size"
            % (
                " x ".join(str(v) for v in padded),
                " x ".join(str(v) for v in eff_k),
            )
        )

    n_ = int(x.shape[0])
    KVOL = K[0] * K[1] * K[2]
    K12 = K[1] * K[2]
    K2 = K[2]
    KDIM = C_in * KVOL  # all groups (channels), block-diagonal weight handles groups
    M = n_ * OUT[0] * OUT[1] * OUT[2]
    if M <= 0 or C_out <= 0 or KDIM <= 0:
        return torch.empty((n_, C_out, *OUT[:nd]), device=x.device, dtype=x.dtype)

    xc = x.contiguous()

    # ---- explicit im2col: (M, KDIM) ----
    col = torch.empty(M, KDIM, device=x.device, dtype=x.dtype)
    BLOCK = 1024
    grid = ((M * KDIM + BLOCK - 1) // BLOCK,)
    with torch_device_fn.device(x.device):
        if is_transposed:
            _im2col_transposed_kernel[grid](
                xc, col, KDIM, OUT[0], OUT[1], OUT[2], IN[0], IN[1], IN[2],
                S[0], S[1], S[2], P[0], P[1], P[2], DL[0], DL[1], DL[2], KVOL, K12, K2,
                xc.stride(0), xc.stride(1), xc.stride(2),
                xc.stride(3) if nd >= 2 else 0, xc.stride(4) if nd >= 3 else 0,
                BLOCK=BLOCK,
            )
        else:
            _im2col_forward_kernel[grid](
                xc, col, KDIM, OUT[0], OUT[1], OUT[2], IN[0], IN[1], IN[2],
                S[0], S[1], S[2], P[0], P[1], P[2], DL[0], DL[1], DL[2], KVOL, K12, K2,
                xc.stride(0), xc.stride(1), xc.stride(2),
                xc.stride(3) if nd >= 2 else 0, xc.stride(4) if nd >= 3 else 0,
                BLOCK=BLOCK,
            )

    # ---- block-diagonal weight: (KDIM, C_out) = (K, N) ----
    wc = w.contiguous()
    wbd = torch.zeros(KDIM, C_out, device=x.device, dtype=x.dtype)
    for gi in range(g):
        if is_transposed:
            # weight layout (C_in, C_out_pg, K); B[(ci,k), oc] = w[ci, oc, k]
            wbd[
                gi * C_in_pg * KVOL:(gi + 1) * C_in_pg * KVOL,
                gi * C_out_pg:(gi + 1) * C_out_pg,
            ] = (
                wc[gi * C_in_pg:(gi + 1) * C_in_pg]
                .reshape(C_in_pg, C_out_pg, KVOL)
                .permute(0, 2, 1)
                .reshape(C_in_pg * KVOL, C_out_pg)
            )
        else:
            # weight layout (C_out, C_in_pg, K); B[(ci,k), oc] = w[oc, ci, k]
            wbd[
                gi * C_in_pg * KVOL:(gi + 1) * C_in_pg * KVOL,
                gi * C_out_pg:(gi + 1) * C_out_pg,
            ] = wc[gi * C_out_pg:(gi + 1) * C_out_pg].reshape(C_out_pg, C_in_pg * KVOL).T

    # ---- padded matmul ----
    BM, BN, BK = 64, 16, 32
    Kpad = ((KDIM + BK - 1) // BK) * BK
    Npad = ((C_out + BN - 1) // BN) * BN
    Mpad = ((M + BM - 1) // BM) * BM
    col_p = torch.zeros(Mpad, Kpad, device=x.device, dtype=x.dtype)
    col_p[:M, :KDIM] = col
    w_p = torch.zeros(Kpad, Npad, device=x.device, dtype=x.dtype)
    w_p[:KDIM, :C_out] = wbd
    out_p = torch.empty(Mpad, Npad, device=x.device, dtype=torch.float32)
    with torch_device_fn.device(x.device):
        _mm_kernel[((Mpad // BM), (Npad // BN))](
            col_p, w_p, out_p, Kpad, USE_FP16=(x.dtype == torch.float16), BM=BM, BN=BN, BK=BK
        )

    out = out_p[:M, :C_out].to(x.dtype).contiguous().reshape(n_, *OUT[:nd], C_out).permute(
        0, nd + 1, *range(1, nd + 1)
    )

    if bias is not None and bias.numel() > 0:
        # The reference rounds the conv result to the output dtype, then adds
        # the bias in that same dtype (a second rounding). Add in the output
        # dtype (not fp32) so low-precision roundings match exactly.
        bshape = (1, -1) + (1,) * nd
        out = out.to(x.dtype)
        out = (out + bias.reshape(bshape).to(x.dtype)).to(x.dtype)
    return out
