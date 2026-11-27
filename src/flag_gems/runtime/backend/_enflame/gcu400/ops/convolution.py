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

"""Triton implementation of the generic ATen ``convolution`` operator.

Covers forward and transposed convolution for 1D/2D/3D spatial ranks with
grouped channels.  Formulated as an implicit GEMM:

    M    = N * OUT0 * OUT1 * OUT2
    Kdim = (C_in / groups) * (K0 * K1 * K2)
    Ndim = C_out / groups

Program grid = (cdiv(M, BLOCK_M), cdiv(C_out_pg, BLOCK_N), groups).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _conv_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N,
    C_in_pg,
    C_out_pg,
    OUT0,
    OUT1,
    OUT2,
    IN0,
    IN1,
    IN2,
    KVOL,
    K12,
    K2,
    S0,
    S1,
    S2,
    P0,
    P1,
    P2,
    DL0,
    DL1,
    DL2,
    KDIM,
    x_ns,
    x_cs,
    x_0s,
    x_1s,
    x_2s,
    w_ic_s,
    w_oc_s,
    w_k0_s,
    w_k1_s,
    w_k2_s,
    out_ns,
    out_cs,
    out_0s,
    out_1s,
    out_2s,
    F32: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)

    M = N * OUT0 * OUT1 * OUT2

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # decode flattened output position m -> (n, o0, o1, o2)
    o2 = offs_m % OUT2
    t1 = offs_m // OUT2
    o1 = t1 % OUT1
    t0 = t1 // OUT1
    o0 = t0 % OUT0
    bidx = t0 // OUT0

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < C_out_pg
    co_g = pid_g * C_out_pg + offs_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_off = tl.arange(0, BLOCK_K)

    for kstart in range(0, KDIM, BLOCK_K):
        ks = kstart + k_off
        kmask = ks < KDIM

        # decode the reduction index into (input-channel-in-group, k0, k1, k2)
        ci_loc = ks // KVOL
        rem = ks % KVOL
        ka = rem // K12
        rem2 = rem % K12
        kb = rem2 // K2
        kc = rem2 % K2

        ci_g = pid_g * C_in_pg + ci_loc

        # A masked lane can still point past its channel block: GCU touches the
        # address of a masked load, so an unclamped index reads the
        # neighbouring allocation (which may hold NaN/Inf left in a freed
        # buffer) instead of the masked-off lanes.  Clamp the channel indices
        # the same way the spatial ones (i0/i1/i2) are clamped below.
        ci_loc_v = tl.where(kmask, ci_loc, 0)
        ci_g_v = tl.where(kmask, ci_g, 0)

        if TRANSPOSED:
            n0 = o0[:, None] + P0 - ka[None, :] * DL0
            i0 = n0 // S0
            v0 = (i0 * S0 == n0) & (i0 >= 0) & (i0 < IN0)
            n1 = o1[:, None] + P1 - kb[None, :] * DL1
            i1 = n1 // S1
            v1 = (i1 * S1 == n1) & (i1 >= 0) & (i1 < IN1)
            n2 = o2[:, None] + P2 - kc[None, :] * DL2
            i2 = n2 // S2
            v2 = (i2 * S2 == n2) & (i2 >= 0) & (i2 < IN2)
        else:
            i0 = o0[:, None] * S0 - P0 + ka[None, :] * DL0
            v0 = (i0 >= 0) & (i0 < IN0)
            i1 = o1[:, None] * S1 - P1 + kb[None, :] * DL1
            v1 = (i1 >= 0) & (i1 < IN1)
            i2 = o2[:, None] * S2 - P2 + kc[None, :] * DL2
            v2 = (i2 >= 0) & (i2 < IN2)

        i0 = tl.where(v0, i0, 0)
        i1 = tl.where(v1, i1, 0)
        i2 = tl.where(v2, i2, 0)

        amask = mask_m[:, None] & kmask[None, :] & v0 & v1 & v2
        aoff = (
            (bidx * x_ns)[:, None]
            + (ci_g_v * x_cs)[None, :]
            + i0 * x_0s
            + i1 * x_1s
            + i2 * x_2s
        )
        a = tl.load(x_ptr + aoff, mask=amask, other=0.0)

        if TRANSPOSED:
            icv = ci_g_v[:, None]
            ocv = tl.where(mask_n, offs_n, 0)[None, :]
        else:
            icv = ci_loc_v[:, None]
            ocv = tl.where(mask_n, co_g, 0)[None, :]

        woff = (
            icv * w_ic_s
            + ocv * w_oc_s
            + ka[:, None] * w_k0_s
            + kb[:, None] * w_k1_s
            + kc[:, None] * w_k2_s
        )
        bmask = kmask[:, None] & mask_n[None, :]
        b = tl.load(w_ptr + woff, mask=bmask, other=0.0)

        if F32:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc)

    if HAS_BIAS:
        # The reference rounds the convolution result to the output dtype and
        # only then adds the bias (a second rounding).  Reproduce that
        # round-trip instead of a single fused fp32 accumulate + bias + cast,
        # which is what makes bias-bearing reduced-precision workloads drift.
        bv = tl.load(b_ptr + tl.where(mask_n, co_g, 0), mask=mask_n, other=0.0).to(
            tl.float32
        )
        acc = acc.to(out_ptr.dtype.element_ty).to(tl.float32) + bv[None, :]

    ooff = (
        (bidx * out_ns)[:, None]
        + (co_g * out_cs)[None, :]
        + (o0 * out_0s)[:, None]
        + (o1 * out_1s)[:, None]
        + (o2 * out_2s)[:, None]
    )
    smask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + ooff, acc.to(out_ptr.dtype.element_ty), mask=smask)


def _next_pow2(v, lo, hi):
    """Clamp ``v`` up to a power of two inside ``[lo, hi]``."""
    p = lo
    while p < v and p < hi:
        p *= 2
    return p


# Below this many implicit-GEMM MACs the kernel retires inside its own launch
# latency, so the tiling cannot be the bottleneck and is left alone.
_COMPUTE_BOUND_MACS = 1 << 20


def _select_tiles(M, C_out_pg, KDIM, is_ieee_fp32, nd):
    """Pick implicit-GEMM tiles for an (M, C_out_pg, KDIM) problem.

    GCU400 rework (measured on Triton GCU 4.1.6, bench core shapes, fp16/fp32,
    per-shape sweep of BM/BN/BK x num_warps 1/2/4/8):

      * num_warps=1 wins everywhere. A single warp keeps the MMA pipeline
        ownership intact; num_warps>=2 splits the (BLOCK_M, BLOCK_K) operand
        tiles across warps, and on this backend that serializes the dot
        issue path instead of helping it (group 612us at w1 vs 7279us at w2;
        conv1d fp32 311us at w1 vs 4321us at w2).  The old table only reached
        w1 for 128x128 tiles, which left every 64-wide-N shape on w4 and an
        8-9x gap on the wide layers.
      * fp32 (ieee FMA, no MMA) splits by shape: plain 2-D/3-D layers prefer
        a shallower BM=64 tile -- more independent programs in flight beat a
        deeper per-program loop when there is no tensor-core pipeline to
        fill (stem fp32 7.3ms at (64,64,32) vs 15.5ms at (128,64,32)).
        Single-block-K (1x1) and 1-D fp32 keep BM=128 (1x1 429us vs 557us;
        conv1d 307us vs 422us).
      * 16-bit keeps BM=128 for 2-D layers but drops to BM=64 for the 3-D
        gather (three-deep tap decode makes the (128,*) address vectors the
        per-program bottleneck: conv3d 1562us at (64,64,32) vs 3251us at
        (128,64,32)).
      * BN is sized from the problem (pow2, floor 64 for MMA width, cap 128)
      and BK stays 32: BK=64 doubles the 2-D index/mask vectors and measured
      slower everywhere (stem 52ms vs 46ms at w4, 15.5ms vs 5.3ms at w1
      fp16); BK=16 won 1.06-1.22x in short probes but regressed up to 41%
      under the official bench protocol, so it is vetoed.
      * Tiny problems (launch-latency bound) sit on (32,16,32) w1: the old
        (64,16,32) w4 grid left the whole conv on one program (~44us); a
        32-row tile doubles the grid and a single warp runs the small
        pipeline for ~11-13us.
    """
    if M * C_out_pg * KDIM < _COMPUTE_BOUND_MACS:
        return 32, 16, 32, 1

    if is_ieee_fp32:
        # fp32 (ieee FMA, no MMA) regimes, all measured at w1:
        #  * tiny K (1x1 projection): the whole reduction is one BK=32 block,
        #    and the wider BM=128 tile keeps the per-program loop busy (1x1
        #    429us at (128,64,32) vs 557us at (64,64,32)).
        #  * 1-D: a wide BN=128 matches the 128-channel head (307us at
        #    (128,128,32) vs 422us at (64,128,32)).
        #  * plain 2-D/3-D: a shallower BM=64 tile -- more independent
        #    programs in flight beat a deeper per-program loop when there is
        #    no tensor-core pipeline to fill (stem fp32 7.3ms at (64,64,32)
        #    vs 15.3ms at (128,64,32)).
        if KDIM <= 64:
            return 128, 64, 32, 1
        if nd == 1:
            return 128, 128, 32, 1
        return 64, 64, 32, 1

    # 16-bit regimes.  BK=32: BK=64 doubles the 2-D index/mask vectors and
    # loses everywhere (stem 52ms vs 46ms at w4, 15.5ms vs 5.3ms at w1), and
    # BK=16 wins 1.06x in a probe yet regressed 41% on the 3-D gather under
    # the official bench protocol, so 32 stays.
    if nd == 3:
        return 64, 64, 32, 1
    return 128, _next_pow2(C_out_pg, 64, 128), 32, 1


def _gv(v, i, d):
    """Element-wise accessor for stride/padding/dilation-style arguments."""
    if v is None:
        return d
    if isinstance(v, (int, float, bool)):
        return int(v)
    v = list(v)
    if len(v) == 1:
        return int(v[0]) if i == 0 else d
    return int(v[i])


def convolution(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    logger.debug("GEMS_ENFLAME CONVOLUTION")
    x = input
    w = weight
    nd = x.dim() - 2
    is_transposed = bool(transposed)
    g = int(groups)

    # The input and the weight must describe the same spatial rank.  Taking the
    # rank from the input alone would silently run a 1-D convolution against a
    # 2-D weight and read the kernel's trailing extent as a stride.
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

    # A zero-sized weight has no output channel to write, which ATen reports
    # rather than returning an empty result.
    if int(w.shape[0]) < 1:
        raise RuntimeError(
            "Given groups=%d, expected weight to be at least 1 at dimension 0, "
            "but got weight of size %s instead" % (g, list(w.shape))
        )

    n_ = int(x.shape[0])
    C_in = int(x.shape[1])

    # ATen validates the channel bookkeeping before touching any data.  The
    # expected input-channel count is read off the weight, which is laid out as
    # (C_out, C_in/groups, *K) for a forward convolution and
    # (C_in, C_out/groups, *K) for a transposed one.
    if is_transposed:
        expected_in = int(w.shape[0])
    else:
        expected_in = int(w.shape[1]) * g
    if C_in != expected_in:
        raise RuntimeError(
            "Given %sgroups=%d, weight of size %s, expected input%s to have %d "
            "channels, but got %d channels instead"
            % (
                "transposed " if is_transposed else "",
                g,
                list(w.shape),
                list(x.shape),
                expected_in,
                C_in,
            )
        )
    out_channels = int(w.shape[1]) * g if is_transposed else int(w.shape[0])
    if out_channels % g != 0:
        raise RuntimeError(
            "Given groups=%d, expected weight to be divisible by %d, but got "
            "output channels of %d" % (g, g, out_channels)
        )

    if bias is not None and bias.numel() > 0:
        # A short bias would be read past its end by the kernel; a long one
        # means the caller mismatched the weight.
        if bias.dim() != 1 or int(bias.shape[0]) != out_channels:
            raise RuntimeError(
                "Given weight of size %s, expected bias to be 1-dimensional "
                "with %d elements, but got bias of size %s instead"
                % (list(w.shape), out_channels, list(bias.shape))
            )

    # A zero stride divides by zero in the output-extent formula and a
    # non-positive dilation collapses the kernel footprint; both are rejected
    # before any index arithmetic runs.
    for i in range(nd):
        if _gv(stride, i, 1) <= 0:
            raise RuntimeError("non-positive stride is not supported")
        if _gv(dilation, i, 1) <= 0:
            raise ValueError("Expected positive dilation in convolution.")
        if _gv(padding, i, 0) < 0:
            raise RuntimeError("negative padding is not supported")
        if _gv(output_padding, i, 0) < 0:
            raise RuntimeError("negative output_padding is not supported")

    # ATen requires the output padding to stay below the stride or the
    # dilation, otherwise the transposed output extent is ambiguous.
    if is_transposed:
        for i in range(nd):
            op = _gv(output_padding, i, 0)
            if op >= _gv(stride, i, 1) and op >= _gv(dilation, i, 1):
                raise RuntimeError(
                    "output padding must be smaller than either stride or "
                    "dilation, but got output_padding=%d, stride=%d, dilation=%d"
                    % (op, _gv(stride, i, 1), _gv(dilation, i, 1))
                )

    C_in_pg = C_in // g

    if nd == 1:
        S = [_gv(stride, 0, 1), 1, 1]
        P = [_gv(padding, 0, 0), 0, 0]
        DL = [_gv(dilation, 0, 1), 1, 1]
        OP = [_gv(output_padding, 0, 0), 0, 0]
        IN = [int(x.shape[2]), 1, 1]
        K = [int(w.shape[2]), 1, 1]
        x_0s, x_1s, x_2s = int(x.stride(2)), 0, 0
    elif nd == 2:
        S = [_gv(stride, 0, 1), _gv(stride, 1, 1), 1]
        P = [_gv(padding, 0, 0), _gv(padding, 1, 0), 0]
        DL = [_gv(dilation, 0, 1), _gv(dilation, 1, 1), 1]
        OP = [_gv(output_padding, 0, 0), _gv(output_padding, 1, 0), 0]
        IN = [int(x.shape[2]), int(x.shape[3]), 1]
        K = [int(w.shape[2]), int(w.shape[3]), 1]
        x_0s, x_1s, x_2s = int(x.stride(2)), int(x.stride(3)), 0
    else:
        S = [_gv(stride, 0, 1), _gv(stride, 1, 1), _gv(stride, 2, 1)]
        P = [_gv(padding, 0, 0), _gv(padding, 1, 0), _gv(padding, 2, 0)]
        DL = [_gv(dilation, 0, 1), _gv(dilation, 1, 1), _gv(dilation, 2, 1)]
        OP = [
            _gv(output_padding, 0, 0),
            _gv(output_padding, 1, 0),
            _gv(output_padding, 2, 0),
        ]
        IN = [int(x.shape[2]), int(x.shape[3]), int(x.shape[4])]
        K = [int(w.shape[2]), int(w.shape[3]), int(w.shape[4])]
        x_0s, x_1s, x_2s = int(x.stride(2)), int(x.stride(3)), int(x.stride(4))

    OUT = []
    for i in range(3):
        if is_transposed:
            o = (IN[i] - 1) * S[i] - 2 * P[i] + DL[i] * (K[i] - 1) + 1 + OP[i]
        else:
            o = (IN[i] + 2 * P[i] - DL[i] * (K[i] - 1) - 1) // S[i] + 1
        OUT.append(int(o))

    # A dilated kernel wider than the padded input gives a non-positive output
    # extent.  ATen reports it; allocating a negative-sized tensor would raise
    # an unrelated error from the allocator.
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

    if is_transposed:
        C_out_pg = int(w.shape[1])
        C_out = C_out_pg * g
    else:
        C_out = int(w.shape[0])
        C_out_pg = C_out // g

    if nd == 1:
        out = torch.empty((n_, C_out, OUT[0]), device=x.device, dtype=x.dtype)
        out_0s, out_1s, out_2s = int(out.stride(2)), 0, 0
    elif nd == 2:
        out = torch.empty((n_, C_out, OUT[0], OUT[1]), device=x.device, dtype=x.dtype)
        out_0s, out_1s, out_2s = int(out.stride(2)), int(out.stride(3)), 0
    else:
        out = torch.empty(
            (n_, C_out, OUT[0], OUT[1], OUT[2]), device=x.device, dtype=x.dtype
        )
        out_0s, out_1s, out_2s = (
            int(out.stride(2)),
            int(out.stride(3)),
            int(out.stride(4)),
        )

    KVOL = K[0] * K[1] * K[2]
    K12 = K[1] * K[2]
    KDIM = C_in_pg * KVOL

    M = n_ * OUT[0] * OUT[1] * OUT[2]
    if M <= 0 or C_out_pg <= 0 or KDIM <= 0 or C_out <= 0:
        return out

    if is_transposed:
        w_ic_s = int(w.stride(0))
        w_oc_s = int(w.stride(1))
    else:
        w_oc_s = int(w.stride(0))
        w_ic_s = int(w.stride(1))
    w_k0_s = int(w.stride(2))
    w_k1_s = int(w.stride(3)) if nd >= 2 else 0
    w_k2_s = int(w.stride(4)) if nd >= 3 else 0

    has_bias = bias is not None and getattr(bias, "numel", lambda: 0)() > 0
    b_ptr = bias if has_bias else x

    BM, BN, BK, num_warps = _select_tiles(
        M, C_out_pg, KDIM, x.dtype == torch.float32, nd
    )

    grid = (triton.cdiv(M, BM), triton.cdiv(C_out_pg, BN), g)
    _conv_kernel[grid](
        x,
        w,
        b_ptr,
        out,
        n_,
        C_in_pg,
        C_out_pg,
        OUT[0],
        OUT[1],
        OUT[2],
        IN[0],
        IN[1],
        IN[2],
        KVOL,
        K12,
        K[2],
        S[0],
        S[1],
        S[2],
        P[0],
        P[1],
        P[2],
        DL[0],
        DL[1],
        DL[2],
        KDIM,
        int(x.stride(0)),
        int(x.stride(1)),
        x_0s,
        x_1s,
        x_2s,
        w_ic_s,
        w_oc_s,
        w_k0_s,
        w_k1_s,
        w_k2_s,
        int(out.stride(0)),
        int(out.stride(1)),
        out_0s,
        out_1s,
        out_2s,
        F32=(x.dtype == torch.float32),
        HAS_BIAS=has_bias,
        TRANSPOSED=is_transposed,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=num_warps,
    )
    return out


__all__ = ["convolution"]
