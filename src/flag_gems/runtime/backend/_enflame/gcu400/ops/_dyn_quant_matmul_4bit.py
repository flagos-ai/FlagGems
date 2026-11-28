# Copyright 2026, The FlagOS Contributors.
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _quantization_divide(
    a,  # Dividend.
    b,  # Divisor.
):
    # Some vendor div_rn lowerings use an approximate reciprocal. Recover the
    # residual with an explicit FMA before rounding the corrected quotient.
    q = tl.div_rn(a, b)
    residual = tl.fma(-q, b, a)
    corrected = tl.fma(residual, tl.div_rn(1.0, b), q)
    # A finite input row can have an overflowing range. Preserve a/inf == 0
    # instead of replacing it with the NaN residual from 0*inf.
    return tl.where(residual == residual, corrected, q)


@triton.jit
def _quantize_rows(
    X,  # Input activations [M, K].
    Q,  # Quantized INT8 activations [M, K].
    Scale,  # Activation dequantization scale per row.
    Offset,  # Negative activation zero point per row.
    M,  # Number of input rows.
    K: tl.constexpr,  # Number of input features.
    SM: tl.constexpr,  # Input row stride, in elements.
    SK: tl.constexpr,  # Input feature stride, in elements.
    B: tl.constexpr,  # Features processed per quantization tile.
):
    # GCU400: a scalar (1,)-shaped tl.fma lowers to the non-PIC extern
    # tops_fma_fp32, whose ICALL relocation is rejected by ld.lld (P03). The
    # per-row correction chain therefore runs on a lane-carrying (R,) vessel:
    # staged min/max reductions keep lanes provably equal only at runtime, so
    # the chain stays vectorized (tops_fma_v2_fp32 links fine) while every lane
    # computes the identical corrected quotient. Staging is exact because
    # min/max are associative, and lane extraction via a second exact
    # reduction preserves the scalar results bit-for-bit.
    C: tl.constexpr = 32
    R: tl.constexpr = 256
    S: tl.constexpr = B // C
    cols = tl.arange(0, B)
    for row in range(tl.program_id(0), M, tl.num_programs(0)):
        row64 = row.to(tl.int64)
        lo = tl.full((B,), 0.0, tl.float32)
        hi = tl.full((B,), 0.0, tl.float32)
        for start in range(tl.cdiv(K, B)):
            k = start * B + cols
            x = tl.load(X + row64 * SM + k.to(tl.int64) * SK, k < K, 0).to(tl.float32)
            lo = tl.minimum(lo, x)
            hi = tl.maximum(hi, x)
        part_lo = tl.min(tl.reshape(lo, (S, C)), 0)
        part_hi = tl.max(tl.reshape(hi, (S, C)), 0)
        rmin_v = tl.min(tl.broadcast_to(part_lo[None], (R, C)), 1)
        rmax_v = tl.max(tl.broadcast_to(part_hi[None], (R, C)), 1)
        mult_v = tl.where(
            rmin_v == rmax_v,
            1.0,
            _quantization_divide(tl.full((R,), 255.0, tl.float32), rmax_v - rmin_v),
        )
        multiplier = tl.min(mult_v, 0)[None]
        scale_v = tl.where(
            mult_v != 0.0,
            _quantization_divide(tl.full((R,), 1.0, tl.float32), mult_v),
            0.0,
        )
        scale = tl.min(scale_v, 0)[None]
        rmin = tl.min(part_lo, 0)[None]
        rmax = tl.max(part_hi, 0)[None]

        scaled_min = rmin * multiplier
        scaled_max = rmax * multiplier
        zp = tl.where(
            (-128.0 + scaled_min) + (127.0 + scaled_max) > 0.0,
            -128.0 - scaled_min,
            127.0 - scaled_max,
        )
        zp = tl.minimum(tl.maximum(zp, -128.0), 127.0)
        # lrintf: nearest, ties to even. Activation values use round, ties away.
        lower = tl.floor(zp)
        frac = zp - lower
        zp_int = lower.to(tl.int32) + (
            (frac > 0.5) | ((frac == 0.5) & ((lower.to(tl.int32) & 1) != 0))
        ).to(tl.int32)
        tl.store(Scale + row64 + tl.arange(0, 1), scale)
        tl.store(Offset + row64 + tl.arange(0, 1), -zp_int)
        for start in range(tl.cdiv(K, B)):
            k = start * B + cols
            x = tl.load(X + row64 * SM + k.to(tl.int64) * SK, k < K, 0).to(tl.float32)
            v = x * multiplier
            rounded = tl.floor(tl.abs(v) + 0.5)
            rounded = tl.where(v < 0, -rounded, rounded).to(tl.int32)
            q = tl.minimum(tl.maximum(rounded + zp_int, -128), 127)
            tl.store(Q + row64 * K + k, q.to(tl.int8), k < K)


@triton.jit
def _w4a8_matmul(
    Q,  # Quantized INT8 activations [M, K].
    Scale,  # Activation dequantization scale per row.
    Offset,  # Negative activation zero point per row.
    Packed,  # Portable weights, weight scales, and optional bias.
    Y,  # Output matrix [M, N].
    M,  # Number of input rows.
    N: tl.constexpr,  # Number of output features.
    K: tl.constexpr,  # Number of input features.
    GROUP: tl.constexpr,  # Input features per weight scale group.
    HAS_BIAS: tl.constexpr,  # Whether the packed buffer includes bias.
    BM: tl.constexpr,  # Output rows per matrix tile.
    BN: tl.constexpr,  # Output columns per matrix tile.
    BK: tl.constexpr,  # Reduction features per dot tile.
    JOIN: tl.constexpr,  # Unpack each packed byte once via join+permute.
):
    # GCU400 adaptations (P03/P09 family):
    # 1. The generic nested group/chunk loops miscompile once BOTH loops run at
    #    least twice (2D loop-carried accumulators silently corrupt; P09), and a
    #    dedicated single-group branch PassManager-crashes for BM=32 tiles. The
    #    flattened (group, chunk) loop with exact where-based end-of-group
    #    apply/reset compiles and stays bit-exact for every configuration.
    # 2. GCU contracts fp32 mul+add into fma even under enable_fp_fusion=False,
    #    which skips the CPU-reference's product rounding. Every product that
    #    the reference rounds before adding is materialized with tl.fma(x, y,
    #    0.0) (a single-rounding multiply LLVM cannot re-contract), restoring
    #    the reference's double rounding bit-for-bit.
    # 3. JOIN loads each packed byte once (halving weight-segment traffic for
    #    wide tiles) and rewrites the nibble pair as adjacent k rows via
    #    join+permute+reshape; values are identical to the k-indexed unpack.
    tiles_n = tl.cdiv(N, BN)
    groups: tl.constexpr = K // GROUP
    weight_count: tl.constexpr = N * (K // 2)
    ks = tl.arange(0, BK)
    hks = tl.arange(0, BK // 2)
    CHUNKS: tl.constexpr = (GROUP + BK - 1) // BK
    for tile in range(tl.program_id(0), tl.cdiv(M, BM) * tiles_n, tl.num_programs(0)):
        rows = (tile // tiles_n * BM + tl.arange(0, BM)).to(tl.int64)
        ns = (tile % tiles_n * BN + tl.arange(0, BN)).to(tl.int64)
        offset = tl.load(Offset + rows, rows < M, 0)
        total = tl.zeros((BM, BN), tl.float32)
        acc = tl.zeros((BM, BN), tl.int32)
        for t in range(groups * CHUNKS):
            group = t // CHUNKS
            chunk = t % CHUNKS
            local_k = chunk * BK + ks
            k = group * GROUP + local_k
            valid_k = local_k < GROUP
            q = tl.load(
                Q + rows[:, None] * K + k[None, :],
                (rows[:, None] < M) & valid_k[None, :],
                0,
            ).to(tl.int32)
            centered = tl.where(valid_k[None, :], q + offset[:, None], 0)
            if JOIN:
                local_hk = chunk * (BK // 2) + hks
                hk = group * ((GROUP + 1) // 2) + local_hk
                valid_hk = local_hk < ((GROUP + 1) // 2)
                packed = tl.load(
                    Packed + ns[None, :] * (K // 2) + hk[:, None],
                    (ns[None, :] < N) & valid_hk[:, None],
                    0,
                ).to(tl.int32)
                w = tl.permute(
                    tl.join((packed & 15) - 8, (packed >> 4) - 8), (0, 2, 1)
                ).reshape(BK, BN)
            else:
                packed = tl.load(
                    Packed + ns[None, :] * (K // 2) + (k[:, None] // 2),
                    (ns[None, :] < N) & valid_k[:, None],
                    0,
                ).to(tl.int32)
                w = ((packed >> ((k[:, None] % 2) * 4)) & 15) - 8
            w = tl.where(valid_k[:, None], w, 0)
            # These integers are exactly representable in BF16. Each short dot
            # is exact in FP32; accumulate chunks in INT32, including long K.
            partial = tl.dot(centered.to(tl.bfloat16), w.to(tl.bfloat16))
            acc += partial.to(tl.int32)
            last = chunk == CHUNKS - 1
            ws = tl.load(Packed + weight_count + ns * groups + group, ns < N, 0)
            sel = tl.where(last, acc.to(tl.float32), 0.0)
            total = total + tl.fma(sel, ws[None, :], 0.0)
            acc = tl.where(last, 0, acc)
        scale = tl.load(Scale + rows, rows < M, 0)
        total = tl.fma(total, scale[:, None], 0.0)
        if HAS_BIAS:
            bias = tl.load(Packed + weight_count + N * groups + ns, ns < N, 0)
            total = total + bias[None, :]
        tl.store(
            Y + rows[:, None] * N + ns[None, :],
            total,
            (rows[:, None] < M) & (ns[None, :] < N),
        )


def _dyn_quant_matmul_4bit(inp, packed_weights, block_size, in_features, out_features):
    """W4A8 linear using the FP32 portable _dyn_quant_pack_4bit_weight ABI.

    The contiguous buffer contains N*K/2 numeric byte values, N*(K/block_size)
    row-major scales, then optionally N bias values. Even K is the low nibble;
    subtract eight from each nibble to obtain signed INT4. CPU KleidiAI opaque
    buffers are incompatible and must be repacked on the target device.
    """
    if not torch.compiler.is_compiling():
        logger.debug("GEMS_ENFLAME _DYN_QUANT_MATMUL_4BIT")
    if inp.ndim != 2:
        raise RuntimeError("inp must be two-dimensional")
    if inp.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError("inp must have float32 or bfloat16 dtype")
    if in_features <= 0 or in_features % 2:
        raise RuntimeError("in_features must be positive and even")
    if out_features < 0:
        raise RuntimeError("out_features must be nonnegative")
    if inp.shape[1] != in_features:
        raise RuntimeError("inp.size(1) must equal in_features")
    if block_size <= 0 or (
        block_size != in_features
        and (block_size % 32 != 0 or in_features % block_size != 0)
    ):
        raise RuntimeError(
            "block_size must equal in_features or divide it as a multiple of 32"
        )
    if inp.dtype == torch.bfloat16 and block_size != in_features:
        raise RuntimeError("bfloat16 requires block_size == in_features")
    if packed_weights.dtype == torch.uint8:
        raise RuntimeError(
            "opaque uint8 packed_weights are unsupported; repack on the target device with FlagGems"
        )
    if packed_weights.dtype != torch.float32:
        raise RuntimeError("packed_weights must use the float32 portable ABI")
    if packed_weights.device != inp.device:
        raise RuntimeError("packed_weights and inp must be on the same device")
    if packed_weights.ndim != 1 or not packed_weights.is_contiguous():
        raise RuntimeError("packed_weights must be one-dimensional and contiguous")
    base = out_features * (in_features // 2 + in_features // block_size)
    if packed_weights.numel() not in (base, base + out_features):
        raise RuntimeError(
            "packed_weights length does not match weights, scales, and optional bias"
        )
    m = inp.shape[0]
    output = torch.empty((m, out_features), dtype=inp.dtype, device=inp.device)
    if m == 0 or out_features == 0:
        return output
    q = torch.empty((m, in_features), dtype=torch.int8, device=inp.device)
    scale = torch.empty((m,), dtype=torch.float32, device=inp.device)
    offset = torch.empty((m,), dtype=torch.int32, device=inp.device)
    # Bounded vector tiles also support rows larger than a single program's limits.
    quant_block = max(32, min(triton.next_power_of_2(in_features), 1024))
    # GCU400 tile ladder, measured on the core shapes (bit-exact in each mode):
    # channelwise wants tall-narrow tiles at small N and a wide joined unpack
    # at large N; groupwise (GROUP always a multiple of 32) fits BK exactly.
    bm = 32
    if block_size == in_features:
        if out_features > 2048:
            bn, bk, join = 128, 128, True
        else:
            bn, bk, join = 32, 64, False
    else:
        if out_features > 2048:
            bn = 128
            bk = 32 if block_size == 32 else 64
            join = True
        else:
            bn = 32
            bk = 32 if block_size == 32 else 64
            join = False
    device_context = (
        nullcontext()
        if torch.compiler.is_compiling()
        else torch_device_fn.device(inp.device)
    )
    with device_context:
        _quantize_rows[(min(m, 65535),)](
            inp,
            q,
            scale,
            offset,
            m,
            in_features,
            inp.stride(0),
            inp.stride(1),
            quant_block,
            enable_fp_fusion=False,
        )
        _w4a8_matmul[(min(triton.cdiv(m, bm) * triton.cdiv(out_features, bn), 65535),)](
            q,
            scale,
            offset,
            packed_weights,
            output,
            m,
            out_features,
            in_features,
            block_size,
            packed_weights.numel() == base + out_features,
            bm,
            bn,
            bk,
            join,
            enable_fp_fusion=False,
        )
    return output
