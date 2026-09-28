# Copyright 2026, The FlagOS Contributors.
# SPDX-License-Identifier: Apache-2.0

import logging

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
    # Ascend cannot lower scalar precise_divf. Correct ordinary division with
    # an explicit FMA residual to retain quantization boundary accuracy.
    q = a / b
    residual = tl.fma(-q, b, a)
    corrected = tl.fma(residual, 1.0 / b, q)
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
        rmin = tl.min(lo, 0)[None]
        rmax = tl.max(hi, 0)[None]
        multiplier = tl.where(
            rmin == rmax, 1.0, _quantization_divide(255.0, rmax - rmin)
        )
        scale = tl.where(multiplier != 0.0, _quantization_divide(1.0, multiplier), 0.0)
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
):
    tiles_n = tl.cdiv(N, BN)
    groups: tl.constexpr = K // GROUP
    weight_count: tl.constexpr = N * (K // 2)
    for tile in range(tl.program_id(0), tl.cdiv(M, BM) * tiles_n, tl.num_programs(0)):
        rows = (tile // tiles_n * BM + tl.arange(0, BM)).to(tl.int64)
        ns = (tile % tiles_n * BN + tl.arange(0, BN)).to(tl.int64)
        ks = tl.arange(0, BK)
        offset = tl.load(Offset + rows, rows < M, 0)
        total = tl.zeros((BM, BN), tl.float32)
        for group in range(groups):
            acc = tl.zeros((BM, BN), tl.int32)
            for chunk in range(tl.cdiv(GROUP, BK)):
                local_k = chunk * BK + ks
                k = group * GROUP + local_k
                valid_k = local_k < GROUP
                q = tl.load(
                    Q + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & valid_k[None, :],
                    0,
                ).to(tl.int32)
                centered = tl.where(valid_k[None, :], q + offset[:, None], 0)
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
            ws = tl.load(Packed + weight_count + ns * groups + group, ns < N, 0)
            total += acc.to(tl.float32) * ws[None, :]
        scale = tl.load(Scale + rows, rows < M, 0)
        total = total * scale[:, None]
        if HAS_BIAS:
            bias = tl.load(Packed + weight_count + N * groups + ns, ns < N, 0)
            total += bias[None, :]
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
    if torch.compiler.is_compiling():
        return _matmul_graph(inp, packed_weights, block_size, in_features, out_features)
    logger.debug("GEMS_ASCEND _DYN_QUANT_MATMUL_4BIT")
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
    bm, bn, bk = 16, 32, 32
    with torch_device_fn.device(inp.device):
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
            enable_fp_fusion=False,
        )
    return output


# Ascend uses an explicit graph operator for fullgraph capture. Its device
# implementation continues to launch the same two Triton kernels above,
# without tracing into the vendor JIT or depending on its version.
_matmul_graph = torch.library.custom_op(
    "flag_gems::_dyn_quant_matmul_4bit_ascend",
    _dyn_quant_matmul_4bit,
    mutates_args=(),
    device_types="npu",
    schema="(Tensor inp, Tensor packed_weights, int block_size, int in_features, int out_features) -> Tensor",
)


@_matmul_graph.register_fake
def _matmul_graph_fake(inp, packed_weights, block_size, in_features, out_features):
    return torch.empty((inp.shape[0], out_features), dtype=inp.dtype, device=inp.device)
