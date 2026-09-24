# Copyright 2026, The FlagOS Contributors.
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.ops._dyn_quant_matmul_4bit import _quantize_rows
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _w4a8_reduce(
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
    BN: tl.constexpr,  # Output columns per reduction tile.
    BK: tl.constexpr,  # Input features per reduction tile.
):
    tiles_n = tl.cdiv(N, BN)
    groups: tl.constexpr = K // GROUP
    weight_count: tl.constexpr = N * (K // 2)
    ns_base = tl.arange(0, BN)
    ks = tl.arange(0, BK)
    for tile in range(tl.program_id(0), M * tiles_n, tl.num_programs(0)):
        row = (tile // tiles_n).to(tl.int64)
        ns = (tile % tiles_n * BN + ns_base).to(tl.int64)
        offset = tl.load(Offset + row)
        total = tl.zeros((BN,), tl.float32)
        for group in range(groups):
            acc = tl.zeros((BN,), tl.int32)
            for chunk in range(tl.cdiv(GROUP, BK)):
                local_k = chunk * BK + ks
                k = group * GROUP + local_k
                valid_k = local_k < GROUP
                q = tl.load(Q + row * K + k, valid_k, 0).to(tl.int32)
                centered = tl.where(valid_k, q + offset, 0)
                packed = tl.load(
                    Packed + ns[:, None] * (K // 2) + k[None, :] // 2,
                    (ns[:, None] < N) & valid_k[None, :],
                    0,
                ).to(tl.int32)
                w = ((packed >> ((k[None, :] % 2) * 4)) & 15) - 8
                # CoreX 4.4 rejects the shared-to-MMA layout produced by this
                # fused unpack path. Integer multiply and reduction preserve the
                # same W4A8 accumulator without using a native operator.
                acc += tl.sum(centered[None, :] * w, 1)
            ws = tl.load(Packed + weight_count + ns * groups + group, ns < N, 0)
            total += acc.to(tl.float32) * ws
        scale = tl.load(Scale + row)
        total = total * scale
        if HAS_BIAS:
            bias = tl.load(Packed + weight_count + N * groups + ns, ns < N, 0)
            total += bias
        tl.store(Y + row * N + ns, total, ns < N)


def _dyn_quant_matmul_4bit(inp, packed_weights, block_size, in_features, out_features):
    """W4A8 linear using the FP32 portable _dyn_quant_pack_4bit_weight ABI.

    The contiguous buffer contains N*K/2 numeric byte values, N*(K/block_size)
    row-major scales, then optionally N bias values. Even K is the low nibble;
    subtract eight from each nibble to obtain signed INT4. CPU KleidiAI opaque
    buffers are incompatible and must be repacked on the target device.
    """
    if not torch.compiler.is_compiling():
        logger.debug("GEMS_ILUVATAR _DYN_QUANT_MATMUL_4BIT")
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
    bn, bk = 8, 128
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
        _w4a8_reduce[(min(m * triton.cdiv(out_features, bn), 65535),)](
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
            bn,
            bk,
            enable_fp_fusion=False,
        )
    return output
