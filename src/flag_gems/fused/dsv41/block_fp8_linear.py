# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Block-scaled E4M3 linear: native FP8 or explicit BF16 computation.

Weights remain in E4M3 byte storage in all modes. The emulation kernel decodes
only a K tile; it never creates a full BF16 weight copy. Numeric FP32 scales are
required. UE8M0 checkpoint bytes must be decoded by the caller.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _round_e4m3(x):
    ax = tl.minimum(tl.abs(x), 448.0)
    exponent = tl.floor(tl.log2(tl.maximum(ax, 0.015625)))
    step = tl.exp2(exponent - 3.0)
    units = ax / step
    low = tl.floor(units)
    frac = units - low
    inc = (frac > 0.5) | ((frac == 0.5) & ((low.to(tl.int32) & 1) != 0))
    value = (low + inc.to(tl.float32)) * step
    sign = x.to(tl.uint32, bitcast=True) & 0x80000000
    return (value.to(tl.uint32, bitcast=True) | sign).to(tl.float32, bitcast=True)


@triton.jit
def _quantize(
    X,
    Q,
    S,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    POW2: tl.constexpr,
    EMULATE: tl.constexpr,
):
    group = tl.program_id(0)
    offsets = group * GROUP + tl.arange(0, GROUP)
    x = tl.load(X + offsets).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0), 1.0e-10) / 448.0
    if POW2:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    q = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)
    if EMULATE:
        q = _round_e4m3(q)
    tl.store(Q + offsets, q)
    tl.store(S + group, scale)


@triton.jit
def _quantize_groups(
    X,
    Q,
    S,
    NUM_GROUPS: tl.constexpr,
    GROUP: tl.constexpr,
    POW2: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    """Amortize block scheduling across independent quantization groups."""
    groups = tl.program_id(0) * GROUPS_PER_BLOCK + tl.arange(0, GROUPS_PER_BLOCK)
    offsets = groups[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    x = tl.load(X + offsets, groups[:, None] < NUM_GROUPS, 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 1), 1.0e-10) / 448.0
    if POW2:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    q = tl.minimum(tl.maximum(x / scale[:, None], -448.0), 448.0)
    tl.store(Q + offsets, q, groups[:, None] < NUM_GROUPS)
    tl.store(S + groups, scale, groups < NUM_GROUPS)


def _quantize_group_config(num_groups):
    # H100 / FlagTree 3.7 replay measurements: enough work per block without
    # sacrificing parallelism on the small decode projections.
    if num_groups >= 32768:
        return 32, 1
    if num_groups >= 8192:
        return 64, 4
    return 32, 4


@triton.jit
def _decode_e4m3(bits):
    exponent = ((bits >> 3) & 15).to(tl.int32)
    mantissa = (bits & 7).to(tl.float32)
    value = tl.where(
        exponent == 0,
        mantissa * 0.001953125,
        (1.0 + mantissa / 8.0) * tl.exp2(exponent - 7.0),
    )
    value = tl.where((bits & 127) == 127, float("nan"), value)
    return tl.where((bits & 128) != 0, -value, value)


@triton.jit
def _bf16_matmul(
    A,
    W,
    AS,
    WS,
    O,
    M,
    N,
    K: tl.constexpr,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    QUANTIZED: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, GROUP_K)
    acc = tl.zeros((BM, BN), tl.float32)
    for group in range(tl.cdiv(K, GROUP_K)):
        k = group * GROUP_K + kk
        a = tl.load(A + rows[:, None] * K + k[None, :], rows[:, None] < M, 0)
        b = tl.load(W + cols[None, :] * K + k[:, None], cols[None, :] < N, 0)
        b = _decode_e4m3(b)
        ws = tl.load(WS + (cols // GROUP_N) * (K // GROUP_K) + group, cols < N, 0)
        if QUANTIZED:
            sa = tl.load(AS + rows * (K // GROUP_K) + group, rows < M, 0)
            dot = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
            acc += dot * sa[:, None] * ws[None, :]
        else:
            # BF16 mode intentionally changes activation and weight rounding.
            b = (b * ws[None, :]).to(tl.bfloat16)
            acc += tl.dot(a.to(tl.bfloat16), b)
    tl.store(
        O + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def block_fp8_linear(
    input,
    weight,
    block_size,
    weight_scale,
    input_scale=None,
    bias=None,
    act_scale_ue8m0=False,
    *,
    mode="quantized",
    native_fp8=False,
):
    if mode not in ("quantized", "bf16"):
        raise ValueError(f"Unknown DSV4.1 quantization mode: {mode}")
    if len(block_size) != 2 or min(block_size) < 1:
        raise ValueError("Expected positive (block_n, block_k)")
    bn, bk = block_size
    if bk < 32 or bk & (bk - 1):
        raise ValueError("block_k must be a power of two >= 32")
    n, k = weight.shape
    if input.shape[-1] != k or k == 0 or k % bk:
        raise ValueError("K must be positive and divisible by block_k")
    if weight.dtype not in (torch.float8_e4m3fn, torch.uint8):
        raise TypeError("Weights must contain E4M3FN storage")
    if not weight_scale.is_floating_point():
        raise TypeError("Expected numeric scales, not UE8M0 exponent bytes")
    if tuple(weight_scale.shape) != (triton.cdiv(n, bn), k // bk):
        raise ValueError("Invalid weight scale layout")
    rows = input.reshape(-1, k).contiguous()
    m = rows.shape[0]
    out_dtype = torch.bfloat16 if input_scale is not None else input.dtype
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("Unquantized input must use BF16, FP16 or FP32")
    if input_scale is not None and mode != "quantized":
        raise ValueError("BF16 mode does not accept pre-quantized activations")
    out_shape = (*input.shape[:-1], n)
    if m == 0 or n == 0:
        return torch.empty(out_shape, dtype=out_dtype, device=input.device)
    ws = weight_scale.float().contiguous()
    w = weight.contiguous()
    if mode == "quantized":
        if input_scale is None:
            dtype = torch.float8_e4m3fn if native_fp8 else torch.bfloat16
            q = torch.empty_like(rows, dtype=dtype)
            scales = torch.empty((m, k // bk), dtype=torch.float32, device=input.device)
            if (
                native_fp8
                and bk == 32
                and torch.cuda.get_device_capability(input.device) == (9, 0)
            ):
                groups = m * (k // bk)
                groups_per_block, warps = _quantize_group_config(groups)
                _quantize_groups[(triton.cdiv(groups, groups_per_block),)](
                    rows,
                    q,
                    scales,
                    groups,
                    bk,
                    act_scale_ue8m0,
                    groups_per_block,
                    num_warps=warps,
                )
            else:
                _quantize[(m * (k // bk),)](
                    rows, q, scales, k, bk, act_scale_ue8m0, not native_fp8, num_warps=1
                )
        else:
            if (
                input.dtype != torch.float8_e4m3fn
                or not input_scale.is_floating_point()
            ):
                raise TypeError(
                    "Pre-quantized input requires E4M3FN and numeric scales"
                )
            if tuple(input_scale.shape) != (m, k // bk):
                raise ValueError("Invalid input scale layout")
            q = rows if native_fp8 else rows.to(torch.bfloat16)
            scales = input_scale.float().contiguous()
        if native_fp8:
            from flag_gems.ops.w8a8_block_fp8_matmul import w8a8_block_fp8_matmul

            out = w8a8_block_fp8_matmul(
                q,
                w.view(torch.float8_e4m3fn),
                scales,
                ws,
                block_size,
                output_dtype=out_dtype,
            )
        else:
            out = torch.empty((m, n), device=input.device, dtype=out_dtype)
            _bf16_matmul[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
                q,
                w.view(torch.uint8),
                scales,
                ws,
                out,
                m,
                n,
                k,
                bn,
                bk,
                True,
                16,
                32,
                num_warps=4,
            )
    else:
        out = torch.empty((m, n), device=input.device, dtype=out_dtype)
        _bf16_matmul[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
            rows,
            w.view(torch.uint8),
            ws,
            ws,
            out,
            m,
            n,
            k,
            bn,
            bk,
            False,
            16,
            32,
            num_warps=4,
        )
    if bias is not None:
        out += bias
    return out.reshape(out_shape)
