# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Clipped SwiGLU plus E4M3 row quantization with explicit numeric callbacks.

Input rows are [gate, up]. A contiguous integer vector identifies live rows
by 0 <= route_id < num_experts. Invalid rows are left untouched, including
caller-owned output padding. Callers retain their original calc/finalize/
conversion JIT functions; this module has no Humming or SGLang dependency.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _r10_activation(gate, up):
    gate = tl.minimum(gate, 10.0)
    up = tl.maximum(tl.minimum(up, 10.0), -10.0)
    gate_f32 = gate.to(tl.float32)
    activated = gate_f32 * tl.sigmoid(gate_f32)
    return (activated * up).to(tl.bfloat16)


@triton.jit
def _activation_quant_row(
    GateUp,
    Quant,
    Scales,
    row,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CALC_SCALE: tl.constexpr,
    FINALIZE_SCALE: tl.constexpr,
    QUANT_TENSOR: tl.constexpr,
):
    c = tl.arange(0, BLOCK_N)
    in_range = c < N
    gate = tl.load(GateUp + row * (2 * N) + c, mask=in_range, other=0.0)
    up = tl.load(GateUp + row * (2 * N) + N + c, mask=in_range, other=0.0)
    # Exact R10 contract: asymmetric gate clamp, symmetric up clamp,
    # FP32 SiLU and multiply, then the original BF16 materialization rounding.
    act_bf16 = _r10_activation(gate, up)
    act_f32 = act_bf16.to(tl.float32)
    # Masked tail lanes are exactly zero. Reuse the pinned native quant math.
    act_f32 = tl.where(in_range, act_f32, 0.0)
    scale = CALC_SCALE(act_f32, "float8e4m3")
    scale, stored_scale = FINALIZE_SCALE(scale, "float32", 1.0, 1.0)
    inv_scale = 1.0 / scale
    q = QUANT_TENSOR(act_f32 * inv_scale, "float8e4m3")
    tl.store(Quant + row * N + c, q, mask=in_range)
    tl.store(Scales + row, stored_scale)


@triton.jit
def _fused_rows(
    GateUp,
    Quant,
    Scales,
    TopIds,
    M,
    N: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CALC_SCALE: tl.constexpr,
    FINALIZE_SCALE: tl.constexpr,
    QUANT_TENSOR: tl.constexpr,
):
    row = tl.program_id(0)
    expert = tl.load(TopIds + row)
    if (expert >= 0) & (expert < NUM_EXPERTS):
        _activation_quant_row(
            GateUp,
            Quant,
            Scales,
            row,
            N,
            BLOCK_N,
            CALC_SCALE,
            FINALIZE_SCALE,
            QUANT_TENSOR,
        )


def swiglu_row_quant(
    gate_up,
    route_ids,
    quanted,
    scales,
    *,
    num_experts,
    calc_scale,
    finalize_scale,
    quant_tensor,
):
    if gate_up.ndim != 2 or gate_up.shape[1] % 2:
        raise ValueError("expected [rows, 2*width] gate/up matrix")
    rows, twice_width = gate_up.shape
    width = twice_width // 2
    if not (
        0 < rows and 0 < width <= 8192 and type(num_experts) is int and num_experts > 0
    ):
        raise ValueError("invalid row, feature or expert count")
    if not (
        gate_up.dtype == torch.bfloat16
        and quanted.dtype == torch.float8_e4m3fn
        and scales.dtype == torch.float32
        and route_ids.dtype in (torch.int32, torch.int64)
    ):
        raise TypeError(
            "expected BF16 input, integer routes, E4M3 output and FP32 scales"
        )
    if (
        route_ids.numel() != rows
        or quanted.shape != (rows, width)
        or scales.shape != (rows, 1)
    ):
        raise ValueError("incompatible physical row shapes")
    tensors = (gate_up, route_ids, quanted, scales)
    if not gate_up.is_cuda or any(
        t.device != gate_up.device or not t.is_contiguous() for t in tensors
    ):
        raise ValueError("expected contiguous tensors on one CUDA device")
    intervals = []
    for tensor in tensors:
        low = tensor.data_ptr()
        high = low + tensor.numel() * tensor.element_size()
        if any(max(low, a) < min(high, b) for a, b in intervals):
            raise ValueError("input, route and output buffers overlap")
        intervals.append((low, high))
    if any(
        not isinstance(f, triton.runtime.jit.JITFunction)
        for f in (calc_scale, finalize_scale, quant_tensor)
    ):
        raise TypeError("numeric callbacks must be Triton JIT functions")
    _fused_rows[(rows,)](
        gate_up,
        quanted,
        scales,
        route_ids,
        rows,
        N=width,
        NUM_EXPERTS=num_experts,
        BLOCK_N=triton.next_power_of_2(width),
        CALC_SCALE=calc_scale,
        FINALIZE_SCALE=finalize_scale,
        QUANT_TENSOR=quant_tensor,
        num_warps=4,
        num_stages=1,
    )
    return quanted, scales
