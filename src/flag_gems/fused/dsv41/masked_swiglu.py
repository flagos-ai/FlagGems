# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Portions adapted from SGLang (Copyright 2023-2026 SGLang Team, Apache-2.0).
"""Masked clipped SwiGLU with an explicit caller-owned activation JIT helper."""

import torch
import triton
import triton.language as tl


@triton.jit
def _masked_compact(
    gateup_output,
    down_input,
    masked_m,
    CAPACITY: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROW_SPLITS: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    HAS_SWIGLU_LIMIT: tl.constexpr,
    ACTIVATE: tl.constexpr,
):
    # Static grid is E * ROW_SPLITS * ceil(I / BLOCK_SIZE), independent of
    # padded capacity and dynamic counts. Each slot walks one residue class.
    expert = tl.program_id(0)
    slot = tl.program_id(1)
    count = tl.load(masked_m + expert)
    tl.device_assert((count >= 0) & (count <= CAPACITY), "masked_m outside capacity")
    offset = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < INTERMEDIATE
    OutDtype = down_input.dtype.element_ty
    for row in tl.range(slot, count, ROW_SPLITS):
        pid = expert * CAPACITY + row
        gate_output_ptr = gateup_output + pid * (INTERMEDIATE * 2)
        up_output_ptr = gate_output_ptr + INTERMEDIATE
        down_input_ptr = down_input + pid * INTERMEDIATE
        gate_output = tl.load(gate_output_ptr + offset, mask=mask)
        up_output = tl.load(up_output_ptr + offset, mask=mask)
        if HAS_SWIGLU_LIMIT:
            gate_output = tl.minimum(gate_output, SWIGLU_LIMIT)
            up_output = tl.maximum(tl.minimum(up_output, SWIGLU_LIMIT), -SWIGLU_LIMIT)
        gate_output_activated = ACTIVATE(gate_output, "silu")
        act_mul_output = gate_output_activated * up_output
        act_mul_output = act_mul_output.to(OutDtype)
        tl.store(down_input_ptr + offset, act_mul_output, mask=mask)


def masked_swiglu(gate_up, down, counts, *, activate, swiglu_limit=10.0):
    if gate_up.ndim != 3 or gate_up.shape[-1] % 2:
        raise ValueError("expected [experts, capacity, 2*width] gate/up input")
    e, cap, twice_width = gate_up.shape
    width = twice_width // 2
    if gate_up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16:
        raise TypeError("expected BF16 input/output")
    if (
        down.shape != (e * cap, width)
        or counts.shape != (e,)
        or counts.dtype != torch.int32
    ):
        raise ValueError("incompatible physical output/count geometry")
    if not gate_up.is_cuda or any(
        t.device != gate_up.device or not t.is_contiguous()
        for t in (gate_up, down, counts)
    ):
        raise ValueError("expected contiguous tensors on one CUDA device")
    low, high = (
        gate_up.data_ptr(),
        gate_up.data_ptr() + gate_up.numel() * gate_up.element_size(),
    )
    out_low, out_high = (
        down.data_ptr(),
        down.data_ptr() + down.numel() * down.element_size(),
    )
    if max(low, out_low) < min(high, out_high):
        raise ValueError("activation input/output storage overlaps")
    if not isinstance(activate, triton.runtime.jit.JITFunction):
        raise TypeError("expected activation JIT callback")
    _masked_compact[(e, 16, triton.cdiv(width, 512))](
        gate_up,
        down,
        counts,
        CAPACITY=cap,
        INTERMEDIATE=width,
        BLOCK_SIZE=512,
        ROW_SPLITS=16,
        SWIGLU_LIMIT=swiglu_limit,
        HAS_SWIGLU_LIMIT=True,
        ACTIVATE=activate,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=True,
    )
    return down
