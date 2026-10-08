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

"""Baseline ARM Triton kernels for packed W4A8-G128 linear layers."""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _round_to_nearest_even(value):
    """Standard RNE; LLVM selects the target's native rounding instruction."""
    return libdevice.rint(value)


@triton.jit
def _round_away_from_zero(value):
    """Match KAI/AArch64 ties-away rounding as a compiler-visible math op."""
    return libdevice.round(value.to(tl.float32)).to(tl.int32)


@triton.jit
def _quantize_kai_asymmetric_i8(values, quant_multiplier, zero_point):
    rounded = _round_away_from_zero(values.to(tl.float32) * quant_multiplier)
    shifted = rounded + zero_point.to(tl.int32)
    return tl.minimum(tl.maximum(shifted, -128), 127).to(tl.int8)


@triton.jit
def _q4_kai_asymmetric_qparams_from_minmax(row_min, row_max):
    """KleidiAI ``qai8dxp_f32`` row quantization parameters."""
    row_min = tl.minimum(row_min, 0.0)
    row_max = tl.maximum(row_max, 0.0)
    value_range = row_max - row_min
    quant_multiplier = tl.where(value_range == 0.0, 1.0, 255.0 / value_range)
    dequant_scale = tl.where(quant_multiplier == 0.0, 0.0, 1.0 / quant_multiplier)
    descaled_min = row_min * quant_multiplier
    descaled_max = row_max * quant_multiplier
    choose_min = -128.0 + descaled_min + 127.0 + descaled_max > 0.0
    zero_point = tl.where(choose_min, -128.0 - descaled_min, 127.0 - descaled_max)
    zero_point = tl.minimum(tl.maximum(zero_point, -128.0), 127.0)
    zero_point = _round_to_nearest_even(zero_point).to(tl.int8)
    return dequant_scale, quant_multiplier, zero_point


@triton.jit
def _q4_token_asymmetric_qparams_kai_f32(x_base, K: tl.constexpr):
    lanes = tl.arange(0, 16)
    row_min = tl.full((1,), 3.4028234663852886e38, tl.float32)
    row_max = tl.full((1,), -3.4028234663852886e38, tl.float32)
    for start in tl.range(0, K, 16, loop_unroll_factor=1):
        values = tl.load(x_base + start + lanes).to(tl.float32)
        row_min = tl.minimum(row_min, tl.min(values, axis=0))
        row_max = tl.maximum(row_max, tl.max(values, axis=0))
    return _q4_kai_asymmetric_qparams_from_minmax(row_min, row_max)


@triton.jit
def _q4_store_token_asymmetric_k32_kai(data_base, x_base, quant_multiplier, zero_point):
    lanes = tl.arange(0, 8)
    quant0 = _quantize_kai_asymmetric_i8(
        tl.load(x_base + lanes), quant_multiplier, zero_point
    )
    quant1 = _quantize_kai_asymmetric_i8(
        tl.load(x_base + 8 + lanes), quant_multiplier, zero_point
    )
    quant2 = _quantize_kai_asymmetric_i8(
        tl.load(x_base + 16 + lanes), quant_multiplier, zero_point
    )
    quant3 = _quantize_kai_asymmetric_i8(
        tl.load(x_base + 24 + lanes), quant_multiplier, zero_point
    )
    quant01 = tl.join(quant0, quant1).permute(1, 0).reshape((16,))
    quant23 = tl.join(quant2, quant3).permute(1, 0).reshape((16,))
    quantized = tl.join(quant01, quant23).permute(1, 0).reshape((32,))
    tl.store(data_base + tl.arange(0, 32), quantized)


@triton.jit
def _q4_decode_asym_g128_k32_lhs(lhs_data_ptr):
    """Load/rearrange one K32 activation once for one or more N4 tiles."""
    x_lanes = tl.arange(0, 8)
    x0 = tl.load(lhs_data_ptr + x_lanes).to(tl.int8).reshape((2, 4))
    x1 = tl.load(lhs_data_ptr + 8 + x_lanes).to(tl.int8).reshape((2, 4))
    x2 = tl.load(lhs_data_ptr + 16 + x_lanes).to(tl.int8).reshape((2, 4))
    x3 = tl.load(lhs_data_ptr + 24 + x_lanes).to(tl.int8).reshape((2, 4))
    x0 = tl.join(x0.reshape((8,)), x0.reshape((8,))).permute(1, 0).reshape((4, 4))
    x1 = tl.join(x1.reshape((8,)), x1.reshape((8,))).permute(1, 0).reshape((4, 4))
    x2 = tl.join(x2.reshape((8,)), x2.reshape((8,))).permute(1, 0).reshape((4, 4))
    x3 = tl.join(x3.reshape((8,)), x3.reshape((8,))).permute(1, 0).reshape((4, 4))
    return x0, x1, x2, x3


@triton.jit
def _q4_decode_asym_g128_k32_rhs_dot(rhs_data_ptr, x0, x1, x2, x3):
    """One compiler-visible packed-Q4 SDOT body with a reusable LHS."""
    q_lanes = tl.arange(0, 16)
    q0 = tl.load(rhs_data_ptr + q_lanes)
    q1 = tl.load(rhs_data_ptr + 16 + q_lanes)
    q2 = tl.load(rhs_data_ptr + 32 + q_lanes)
    q3 = tl.load(rhs_data_ptr + 48 + q_lanes)
    q0_low = (q0 << 4).to(tl.int8).reshape((4, 4))
    q1_low = (q1 << 4).to(tl.int8).reshape((4, 4))
    q2_low = (q2 << 4).to(tl.int8).reshape((4, 4))
    q3_low = (q3 << 4).to(tl.int8).reshape((4, 4))
    q0_high = (q0 & 0xF0).to(tl.int8).reshape((4, 4))
    q1_high = (q1 & 0xF0).to(tl.int8).reshape((4, 4))
    q2_high = (q2 & 0xF0).to(tl.int8).reshape((4, 4))
    q3_high = (q3 & 0xF0).to(tl.int8).reshape((4, 4))

    partial01 = tl.sum(q0_low.to(tl.int32) * x0.to(tl.int32), axis=1)
    partial23 = tl.sum(q1_low.to(tl.int32) * x0.to(tl.int32), axis=1)
    partial01 += tl.sum(q2_low.to(tl.int32) * x1.to(tl.int32), axis=1)
    partial23 += tl.sum(q3_low.to(tl.int32) * x1.to(tl.int32), axis=1)
    partial01 += tl.sum(q0_high.to(tl.int32) * x2.to(tl.int32), axis=1)
    partial23 += tl.sum(q1_high.to(tl.int32) * x2.to(tl.int32), axis=1)
    partial01 += tl.sum(q2_high.to(tl.int32) * x3.to(tl.int32), axis=1)
    partial23 += tl.sum(q3_high.to(tl.int32) * x3.to(tl.int32), axis=1)
    partial = tl.join(partial01, partial23).permute(1, 0).reshape((4, 2))
    return tl.sum(partial, axis=1)


@triton.jit
def _q4_decode_asym_g128_k32_dot(lhs_data_ptr, rhs_data_ptr):
    x0, x1, x2, x3 = _q4_decode_asym_g128_k32_lhs(lhs_data_ptr)
    return _q4_decode_asym_g128_k32_rhs_dot(rhs_data_ptr, x0, x1, x2, x3)


@triton.jit
def _q4_decode_asym_g128_sdot_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
):
    """G128 Q4 decode for partitioned, compact KleidiAI-style activations."""
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    tile_count: tl.constexpr = N // 4
    tiles_per_partition = (tile_count + partitions - 1) // partitions
    local_begin = partition * tiles_per_partition
    local_end = tl.minimum(tile_count, local_begin + tiles_per_partition)
    groups128: tl.constexpr = K // 128
    rhs_group_stride: tl.constexpr = 264
    rhs_tile_stride: tl.constexpr = groups128 * rhs_group_stride + 16
    output_lanes = tl.arange(0, 4)
    lhs_packed_ptr = lhs_packed_ptr.to(tl.pointer_type(tl.uint8))

    lhs_row_ptr = lhs_packed_ptr + (row * partitions + partition) * (8 + K)
    lhs_scale = tl.load(lhs_row_ptr.to(tl.pointer_type(tl.float32)))
    zero_point = tl.load(lhs_row_ptr + 4).to(tl.int8).to(tl.int32)

    for tile in range(local_begin, local_end):
        result = tl.zeros((4,), dtype=tl.float32)
        rhs_tile_ptr = rhs_packed_ptr + tile * rhs_tile_stride
        rhs_group_ptr = rhs_tile_ptr
        lhs_group_ptr = lhs_row_ptr + 8
        for group in tl.range(0, groups128, loop_unroll_factor=1):
            rhs_scale = tl.load(
                rhs_group_ptr.to(tl.pointer_type(tl.bfloat16)) + output_lanes
            ).to(tl.float32)
            dot_scaled16 = tl.zeros((4,), dtype=tl.int32)
            for subgroup in tl.range(0, 4, loop_unroll_factor=1):
                lhs_data_ptr = lhs_group_ptr + subgroup * 32
                dot_scaled16 += _q4_decode_asym_g128_k32_dot(
                    lhs_data_ptr,
                    rhs_group_ptr + 8 + subgroup * 64,
                )
            result += dot_scaled16.to(tl.float32) * rhs_scale
            lhs_group_ptr += 128
            rhs_group_ptr += rhs_group_stride

        weighted_sum_scaled16 = tl.load(
            (rhs_tile_ptr + groups128 * rhs_group_stride).to(
                tl.pointer_type(tl.float32)
            )
            + output_lanes
        )
        result -= zero_point.to(tl.float32) * weighted_sum_scaled16
        result *= lhs_scale * (1.0 / 16.0)

        output_offsets = row * N + tile * 4 + output_lanes
        tl.store(out_ptr + output_offsets, result.to(tl.bfloat16))


@triton.jit
def _q4_fused_decode_asym_g128_kai_sdot_kernel(
    x_ptr,
    workspace_ptr,
    rhs_packed_ptr,
    output_byte_offset,
    stride_xm,
    K: tl.constexpr,
    N: tl.constexpr,
):
    """KleidiAI-compatible FP32 activation pack plus Triton G128 GEMV."""
    row = tl.program_id(0)
    partition = tl.program_id(1)
    partitions = tl.num_programs(1)
    workspace_bytes = workspace_ptr.to(tl.pointer_type(tl.uint8))
    groups32: tl.constexpr = K // 32
    lhs_row_stride: tl.constexpr = 8 + K
    source_row = x_ptr + row * stride_xm
    scale, quant_multiplier, zero_point = _q4_token_asymmetric_qparams_kai_f32(
        source_row, K=K
    )
    scratch_base = (row * partitions + partition) * lhs_row_stride
    scratch_row = workspace_bytes + scratch_base
    tl.store(
        scratch_row.to(tl.pointer_type(tl.float32)) + tl.arange(0, 1),
        scale,
    )
    tl.store(scratch_row + 4 + tl.arange(0, 1), zero_point)
    for group in tl.range(0, groups32, loop_unroll_factor=1):
        _q4_store_token_asymmetric_k32_kai(
            scratch_row + 8 + group * 32,
            source_row + group * 32,
            quant_multiplier,
            zero_point,
        )
    out_ptr = (workspace_bytes + output_byte_offset).to(tl.pointer_type(tl.bfloat16))
    _q4_decode_asym_g128_sdot_kernel(
        workspace_bytes,
        rhs_packed_ptr,
        out_ptr,
        K=K,
        N=N,
    )


@triton.jit
def _pack_lhs_qai8dxp_asym_panel4_kernel(
    x_ptr,
    lhs_packed_ptr,
    M,
    stride_xm,
    K: tl.constexpr,
):
    """Panel4 pack with KleidiAI's FP32 asymmetric activation semantics."""
    panel = tl.program_id(0)
    rows = panel * 4 + tl.arange(0, 4)
    source_rows = tl.minimum(rows, M - 1)
    lanes16 = tl.arange(0, 16)
    row_min = tl.full((4,), 3.4028234663852886e38, tl.float32)
    row_max = tl.full((4,), -3.4028234663852886e38, tl.float32)
    for start in tl.range(0, K, 16, loop_unroll_factor=1):
        values = tl.load(
            x_ptr + source_rows[:, None] * stride_xm + start + lanes16[None, :]
        ).to(tl.float32)
        row_min = tl.minimum(row_min, tl.min(values, axis=1))
        row_max = tl.maximum(row_max, tl.max(values, axis=1))
    scales, multipliers, zero_points = _q4_kai_asymmetric_qparams_from_minmax(
        row_min, row_max
    )
    panel_stride: tl.constexpr = 32 + 4 * K
    panel_base = lhs_packed_ptr + panel * panel_stride
    tl.store(
        panel_base.to(tl.pointer_type(tl.float32)) + tl.arange(0, 4),
        scales,
    )
    tl.store(panel_base + 16 + tl.arange(0, 4), zero_points)

    lanes8 = tl.arange(0, 8)
    store_lanes = tl.arange(0, 32)
    groups32: tl.constexpr = K // 32
    data_base = panel_base + 32
    for group in tl.range(0, groups32, loop_unroll_factor=1):
        for offset in tl.static_range(0, 32, 8):
            values = tl.load(
                x_ptr
                + source_rows[:, None] * stride_xm
                + group * 32
                + offset
                + lanes8[None, :]
            )
            quantized = _quantize_kai_asymmetric_i8(
                values,
                multipliers[:, None],
                zero_points[:, None],
            )
            tl.store(
                data_base + group * 128 + offset * 4 + store_lanes,
                quantized.reshape((32,)),
            )


@triton.jit
def _q4_load_g128_k32_i8mm_weight(rhs_data_ptr):
    """Unpack one K32 slice while preserving the I8MM lowering algebra."""
    block_n: tl.constexpr = 4
    packed_flat = tl.load(rhs_data_ptr + tl.arange(0, 64))
    packed = (
        packed_flat.reshape((2, block_n, 8)).permute(0, 2, 1).reshape((16, block_n))
    )
    weight_low = (packed << 4).to(tl.int8)
    weight_high = (packed & 0xF0).to(tl.int8)
    return tl.join(weight_low, weight_high).permute(0, 2, 1).reshape((32, block_n))


@triton.jit
def _q4_load_g128_k32_i8mm_lhs(lhs_blob):
    """Load one panel4 K32 slice in the layout recognized by the dot pass."""
    lhs_seq = (
        tl.load(lhs_blob + tl.arange(0, 128))
        .to(tl.int8)
        .reshape((4, 4, 8))
        .permute(1, 0, 2)
        .reshape((4, 32))
    )
    return lhs_seq.reshape((4, 2, 16)).permute(0, 2, 1).reshape((4, 32))


@triton.jit
def _q4_prefill_asym_g128_i8mm_tile(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUPS128_RUNTIME,
    PID_M_OVERRIDE: tl.constexpr,
    PID_N_OVERRIDE: tl.constexpr,
):
    """G128 Q4 M4 prefill: four K32 SMMLA bodies per scale/correction."""
    pid_m = PID_M_OVERRIDE
    pid_n = PID_N_OVERRIDE
    block_n: tl.constexpr = 4
    groups128: tl.constexpr = K // 128
    cols = pid_n * block_n + tl.arange(0, block_n)
    lanes_m4 = tl.arange(0, 4)
    result0 = tl.zeros((4, block_n), tl.float32)
    lhs_panel_stride: tl.constexpr = 32 + 4 * K
    lhs_data_offset: tl.constexpr = 32
    lhs_zp_offset: tl.constexpr = 16
    lhs_row_base = lhs_packed_ptr + pid_m * lhs_panel_stride

    rhs_tile_stride: tl.constexpr = groups128 * 264 + 16
    rhs_tile_ptr = rhs_packed_ptr + pid_n * rhs_tile_stride

    # Keep the G128 reduction as tl.range rather than static_range.  The CPU
    # backend can then preserve a compact loop at production K sizes instead
    # of cloning the full M16 microkernel and spilling its accumulators.
    for group128 in tl.range(0, GROUPS128_RUNTIME, loop_unroll_factor=1):
        rhs_blob = rhs_tile_ptr + group128 * 264
        rhs_scale = tl.load(
            rhs_blob.to(tl.pointer_type(tl.bfloat16)) + tl.arange(0, 4)
        ).to(tl.float32)
        dot0 = tl.zeros((4, block_n), dtype=tl.int32)
        for subgroup in tl.range(0, 4, loop_unroll_factor=1):
            group32 = group128 * 4 + subgroup
            weight = _q4_load_g128_k32_i8mm_weight(rhs_blob + 8 + subgroup * 64)
            lhs0_blob = lhs_row_base + lhs_data_offset + group32 * 128
            dot0 += tl.dot(
                _q4_load_g128_k32_i8mm_lhs(lhs0_blob),
                weight,
                out_dtype=tl.int32,
            )

        result0 += dot0.to(tl.float32) * rhs_scale[None, :]

    # Correction values do not participate in the I8MM loop, so load them after
    # the accumulator reduction to keep the loop's live range compact.
    scale0 = tl.load(lhs_row_base.to(tl.pointer_type(tl.float32)) + lanes_m4)
    zp0 = tl.load(lhs_row_base + lhs_zp_offset + lanes_m4).to(tl.int8).to(tl.int32)

    weighted_sum_scaled16 = tl.load(
        (rhs_tile_ptr + groups128 * 264).to(tl.pointer_type(tl.float32))
        + tl.arange(0, 4)
    )
    result0 -= zp0[:, None].to(tl.float32) * weighted_sum_scaled16[None, :]
    result0 *= scale0[:, None] * (1.0 / 16.0)

    output_row = pid_m * 4
    tl.store(
        out_ptr + (output_row + lanes_m4)[:, None] * N + cols[None, :],
        result0.to(tl.bfloat16),
    )


@triton.jit
def _q4_prefill_asym_g128_i8mm_kernel(
    lhs_packed_ptr,
    rhs_packed_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    _q4_prefill_asym_g128_i8mm_tile(
        lhs_packed_ptr,
        rhs_packed_ptr,
        out_ptr,
        N=N,
        K=K,
        GROUPS128_RUNTIME=K // 128,
        PID_M_OVERRIDE=tl.program_id(0),
        PID_N_OVERRIDE=tl.program_id(1),
    )
