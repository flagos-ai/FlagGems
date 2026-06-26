import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.device_info import get_device_capability

if torch_device_fn.is_available() and get_device_capability() >= (9, 0):
    SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn
else:
    SUPPORTED_FP8_DTYPE = torch.float32


logger = logging.getLogger(__name__)


@triton.jit
def _per_token_group_quant_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min,
    fp8_max,
    inv_fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr += (row * y_row_stride) + (row_g_id * group_size)
    y_q_ptr += g_id * group_size
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax * inv_fp8_max

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))

    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    y_s_col_stride,
    eps,
    fp8_min,
    fp8_max,
    inv_fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    group_id = g_id % groups_per_row

    y_ptr += row * y_row_stride + group_id * group_size
    y_q_ptr += g_id * group_size
    y_s_ptr += group_id * y_s_col_stride + row

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax * inv_fp8_max

    if scale_ue8m0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s), 1e-10))))

    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_m2(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min,
    fp8_max,
    inv_fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size
    pid = tl.program_id(0)
    pairs_per_row = groups_per_row // 2
    row = pid // pairs_per_row
    pair_id = pid % pairs_per_row

    group0 = pair_id * 2
    group1 = group0 + 1

    g0 = row * groups_per_row + group0
    g1 = g0 + 1

    base = y_ptr + row * y_row_stride

    y_ptr0 = base + group0 * group_size
    y_ptr1 = base + group1 * group_size

    y_q_ptr0 = y_q_ptr + g0 * group_size
    y_q_ptr1 = y_q_ptr + g1 * group_size

    y_s_ptr0 = y_s_ptr + g0
    y_s_ptr1 = y_s_ptr + g1

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y0 = tl.load(y_ptr0 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = tl.load(y_ptr1 + cols, mask=mask, other=0.0).to(tl.float32)

    abs0 = tl.abs(y0)
    abs1 = tl.abs(y1)

    max0 = tl.max(abs0)
    max1 = tl.max(abs1)

    y_s0 = tl.maximum(max0, eps) * inv_fp8_max
    y_s1 = tl.maximum(max1, eps) * inv_fp8_max

    if scale_ue8m0:
        y_s0 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s0), 1e-10))))
        y_s1 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s1), 1e-10))))

    y_q0 = tl.clamp(y0 / y_s0, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q1 = tl.clamp(y1 / y_s1, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr0 + cols, y_q0, mask=mask)
    tl.store(y_s_ptr0, y_s0)
    tl.store(y_q_ptr1 + cols, y_q1, mask=mask)
    tl.store(y_s_ptr1, y_s1)


@triton.jit
def _per_token_group_quant_fp8_colmajor_m2(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    y_s_col_stride,
    eps,
    fp8_min,
    fp8_max,
    inv_fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size
    pid = tl.program_id(0)
    pairs_per_row = groups_per_row // 2
    row = pid // pairs_per_row
    pair_id = pid % pairs_per_row

    group0 = pair_id * 2
    group1 = group0 + 1

    g0 = row * groups_per_row + group0
    g1 = g0 + 1

    base = y_ptr + row * y_row_stride

    y_ptr0 = base + group0 * group_size
    y_ptr1 = base + group1 * group_size

    y_q_ptr0 = y_q_ptr + g0 * group_size
    y_q_ptr1 = y_q_ptr + g1 * group_size

    y_s_ptr0 = y_s_ptr + group0 * y_s_col_stride + row
    y_s_ptr1 = y_s_ptr + group1 * y_s_col_stride + row

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y0 = tl.load(y_ptr0 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = tl.load(y_ptr1 + cols, mask=mask, other=0.0).to(tl.float32)

    abs0 = tl.abs(y0)
    abs1 = tl.abs(y1)

    max0 = tl.max(abs0)
    max1 = tl.max(abs1)

    y_s0 = tl.maximum(max0, eps) * inv_fp8_max
    y_s1 = tl.maximum(max1, eps) * inv_fp8_max

    if scale_ue8m0:
        y_s0 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s0), 1e-10))))
        y_s1 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s1), 1e-10))))

    y_q0 = tl.clamp(y0 / y_s0, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q1 = tl.clamp(y1 / y_s1, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr0 + cols, y_q0, mask=mask)
    tl.store(y_s_ptr0, y_s0)
    tl.store(y_q_ptr1 + cols, y_q1, mask=mask)
    tl.store(y_s_ptr1, y_s1)


@triton.jit
def _per_token_group_quant_fp8_m4(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min,
    fp8_max,
    inv_fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size
    pid = tl.program_id(0)
    pairs_per_row = groups_per_row // 4
    row = pid // pairs_per_row
    pair_id = pid % pairs_per_row

    group0 = pair_id * 4
    group1 = group0 + 1
    group2 = group0 + 2
    group3 = group0 + 3

    g0 = row * groups_per_row + group0
    g1 = g0 + 1
    g2 = g1 + 1
    g3 = g2 + 1

    base = y_ptr + row * y_row_stride

    y_ptr0 = base + group0 * group_size
    y_ptr1 = base + group1 * group_size
    y_ptr2 = base + group2 * group_size
    y_ptr3 = base + group3 * group_size

    y_q_ptr0 = y_q_ptr + g0 * group_size
    y_q_ptr1 = y_q_ptr + g1 * group_size
    y_q_ptr2 = y_q_ptr + g2 * group_size
    y_q_ptr3 = y_q_ptr + g3 * group_size

    y_s_ptr0 = y_s_ptr + g0
    y_s_ptr1 = y_s_ptr + g1
    y_s_ptr2 = y_s_ptr + g2
    y_s_ptr3 = y_s_ptr + g3

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y0 = tl.load(y_ptr0 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = tl.load(y_ptr1 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = tl.load(y_ptr2 + cols, mask=mask, other=0.0).to(tl.float32)
    y3 = tl.load(y_ptr3 + cols, mask=mask, other=0.0).to(tl.float32)

    abs0 = tl.abs(y0)
    abs1 = tl.abs(y1)
    abs2 = tl.abs(y2)
    abs3 = tl.abs(y3)

    max0 = tl.max(abs0)
    max1 = tl.max(abs1)
    max2 = tl.max(abs2)
    max3 = tl.max(abs3)

    y_s0 = tl.maximum(max0, eps) * inv_fp8_max
    y_s1 = tl.maximum(max1, eps) * inv_fp8_max
    y_s2 = tl.maximum(max2, eps) * inv_fp8_max
    y_s3 = tl.maximum(max3, eps) * inv_fp8_max

    if scale_ue8m0:
        y_s0 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s0), 1e-10))))
        y_s1 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s1), 1e-10))))
        y_s2 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s2), 1e-10))))
        y_s3 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s3), 1e-10))))

    y_q0 = tl.clamp(y0 / y_s0, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q1 = tl.clamp(y1 / y_s1, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q2 = tl.clamp(y2 / y_s2, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q3 = tl.clamp(y3 / y_s3, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr0 + cols, y_q0, mask=mask)
    tl.store(y_s_ptr0, y_s0)
    tl.store(y_q_ptr1 + cols, y_q1, mask=mask)
    tl.store(y_s_ptr1, y_s1)
    tl.store(y_q_ptr2 + cols, y_q2, mask=mask)
    tl.store(y_s_ptr2, y_s2)
    tl.store(y_q_ptr3 + cols, y_q3, mask=mask)
    tl.store(y_s_ptr3, y_s3)


@triton.jit
def _per_token_group_quant_fp8_colmajor_m4(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    y_s_col_stride,
    eps,
    fp8_min,
    fp8_max,
    inv_fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size
    pid = tl.program_id(0)
    pairs_per_row = groups_per_row // 4
    row = pid // pairs_per_row
    pair_id = pid % pairs_per_row

    group0 = pair_id * 4
    group1 = group0 + 1
    group2 = group1 + 1
    group3 = group2 + 1

    g0 = row * groups_per_row + group0
    g1 = g0 + 1
    g2 = g1 + 1
    g3 = g2 + 1

    base = y_ptr + row * y_row_stride

    y_ptr0 = base + group0 * group_size
    y_ptr1 = base + group1 * group_size
    y_ptr2 = base + group2 * group_size
    y_ptr3 = base + group3 * group_size

    y_q_ptr0 = y_q_ptr + g0 * group_size
    y_q_ptr1 = y_q_ptr + g1 * group_size
    y_q_ptr2 = y_q_ptr + g2 * group_size
    y_q_ptr3 = y_q_ptr + g3 * group_size

    y_s_ptr0 = y_s_ptr + group0 * y_s_col_stride + row
    y_s_ptr1 = y_s_ptr + group1 * y_s_col_stride + row
    y_s_ptr2 = y_s_ptr + group2 * y_s_col_stride + row
    y_s_ptr3 = y_s_ptr + group3 * y_s_col_stride + row

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y0 = tl.load(y_ptr0 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = tl.load(y_ptr1 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = tl.load(y_ptr2 + cols, mask=mask, other=0.0).to(tl.float32)
    y3 = tl.load(y_ptr3 + cols, mask=mask, other=0.0).to(tl.float32)

    abs0 = tl.abs(y0)
    abs1 = tl.abs(y1)
    abs2 = tl.abs(y2)
    abs3 = tl.abs(y3)

    max0 = tl.max(abs0)
    max1 = tl.max(abs1)
    max2 = tl.max(abs2)
    max3 = tl.max(abs3)

    y_s0 = tl.maximum(max0, eps) * inv_fp8_max
    y_s1 = tl.maximum(max1, eps) * inv_fp8_max
    y_s2 = tl.maximum(max2, eps) * inv_fp8_max
    y_s3 = tl.maximum(max3, eps) * inv_fp8_max

    if scale_ue8m0:
        y_s0 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s0), 1e-10))))
        y_s1 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s1), 1e-10))))
        y_s2 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s2), 1e-10))))
        y_s3 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s3), 1e-10))))

    y_q0 = tl.clamp(y0 / y_s0, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q1 = tl.clamp(y1 / y_s1, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q2 = tl.clamp(y2 / y_s2, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q3 = tl.clamp(y3 / y_s3, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr0 + cols, y_q0, mask=mask)
    tl.store(y_s_ptr0, y_s0)
    tl.store(y_q_ptr1 + cols, y_q1, mask=mask)
    tl.store(y_s_ptr1, y_s1)
    tl.store(y_q_ptr2 + cols, y_q2, mask=mask)
    tl.store(y_s_ptr2, y_s2)
    tl.store(y_q_ptr3 + cols, y_q3, mask=mask)
    tl.store(y_s_ptr3, y_s3)


@triton.jit
def _per_token_group_quant_fp8_m8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min,
    fp8_max,
    inv_fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size
    pid = tl.program_id(0)
    pairs_per_row = groups_per_row // 8
    row = pid // pairs_per_row
    pair_id = pid % pairs_per_row

    group0 = pair_id * 8
    group1 = group0 + 1
    group2 = group0 + 2
    group3 = group0 + 3
    group4 = group0 + 4
    group5 = group0 + 5
    group6 = group0 + 6
    group7 = group0 + 7

    g0 = row * groups_per_row + group0
    g1 = g0 + 1
    g2 = g1 + 1
    g3 = g2 + 1
    g4 = g3 + 1
    g5 = g4 + 1
    g6 = g5 + 1
    g7 = g6 + 1

    base = y_ptr + row * y_row_stride

    y_ptr0 = base + group0 * group_size
    y_ptr1 = base + group1 * group_size
    y_ptr2 = base + group2 * group_size
    y_ptr3 = base + group3 * group_size
    y_ptr4 = base + group4 * group_size
    y_ptr5 = base + group5 * group_size
    y_ptr6 = base + group6 * group_size
    y_ptr7 = base + group7 * group_size

    y_q_ptr0 = y_q_ptr + g0 * group_size
    y_q_ptr1 = y_q_ptr + g1 * group_size
    y_q_ptr2 = y_q_ptr + g2 * group_size
    y_q_ptr3 = y_q_ptr + g3 * group_size
    y_q_ptr4 = y_q_ptr + g4 * group_size
    y_q_ptr5 = y_q_ptr + g5 * group_size
    y_q_ptr6 = y_q_ptr + g6 * group_size
    y_q_ptr7 = y_q_ptr + g7 * group_size

    y_s_ptr0 = y_s_ptr + g0
    y_s_ptr1 = y_s_ptr + g1
    y_s_ptr2 = y_s_ptr + g2
    y_s_ptr3 = y_s_ptr + g3
    y_s_ptr4 = y_s_ptr + g4
    y_s_ptr5 = y_s_ptr + g5
    y_s_ptr6 = y_s_ptr + g6
    y_s_ptr7 = y_s_ptr + g7

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y0 = tl.load(y_ptr0 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = tl.load(y_ptr1 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = tl.load(y_ptr2 + cols, mask=mask, other=0.0).to(tl.float32)
    y3 = tl.load(y_ptr3 + cols, mask=mask, other=0.0).to(tl.float32)
    y4 = tl.load(y_ptr4 + cols, mask=mask, other=0.0).to(tl.float32)
    y5 = tl.load(y_ptr5 + cols, mask=mask, other=0.0).to(tl.float32)
    y6 = tl.load(y_ptr6 + cols, mask=mask, other=0.0).to(tl.float32)
    y7 = tl.load(y_ptr7 + cols, mask=mask, other=0.0).to(tl.float32)

    abs0 = tl.abs(y0)
    abs1 = tl.abs(y1)
    abs2 = tl.abs(y2)
    abs3 = tl.abs(y3)
    abs4 = tl.abs(y4)
    abs5 = tl.abs(y5)
    abs6 = tl.abs(y6)
    abs7 = tl.abs(y7)

    max0 = tl.max(abs0)
    max1 = tl.max(abs1)
    max2 = tl.max(abs2)
    max3 = tl.max(abs3)
    max4 = tl.max(abs4)
    max5 = tl.max(abs5)
    max6 = tl.max(abs6)
    max7 = tl.max(abs7)
    y_s0 = tl.maximum(max0, eps) * inv_fp8_max
    y_s1 = tl.maximum(max1, eps) * inv_fp8_max
    y_s2 = tl.maximum(max2, eps) * inv_fp8_max
    y_s3 = tl.maximum(max3, eps) * inv_fp8_max
    y_s4 = tl.maximum(max4, eps) * inv_fp8_max
    y_s5 = tl.maximum(max5, eps) * inv_fp8_max
    y_s6 = tl.maximum(max6, eps) * inv_fp8_max
    y_s7 = tl.maximum(max7, eps) * inv_fp8_max

    if scale_ue8m0:
        y_s0 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s0), 1e-10))))
        y_s1 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s1), 1e-10))))
        y_s2 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s2), 1e-10))))
        y_s3 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s3), 1e-10))))
        y_s4 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s4), 1e-10))))
        y_s5 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s5), 1e-10))))
        y_s6 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s6), 1e-10))))
        y_s7 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s7), 1e-10))))

    y_q0 = tl.clamp(y0 / y_s0, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q1 = tl.clamp(y1 / y_s1, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q2 = tl.clamp(y2 / y_s2, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q3 = tl.clamp(y3 / y_s3, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q4 = tl.clamp(y4 / y_s4, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q5 = tl.clamp(y5 / y_s5, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q6 = tl.clamp(y6 / y_s6, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q7 = tl.clamp(y7 / y_s7, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr0 + cols, y_q0, mask=mask)
    tl.store(y_s_ptr0, y_s0)
    tl.store(y_q_ptr1 + cols, y_q1, mask=mask)
    tl.store(y_s_ptr1, y_s1)
    tl.store(y_q_ptr2 + cols, y_q2, mask=mask)
    tl.store(y_s_ptr2, y_s2)
    tl.store(y_q_ptr3 + cols, y_q3, mask=mask)
    tl.store(y_s_ptr3, y_s3)
    tl.store(y_q_ptr4 + cols, y_q4, mask=mask)
    tl.store(y_s_ptr4, y_s4)
    tl.store(y_q_ptr5 + cols, y_q5, mask=mask)
    tl.store(y_s_ptr5, y_s5)
    tl.store(y_q_ptr6 + cols, y_q6, mask=mask)
    tl.store(y_s_ptr6, y_s6)
    tl.store(y_q_ptr7 + cols, y_q7, mask=mask)
    tl.store(y_s_ptr7, y_s7)


@triton.jit
def _per_token_group_quant_fp8_colmajor_m8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    y_s_col_stride,
    eps,
    fp8_min,
    fp8_max,
    inv_fp8_max,
    scale_ue8m0,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size
    pid = tl.program_id(0)
    pairs_per_row = groups_per_row // 8
    row = pid // pairs_per_row
    pair_id = pid % pairs_per_row

    group0 = pair_id * 8
    group1 = group0 + 1
    group2 = group1 + 1
    group3 = group2 + 1
    group4 = group3 + 1
    group5 = group4 + 1
    group6 = group5 + 1
    group7 = group6 + 1

    g0 = row * groups_per_row + group0
    g1 = g0 + 1
    g2 = g1 + 1
    g3 = g2 + 1
    g4 = g3 + 1
    g5 = g4 + 1
    g6 = g5 + 1
    g7 = g6 + 1

    base = y_ptr + row * y_row_stride

    y_ptr0 = base + group0 * group_size
    y_ptr1 = base + group1 * group_size
    y_ptr2 = base + group2 * group_size
    y_ptr3 = base + group3 * group_size
    y_ptr4 = base + group4 * group_size
    y_ptr5 = base + group5 * group_size
    y_ptr6 = base + group6 * group_size
    y_ptr7 = base + group7 * group_size

    y_q_ptr0 = y_q_ptr + g0 * group_size
    y_q_ptr1 = y_q_ptr + g1 * group_size
    y_q_ptr2 = y_q_ptr + g2 * group_size
    y_q_ptr3 = y_q_ptr + g3 * group_size
    y_q_ptr4 = y_q_ptr + g4 * group_size
    y_q_ptr5 = y_q_ptr + g5 * group_size
    y_q_ptr6 = y_q_ptr + g6 * group_size
    y_q_ptr7 = y_q_ptr + g7 * group_size

    y_s_ptr0 = y_s_ptr + group0 * y_s_col_stride + row
    y_s_ptr1 = y_s_ptr + group1 * y_s_col_stride + row
    y_s_ptr2 = y_s_ptr + group2 * y_s_col_stride + row
    y_s_ptr3 = y_s_ptr + group3 * y_s_col_stride + row
    y_s_ptr4 = y_s_ptr + group4 * y_s_col_stride + row
    y_s_ptr5 = y_s_ptr + group5 * y_s_col_stride + row
    y_s_ptr6 = y_s_ptr + group6 * y_s_col_stride + row
    y_s_ptr7 = y_s_ptr + group7 * y_s_col_stride + row

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y0 = tl.load(y_ptr0 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = tl.load(y_ptr1 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = tl.load(y_ptr2 + cols, mask=mask, other=0.0).to(tl.float32)
    y3 = tl.load(y_ptr3 + cols, mask=mask, other=0.0).to(tl.float32)
    y4 = tl.load(y_ptr4 + cols, mask=mask, other=0.0).to(tl.float32)
    y5 = tl.load(y_ptr5 + cols, mask=mask, other=0.0).to(tl.float32)
    y6 = tl.load(y_ptr6 + cols, mask=mask, other=0.0).to(tl.float32)
    y7 = tl.load(y_ptr7 + cols, mask=mask, other=0.0).to(tl.float32)

    abs0 = tl.abs(y0)
    abs1 = tl.abs(y1)
    abs2 = tl.abs(y2)
    abs3 = tl.abs(y3)
    abs4 = tl.abs(y4)
    abs5 = tl.abs(y5)
    abs6 = tl.abs(y6)
    abs7 = tl.abs(y7)

    max0 = tl.max(abs0)
    max1 = tl.max(abs1)
    max2 = tl.max(abs2)
    max3 = tl.max(abs3)
    max4 = tl.max(abs4)
    max5 = tl.max(abs5)
    max6 = tl.max(abs6)
    max7 = tl.max(abs7)

    y_s0 = tl.maximum(max0, eps) * inv_fp8_max
    y_s1 = tl.maximum(max1, eps) * inv_fp8_max
    y_s2 = tl.maximum(max2, eps) * inv_fp8_max
    y_s3 = tl.maximum(max3, eps) * inv_fp8_max
    y_s4 = tl.maximum(max4, eps) * inv_fp8_max
    y_s5 = tl.maximum(max5, eps) * inv_fp8_max
    y_s6 = tl.maximum(max6, eps) * inv_fp8_max
    y_s7 = tl.maximum(max7, eps) * inv_fp8_max

    if scale_ue8m0:
        y_s0 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s0), 1e-10))))
        y_s1 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s1), 1e-10))))
        y_s2 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s2), 1e-10))))
        y_s3 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s3), 1e-10))))
        y_s4 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s4), 1e-10))))
        y_s5 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s5), 1e-10))))
        y_s6 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s6), 1e-10))))
        y_s7 = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(y_s7), 1e-10))))

    y_q0 = tl.clamp(y0 / y_s0, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q1 = tl.clamp(y1 / y_s1, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q2 = tl.clamp(y2 / y_s2, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q3 = tl.clamp(y3 / y_s3, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q4 = tl.clamp(y4 / y_s4, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q5 = tl.clamp(y5 / y_s5, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q6 = tl.clamp(y6 / y_s6, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    y_q7 = tl.clamp(y7 / y_s7, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr0 + cols, y_q0, mask=mask)
    tl.store(y_s_ptr0, y_s0)
    tl.store(y_q_ptr1 + cols, y_q1, mask=mask)
    tl.store(y_s_ptr1, y_s1)
    tl.store(y_q_ptr2 + cols, y_q2, mask=mask)
    tl.store(y_s_ptr2, y_s2)
    tl.store(y_q_ptr3 + cols, y_q3, mask=mask)
    tl.store(y_s_ptr3, y_s3)
    tl.store(y_q_ptr4 + cols, y_q4, mask=mask)
    tl.store(y_s_ptr4, y_s4)
    tl.store(y_q_ptr5 + cols, y_q5, mask=mask)
    tl.store(y_s_ptr5, y_s5)
    tl.store(y_q_ptr6 + cols, y_q6, mask=mask)
    tl.store(y_s_ptr6, y_s6)
    tl.store(y_q_ptr7 + cols, y_q7, mask=mask)
    tl.store(y_s_ptr7, y_s7)


def Groups_per_program(x, group_size) -> int:
    if (x.shape[-1] // group_size) % 8 == 0:
        return 8
    elif (x.shape[-1] // group_size) % 4 == 0:
        return 4
    elif (x.shape[-1] // group_size) % 2 == 0:
        return 2
    else:
        return 1


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.debug("GEMS PER TOKEN GROUP QUANT FP8")
    # dtype: The dype of output tensor. Note that only `torch.float8_e4m3fn`
    fp8_dtype = SUPPORTED_FP8_DTYPE if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    finfo = torch.finfo(fp8_dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    M = x.numel() // group_size
    N = group_size

    if column_major_scales:
        shape = (x.shape[-1] // group_size,) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    groups_per_program = Groups_per_program(x, group_size)
    if column_major_scales:
        if groups_per_program == 8:
            _per_token_group_quant_fp8_colmajor_m8[(M // 8,)](
                x,
                x_q,
                x_s,
                group_size,
                x.shape[1],
                x.stride(0),
                x_s.stride(1),
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                inv_fp8_max=1.0 / fp8_max,
                scale_ue8m0=scale_ue8m0,
                BLOCK=BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        elif groups_per_program == 4:
            _per_token_group_quant_fp8_colmajor_m4[(M // 4,)](
                x,
                x_q,
                x_s,
                group_size,
                x.shape[1],
                x.stride(0),
                x_s.stride(1),
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                inv_fp8_max=1.0 / fp8_max,
                scale_ue8m0=scale_ue8m0,
                BLOCK=BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        elif groups_per_program == 2:
            _per_token_group_quant_fp8_colmajor_m2[(M // 2,)](
                x,
                x_q,
                x_s,
                group_size,
                x.shape[1],
                x.stride(0),
                x_s.stride(1),
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                inv_fp8_max=1.0 / fp8_max,
                scale_ue8m0=scale_ue8m0,
                BLOCK=BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            _per_token_group_quant_fp8_colmajor[(M,)](
                x,
                x_q,
                x_s,
                group_size,
                x.shape[1],
                x.stride(0),
                x_s.stride(1),
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                inv_fp8_max=1.0 / fp8_max,
                scale_ue8m0=scale_ue8m0,
                BLOCK=BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    else:
        if groups_per_program == 8:
            _per_token_group_quant_fp8_m8[(M // 8,)](
                x,
                x_q,
                x_s,
                group_size,
                x.shape[1],
                x.stride(0),
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                inv_fp8_max=1.0 / fp8_max,
                scale_ue8m0=scale_ue8m0,
                BLOCK=BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        elif groups_per_program == 4:
            _per_token_group_quant_fp8_m4[(M // 4,)](
                x,
                x_q,
                x_s,
                group_size,
                x.shape[1],
                x.stride(0),
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                inv_fp8_max=1.0 / fp8_max,
                scale_ue8m0=scale_ue8m0,
                BLOCK=BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        elif groups_per_program == 2:
            _per_token_group_quant_fp8_m2[(M // 2,)](
                x,
                x_q,
                x_s,
                group_size,
                x.shape[1],
                x.stride(0),
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                inv_fp8_max=1.0 / fp8_max,
                scale_ue8m0=scale_ue8m0,
                BLOCK=BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            _per_token_group_quant_fp8[(M,)](
                x,
                x_q,
                x_s,
                group_size,
                x.shape[1],
                x.stride(0),
                eps,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                inv_fp8_max=1.0 / fp8_max,
                scale_ue8m0=scale_ue8m0,
                BLOCK=BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    return x_q, x_s
