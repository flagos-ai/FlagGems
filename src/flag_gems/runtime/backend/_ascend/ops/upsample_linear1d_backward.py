import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.runtime.backend._ascend.utils import CORE_NUM
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(
    f'flag_gems.runtime.backend._ascend.ops.{__name__.split(".")[-1]}'
)
device = device.name

ASCEND_910B_UB_BYTES = 192 * 1024
UB_RESERVED_BYTES = 16 * 1024
TEMP_BYTES_PER_LANE = 288
MAX_ROWS_PER_BLOCK = 16
MIN_BLOCK_W = 64
DOT_COEFF_MAX_ELEMENTS = 4 * 1024 * 1024
DOT_DOWNSAMPLE_COEFF_MAX_ELEMENTS = 8 * 1024 * 1024
DOT_MIN_ROWS = 16
DOT_ALIGN_FALSE_MIN_ROWS = 1024
DOT_BLOCK_M = 128
DOT_BLOCK_N = 256
DOT_BLOCK_K = 64
WINDOW_DOT_MIN_IN_W = 2048
WINDOW_DOT_MAX_ROWS = 256
WINDOW_DOT_BLOCK_M = 64
WINDOW_DOT_BLOCK_N = 32
WINDOW_DOT_BLOCK_K = 128
ALIGN_TRUE_WINDOW_DOT_BLOCK_M = 32
ALIGN_TRUE_WINDOW_DOT_BLOCK_N = 32
ALIGN_TRUE_WINDOW_DOT_BLOCK_K = 128
SCALE2_CONTIG_MIN_IN_W = 2048
SCALE2_CONTIG_BLOCK_W = 1024
SCALE2_CONTIG_BLOCK_2W = SCALE2_CONTIG_BLOCK_W * 2
SCALE2_BOUNDARY_BLOCK_M = 128
TORCH_COMPOSE_MIN_IN_W = 4096
TORCH_COMPOSE_MIN_ROWS = 1024
_TORCH_COMPOSE_CACHE = {}


@triton.jit
def upsample_linear1d_backward_coeff_kernel(
    coeff_ptr,
    IN_W: tl.constexpr,
    OUT_W: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < (OUT_W * IN_W)
    x_out = offsets // IN_W
    x_in = offsets - x_out * IN_W

    x_out_f = x_out.to(tl.float32)
    if ALIGN_CORNERS:
        if OUT_W > 1:
            x_real = x_out_f * ((IN_W - 1) + 0.0) / ((OUT_W - 1) + 0.0)
        else:
            x_real = tl.zeros((BLOCK,), dtype=tl.float32)
    else:
        x_real = (x_out_f + 0.5) * (IN_W + 0.0) / (OUT_W + 0.0) - 0.5
        x_real = tl.maximum(x_real, 0.0)

    x0_f = tl.floor(x_real)
    w1 = x_real - x0_f
    w0 = 1.0 - w1
    x0 = tl.maximum(x0_f, 0.0).to(tl.int32)
    x1 = tl.minimum(x0_f + 1.0, (IN_W - 1) + 0.0).to(tl.int32)
    same = x0 == x1

    weight = tl.where(same & (x_in == x0), w0 + w1, 0.0)
    weight += tl.where((~same) & (x_in == x0), w0, 0.0)
    weight += tl.where((~same) & (x_in == x1), w1, 0.0)
    tl.store(coeff_ptr + offsets, weight, mask=mask)


@triton.jit
def upsample_linear1d_backward_dot_kernel(
    grad_out_ptr,
    coeff_ptr,
    grad_in_ptr,
    rows,
    IN_W: tl.constexpr,
    OUT_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    x_in_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, OUT_W, BLOCK_K):
        x_out_offsets = k_start + tl.arange(0, BLOCK_K)
        grad = tl.load(
            grad_out_ptr + rows_offsets[:, None] * OUT_W + x_out_offsets[None, :],
            mask=(rows_offsets[:, None] < rows) & (x_out_offsets[None, :] < OUT_W),
            other=0.0,
        )
        coeff = tl.load(
            coeff_ptr + x_out_offsets[:, None] * IN_W + x_in_offsets[None, :],
            mask=(x_out_offsets[:, None] < OUT_W) & (x_in_offsets[None, :] < IN_W),
            other=0.0,
        )
        acc += tl.dot(grad, coeff)

    tl.store(
        grad_in_ptr + rows_offsets[:, None] * IN_W + x_in_offsets[None, :],
        acc,
        mask=(rows_offsets[:, None] < rows) & (x_in_offsets[None, :] < IN_W),
    )


@triton.jit
def upsample_linear1d_backward_scale2_align_true_window_dot_kernel(
    grad_out_ptr,
    grad_in_ptr,
    rows,
    IN_W: tl.constexpr,
    OUT_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    x_in = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    x_out_start = pid_n * (BLOCK_N * 2) - 2
    x_out = x_out_start + k_offsets
    safe_x_out = tl.minimum(tl.maximum(x_out, 0), OUT_W - 1)

    grad = tl.load(
        grad_out_ptr + row_offsets[:, None] * OUT_W + safe_x_out[None, :],
        mask=(row_offsets[:, None] < rows)
        & (x_out[None, :] >= 0)
        & (x_out[None, :] < OUT_W),
        other=0.0,
    )

    x_out_f = x_out[:, None].to(tl.float32)
    x_real = x_out_f * ((IN_W - 1) + 0.0) / ((OUT_W - 1) + 0.0)
    x0_f = tl.floor(x_real)
    w1 = x_real - x0_f
    w0 = 1.0 - w1
    x0 = tl.maximum(x0_f, 0.0).to(tl.int32)
    x1 = tl.minimum(x0_f + 1.0, (IN_W - 1) + 0.0).to(tl.int32)
    x_in_i = x_in[None, :].to(tl.int32)
    same = x0 == x1

    coeff = tl.where(same & (x_in_i == x0), w0 + w1, 0.0)
    coeff += tl.where((~same) & (x_in_i == x0), w0, 0.0)
    coeff += tl.where((~same) & (x_in_i == x1), w1, 0.0)
    coeff = tl.where(x_in[None, :] < IN_W, coeff, 0.0).to(tl.float16)

    acc = tl.dot(grad, coeff)
    tl.store(
        grad_in_ptr + row_offsets[:, None] * IN_W + x_in[None, :],
        acc,
        mask=(row_offsets[:, None] < rows) & (x_in[None, :] < IN_W),
    )


@triton.jit
def upsample_linear1d_backward_scale2_window_dot_kernel(
    grad_out_ptr,
    grad_in_ptr,
    rows,
    IN_W: tl.constexpr,
    OUT_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COEFF_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    x_in = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    x_out_start = pid_n * (BLOCK_N * 2) - 1
    x_out = x_out_start + k_offsets
    safe_x_out = tl.minimum(tl.maximum(x_out, 0), OUT_W - 1)

    grad = tl.load(
        grad_out_ptr + row_offsets[:, None] * OUT_W + safe_x_out[None, :],
        mask=(row_offsets[:, None] < rows)
        & (x_out[None, :] >= 0)
        & (x_out[None, :] < OUT_W),
        other=0.0,
    )

    x_in_i = x_in[None, :]
    x_out_i = x_out[:, None]
    coeff = tl.where(x_out_i == x_in_i * 2, 0.75, 0.0)
    coeff += tl.where(x_out_i == x_in_i * 2 + 1, 0.75, 0.0)
    coeff += tl.where((x_out_i == x_in_i * 2 - 1) & (x_in_i > 0), 0.25, 0.0)
    coeff += tl.where((x_out_i == x_in_i * 2 + 2) & (x_in_i < IN_W - 1), 0.25, 0.0)
    coeff += tl.where((x_in_i == 0) & (x_out_i == 0), 0.25, 0.0)
    coeff += tl.where((x_in_i == IN_W - 1) & (x_out_i == OUT_W - 1), 0.25, 0.0)
    coeff = tl.where(x_in_i < IN_W, coeff, 0.0)
    if COEFF_DTYPE == 0:
        coeff = coeff.to(tl.float16)
    else:
        coeff = coeff.to(tl.bfloat16)

    acc = tl.dot(grad, coeff)
    tl.store(
        grad_in_ptr + row_offsets[:, None] * IN_W + x_in[None, :],
        acc,
        mask=(row_offsets[:, None] < rows) & (x_in[None, :] < IN_W),
    )


@triton.jit
def upsample_linear1d_backward_kernel(
    grad_out_ptr,
    grad_in_ptr,
    rows,
    in_w,
    out_w,
    align_corners: tl.constexpr,
    BLOCK_W: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    row_start = ext.program_id(axis=1) * ROWS_PER_BLOCK
    row_step = tl.num_programs(axis=1) * ROWS_PER_BLOCK
    x_in = ext.program_id(axis=0) * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]
    width_mask = x_in < in_w

    x_in_f = x_in.to(tl.float32)
    in_w_f = tl.cast(in_w, tl.float32)
    out_w_f = tl.cast(out_w, tl.float32)

    if align_corners:
        if in_w > 1:
            center = x_in_f * (out_w_f - 1.0) / (in_w_f - 1.0)
        else:
            center = tl.zeros((1, BLOCK_W), dtype=tl.float32)
    else:
        center = (x_in_f + 0.5) * out_w_f / in_w_f - 0.5

    base = tl.floor(center).to(tl.int32)

    while row_start < rows:
        row_offsets = row_start + tl.arange(0, ROWS_PER_BLOCK)[:, None]
        mask = (row_offsets < rows) & width_mask
        go_base = grad_out_ptr + row_offsets * out_w
        acc = tl.zeros((ROWS_PER_BLOCK, BLOCK_W), dtype=tl.float32)

        for i in range(-2, 3):
            x_out = base + i
            valid = (x_out >= 0) & (x_out < out_w)
            x_out_f = x_out.to(tl.float32)

            if align_corners:
                if out_w > 1:
                    x_real = x_out_f * (in_w_f - 1.0) / (out_w_f - 1.0)
                else:
                    x_real = tl.zeros((1, BLOCK_W), dtype=tl.float32)
            else:
                x_real = (x_out_f + 0.5) * in_w_f / out_w_f - 0.5

            x0_f = tl.floor(x_real)
            w1 = x_real - x0_f
            w0 = 1.0 - w1

            x0_i = tl.maximum(x0_f, 0.0).to(tl.int32)
            x1_i = tl.minimum(x0_f + 1.0, in_w_f - 1.0).to(tl.int32)

            g = tl.load(go_base + x_out, mask=mask & valid, other=0.0).to(tl.float32)

            same = x0_i == x1_i
            is_x0 = x_in.to(tl.int32) == x0_i
            is_x1 = x_in.to(tl.int32) == x1_i

            acc += tl.where(same & is_x0, g * (w0 + w1), 0.0)
            acc += tl.where(~same & is_x0, g * w0, 0.0)
            acc += tl.where(~same & is_x1, g * w1, 0.0)

        tl.store(grad_in_ptr + row_offsets * in_w + x_in, acc, mask=mask)
        row_start += row_step


@triton.jit
def upsample_linear1d_backward_align_false_scale_kernel(
    grad_out_ptr,
    grad_in_ptr,
    rows,
    in_w,
    out_w,
    SCALE_MODE: tl.constexpr,
    BLOCK_W: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    row_start = ext.program_id(axis=1) * ROWS_PER_BLOCK
    row_step = tl.num_programs(axis=1) * ROWS_PER_BLOCK
    x_in = ext.program_id(axis=0) * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]
    width_mask = x_in < in_w

    while row_start < rows:
        row_offsets = row_start + tl.arange(0, ROWS_PER_BLOCK)[:, None]
        mask = (row_offsets < rows) & width_mask
        go_base = grad_out_ptr + row_offsets * out_w

        if SCALE_MODE == 0:
            x_out = x_in >> 1
            acc = tl.load(go_base + x_out, mask=mask & (x_out < out_w), other=0.0)
            acc = acc.to(tl.float32) * 0.5
        else:
            x_out = x_in << 1
            is_first = x_in == 0
            is_last = x_in == (in_w - 1)

            g_even = tl.load(
                go_base + x_out, mask=mask & (x_out < out_w), other=0.0
            ).to(tl.float32)

            x_odd = x_out + 1
            g_odd = tl.load(go_base + x_odd, mask=mask & (x_odd < out_w), other=0.0).to(
                tl.float32
            )

            x_prev = x_out - 1
            g_prev = tl.load(
                go_base + tl.maximum(x_prev, 0),
                mask=mask & (x_in > 0),
                other=0.0,
            ).to(tl.float32)

            x_next_even = x_out + 2
            g_next_even = tl.load(
                go_base + x_next_even,
                mask=mask & (x_next_even < out_w) & ~is_last,
                other=0.0,
            ).to(tl.float32)

            acc = (g_even + g_odd) * 0.75 + (g_prev + g_next_even) * 0.25
            acc += tl.where(is_first, g_even * 0.25, 0.0)
            acc += tl.where(is_last, g_odd * 0.25, 0.0)

        tl.store(grad_in_ptr + row_offsets * in_w + x_in, acc, mask=mask)
        row_start += row_step


@triton.jit
def upsample_linear1d_backward_align_false_scale2_contig_kernel(
    grad_out_ptr,
    grad_in_ptr,
    rows,
    IN_W: tl.constexpr,
    OUT_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_2W: tl.constexpr,
):
    row_start = ext.program_id(axis=1)
    row_step = tl.num_programs(axis=1)
    x_start = ext.program_id(axis=0) * BLOCK_W
    x_offsets = x_start + tl.arange(0, BLOCK_W)
    load_offsets = tl.arange(0, BLOCK_2W)
    x_out_start = (x_start << 1) - 1
    width_mask = x_offsets < IN_W

    while row_start < rows:
        go_base = grad_out_ptr + row_start * OUT_W
        v0_pos = x_out_start + load_offsets
        v1_pos = x_out_start + 2 + load_offsets
        v0 = tl.load(
            go_base + tl.maximum(v0_pos, 0),
            mask=(v0_pos >= 0) & (v0_pos < OUT_W),
            other=0.0,
        ).to(tl.float32)
        v1 = tl.load(
            go_base + tl.minimum(tl.maximum(v1_pos, 0), OUT_W - 1),
            mask=(v1_pos >= 0) & (v1_pos < OUT_W),
            other=0.0,
        ).to(tl.float32)

        prev_even = tl.reshape(v0, (BLOCK_W, 2))
        odd_next = tl.reshape(v1, (BLOCK_W, 2))
        g_prev, g_even = tl.split(prev_even)
        g_odd, g_next = tl.split(odd_next)

        acc = (g_even + g_odd) * 0.75 + (g_prev + g_next) * 0.25
        acc += tl.where(x_offsets == 0, g_even * 0.25, 0.0)
        acc += tl.where(x_offsets == IN_W - 1, g_odd * 0.25, 0.0)
        tl.store(grad_in_ptr + row_start * IN_W + x_offsets, acc, mask=width_mask)
        row_start += row_step


@triton.jit
def upsample_linear1d_backward_align_true_scale2_contig_kernel(
    grad_out_ptr,
    grad_in_ptr,
    rows,
    IN_W: tl.constexpr,
    OUT_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_2W: tl.constexpr,
):
    row_start = ext.program_id(axis=1)
    row_step = tl.num_programs(axis=1)
    x_start = ext.program_id(axis=0) * BLOCK_W
    x_offsets = x_start + tl.arange(0, BLOCK_W)
    load_offsets = tl.arange(0, BLOCK_2W)
    x_out_start = (x_start << 1) - 1
    width_mask = x_offsets < IN_W

    denom = (OUT_W - 1) + 0.0
    x_offsets_f = x_offsets.to(tl.float32)
    in_w_f = IN_W + 0.0
    w_prev = (in_w_f - x_offsets_f) / denom
    w_even = 1.0 - x_offsets_f / denom
    w_odd = (in_w_f + x_offsets_f) / denom
    w_next = (x_offsets_f + 1.0) / denom

    while row_start < rows:
        go_base = grad_out_ptr + row_start * OUT_W
        v0_pos = x_out_start + load_offsets
        v1_pos = x_out_start + 2 + load_offsets
        v0 = tl.load(
            go_base + tl.maximum(v0_pos, 0),
            mask=(v0_pos >= 0) & (v0_pos < OUT_W),
            other=0.0,
        ).to(tl.float32)
        v1 = tl.load(
            go_base + tl.minimum(tl.maximum(v1_pos, 0), OUT_W - 1),
            mask=(v1_pos >= 0) & (v1_pos < OUT_W),
            other=0.0,
        ).to(tl.float32)

        prev_even = tl.reshape(v0, (BLOCK_W, 2))
        odd_next = tl.reshape(v1, (BLOCK_W, 2))
        g_prev, g_even = tl.split(prev_even)
        g_odd, g_next = tl.split(odd_next)

        acc = g_prev * w_prev + g_even * w_even + g_odd * w_odd + g_next * w_next
        tl.store(grad_in_ptr + row_start * IN_W + x_offsets, acc, mask=width_mask)
        row_start += row_step


@triton.jit
def upsample_linear1d_backward_scale2_boundary_kernel(
    grad_out_ptr,
    grad_in_ptr,
    rows,
    IN_W: tl.constexpr,
    OUT_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = row_offsets < rows

    left = tl.load(grad_in_ptr + row_offsets * IN_W, mask=mask, other=0.0).to(
        tl.float32
    )
    left_grad = tl.load(grad_out_ptr + row_offsets * OUT_W, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(grad_in_ptr + row_offsets * IN_W, left + left_grad * 0.25, mask=mask)

    right = tl.load(
        grad_in_ptr + row_offsets * IN_W + (IN_W - 1), mask=mask, other=0.0
    ).to(tl.float32)
    right_grad = tl.load(
        grad_out_ptr + row_offsets * OUT_W + (OUT_W - 1), mask=mask, other=0.0
    ).to(tl.float32)
    tl.store(
        grad_in_ptr + row_offsets * IN_W + (IN_W - 1),
        right + right_grad * 0.25,
        mask=mask,
    )


def _normalize_input_size(input_size):
    if len(input_size) == 3:
        return input_size
    if len(input_size) == 2:
        return input_size[0], 1, input_size[1]
    if len(input_size) == 1:
        return 1, 1, input_size[0]
    raise ValueError


def _device_guard(tensor):
    device_index = tensor.device.index
    if device_index is None or device_index == torch_device_fn.current_device():
        return nullcontext()
    return torch_device_fn.device(tensor.device)


def _prev_power_of_2(value):
    return 1 << (int(value).bit_length() - 1)


def _select_tiles(in_w, element_size):
    bytes_per_lane = TEMP_BYTES_PER_LANE + element_size * 8
    usable_ub = ASCEND_910B_UB_BYTES - UB_RESERVED_BYTES
    max_tile_elements = _prev_power_of_2(max(1, usable_ub // bytes_per_lane))
    block_w = min(max(MIN_BLOCK_W, triton.next_power_of_2(in_w)), max_tile_elements)
    rows_per_block = max(1, min(MAX_ROWS_PER_BLOCK, max_tile_elements // block_w))
    return block_w, rows_per_block


def _can_use_dot_path(grad_output, rows, in_w, out_w, align_corners):
    if grad_output.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False

    exact_downsample = out_w * 2 == in_w
    exact_upsample = out_w == in_w * 2
    if not (exact_downsample or exact_upsample):
        return False

    if rows < DOT_MIN_ROWS:
        return False

    if not align_corners and rows < DOT_ALIGN_FALSE_MIN_ROWS:
        return False

    max_coeff_elements = (
        DOT_DOWNSAMPLE_COEFF_MAX_ELEMENTS
        if exact_downsample
        else DOT_COEFF_MAX_ELEMENTS
    )
    return in_w * out_w <= max_coeff_elements


def _dot_block_k(grad_output, in_w, out_w):
    if (
        out_w * 2 == in_w
        and in_w >= 4096
        and grad_output.dtype
        in (
            torch.float16,
            torch.bfloat16,
        )
    ):
        return 128
    return DOT_BLOCK_K


def _can_use_window_dot_path(grad_output, rows, in_w, out_w, align_corners):
    return (
        not align_corners
        and out_w == in_w * 2
        and in_w >= WINDOW_DOT_MIN_IN_W
        and rows <= WINDOW_DOT_MAX_ROWS
        and grad_output.dtype in (torch.float16, torch.bfloat16)
    )


def _can_use_align_true_window_dot_path(grad_output, in_w, out_w, align_corners):
    return (
        align_corners
        and out_w == in_w * 2
        and in_w >= WINDOW_DOT_MIN_IN_W
        and grad_output.dtype == torch.float16
    )


def _can_use_scale2_contig_path(grad_output, rows, in_w, out_w, align_corners):
    if out_w != in_w * 2 or in_w < SCALE2_CONTIG_MIN_IN_W:
        return False

    if align_corners:
        return grad_output.dtype != torch.float16 or rows > WINDOW_DOT_MAX_ROWS
    return True


def _can_use_downsample_view_copy_path(grad_output, rows, in_w, out_w, align_corners):
    return (
        not align_corners
        and out_w * 2 == in_w
        and in_w >= TORCH_COMPOSE_MIN_IN_W
        and rows >= TORCH_COMPOSE_MIN_ROWS
        and grad_output.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def _can_use_scale2_conv_path(grad_output, rows, in_w, out_w, align_corners):
    return (
        not align_corners
        and out_w == in_w * 2
        and in_w >= TORCH_COMPOSE_MIN_IN_W
        and rows >= TORCH_COMPOSE_MIN_ROWS
        and grad_output.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def _can_use_align_true_scale2_conv_path(grad_output, rows, in_w, out_w, align_corners):
    return (
        align_corners
        and out_w == in_w * 2
        and in_w >= TORCH_COMPOSE_MIN_IN_W
        and rows >= TORCH_COMPOSE_MIN_ROWS
        and grad_output.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def _upsample_linear1d_backward_downsample_view_copy(grad_out_3d, n, c, in_w, out_w):
    grad_in = torch.empty(
        (n, c, in_w), device=grad_out_3d.device, dtype=grad_out_3d.dtype
    )
    half_grad = (grad_out_3d * 0.5).unsqueeze(-1).expand(n, c, out_w, 2)
    grad_in.view(n, c, out_w, 2).copy_(half_grad)
    return grad_in


def _cache_key(name, tensor, channels, in_w):
    return (
        name,
        tensor.device.type,
        tensor.device.index,
        tensor.dtype,
        channels,
        in_w,
    )


def _get_scale2_conv_weight(grad_out_3d, channels, in_w):
    dtype = torch.float32 if grad_out_3d.dtype == torch.float16 else grad_out_3d.dtype
    key = (
        "scale2_conv_weight",
        grad_out_3d.device.type,
        grad_out_3d.device.index,
        dtype,
        channels,
        in_w,
    )
    weight = _TORCH_COMPOSE_CACHE.get(key)
    if weight is None:
        weight = torch.tensor(
            [0.25, 0.75, 0.75, 0.25],
            device=grad_out_3d.device,
            dtype=dtype,
        )
        weight = weight.view(1, 1, 1, 4).expand(channels, 1, 1, 4).contiguous()
        _TORCH_COMPOSE_CACHE[key] = weight
    return weight


def _get_align_true_affine_conv_params(grad_out_3d, channels, in_w, out_w):
    dtype = torch.float32 if grad_out_3d.dtype == torch.float16 else grad_out_3d.dtype
    key = (
        "align_true_affine_conv",
        grad_out_3d.device.type,
        grad_out_3d.device.index,
        dtype,
        channels,
        in_w,
    )
    params = _TORCH_COMPOSE_CACHE.get(key)
    if params is None:
        denom = float(out_w - 1)
        base = torch.tensor(
            [in_w / denom, 1.0, in_w / denom, 1.0 / denom],
            device=grad_out_3d.device,
            dtype=dtype,
        )
        slope = torch.tensor(
            [-1.0 / denom, -1.0 / denom, 1.0 / denom, 1.0 / denom],
            device=grad_out_3d.device,
            dtype=dtype,
        )
        weight = torch.empty(
            (channels * 2, 1, 1, 4),
            device=grad_out_3d.device,
            dtype=dtype,
        )
        weight[0::2, 0, 0, :].copy_(base)
        weight[1::2, 0, 0, :].copy_(slope)
        offsets = torch.arange(in_w, device=grad_out_3d.device, dtype=dtype).view(
            1, 1, in_w
        )
        params = weight, offsets
        _TORCH_COMPOSE_CACHE[key] = params
    return params


def _upsample_linear1d_backward_scale2_conv(grad_out_3d, n, c, in_w, out_w):
    weight = _get_scale2_conv_weight(grad_out_3d, c, in_w)
    compute_grad = (
        torch.ops.npu.npu_dtype_cast(grad_out_3d, torch.float32)
        if grad_out_3d.dtype == torch.float16
        else grad_out_3d
    )
    grad_in = torch.ops.npu.npu_conv2d(
        compute_grad.view(n, c, 1, out_w),
        weight,
        None,
        [1, 2],
        [0, 1],
        [1, 1],
        c,
    ).view(n, c, in_w)
    upsample_linear1d_backward_scale2_boundary_kernel[
        (triton.cdiv(n * c, SCALE2_BOUNDARY_BLOCK_M),)
    ](
        compute_grad,
        grad_in,
        n * c,
        IN_W=in_w,
        OUT_W=out_w,
        BLOCK_M=SCALE2_BOUNDARY_BLOCK_M,
    )
    if grad_out_3d.dtype == torch.float16:
        grad_in = torch.ops.npu.npu_dtype_cast(grad_in, torch.float16)
    return grad_in


def _upsample_linear1d_backward_align_true_scale2_conv(grad_out_3d, n, c, in_w, out_w):
    weight, offsets = _get_align_true_affine_conv_params(grad_out_3d, c, in_w, out_w)
    compute_grad = (
        torch.ops.npu.npu_dtype_cast(grad_out_3d, torch.float32)
        if grad_out_3d.dtype == torch.float16
        else grad_out_3d
    )
    pair = torch.ops.npu.npu_conv2d(
        compute_grad.view(n, c, 1, out_w),
        weight,
        None,
        [1, 2],
        [0, 1],
        [1, 1],
        c,
    ).view(n, c, 2, in_w)
    grad_in = pair[:, :, 0, :]
    grad_in.addcmul_(pair[:, :, 1, :], offsets)
    if grad_out_3d.dtype == torch.float16:
        grad_in = torch.ops.npu.npu_dtype_cast(grad_in, torch.float16)
    return grad_in


def _upsample_linear1d_backward_dot(
    grad_out_3d, n, c, rows, in_w, out_w, align_corners
):
    grad_out_2d = grad_out_3d.view(rows, out_w)
    grad_in_2d = torch.empty(
        (rows, in_w), device=grad_out_3d.device, dtype=grad_out_3d.dtype
    )
    coeff = torch.empty(
        (out_w, in_w), device=grad_out_3d.device, dtype=grad_out_3d.dtype
    )

    upsample_linear1d_backward_coeff_kernel[(triton.cdiv(out_w * in_w, 1024),)](
        coeff,
        IN_W=in_w,
        OUT_W=out_w,
        ALIGN_CORNERS=align_corners,
        BLOCK=1024,
    )

    grid = (
        triton.cdiv(rows, DOT_BLOCK_M),
        triton.cdiv(in_w, DOT_BLOCK_N),
    )
    upsample_linear1d_backward_dot_kernel[grid](
        grad_out_2d,
        coeff,
        grad_in_2d,
        rows,
        IN_W=in_w,
        OUT_W=out_w,
        BLOCK_M=DOT_BLOCK_M,
        BLOCK_N=DOT_BLOCK_N,
        BLOCK_K=_dot_block_k(grad_out_3d, in_w, out_w),
    )
    return grad_in_2d.view(n, c, in_w)


def _upsample_linear1d_backward_window_dot(grad_out_3d, n, c, rows, in_w, out_w):
    grad_out_2d = grad_out_3d.view(rows, out_w)
    grad_in_2d = torch.empty(
        (rows, in_w), device=grad_out_3d.device, dtype=grad_out_3d.dtype
    )
    grid = (
        triton.cdiv(rows, WINDOW_DOT_BLOCK_M),
        triton.cdiv(in_w, WINDOW_DOT_BLOCK_N),
    )
    upsample_linear1d_backward_scale2_window_dot_kernel[grid](
        grad_out_2d,
        grad_in_2d,
        rows,
        IN_W=in_w,
        OUT_W=out_w,
        BLOCK_M=WINDOW_DOT_BLOCK_M,
        BLOCK_N=WINDOW_DOT_BLOCK_N,
        BLOCK_K=WINDOW_DOT_BLOCK_K,
        COEFF_DTYPE=1 if grad_out_3d.dtype == torch.bfloat16 else 0,
    )
    return grad_in_2d.view(n, c, in_w)


def _upsample_linear1d_backward_align_true_window_dot(
    grad_out_3d, n, c, rows, in_w, out_w
):
    grad_out_2d = grad_out_3d.view(rows, out_w)
    grad_in_2d = torch.empty(
        (rows, in_w), device=grad_out_3d.device, dtype=grad_out_3d.dtype
    )
    grid = (
        triton.cdiv(rows, ALIGN_TRUE_WINDOW_DOT_BLOCK_M),
        triton.cdiv(in_w, ALIGN_TRUE_WINDOW_DOT_BLOCK_N),
    )
    upsample_linear1d_backward_scale2_align_true_window_dot_kernel[grid](
        grad_out_2d,
        grad_in_2d,
        rows,
        IN_W=in_w,
        OUT_W=out_w,
        BLOCK_M=ALIGN_TRUE_WINDOW_DOT_BLOCK_M,
        BLOCK_N=ALIGN_TRUE_WINDOW_DOT_BLOCK_N,
        BLOCK_K=ALIGN_TRUE_WINDOW_DOT_BLOCK_K,
    )
    return grad_in_2d.view(n, c, in_w)


def _upsample_linear1d_backward_scale2_contig(
    grad_out_3d, n, c, rows, in_w, out_w, align_corners
):
    grad_out_2d = grad_out_3d.view(rows, out_w)
    grad_in_2d = torch.empty(
        (rows, in_w), device=grad_out_3d.device, dtype=grad_out_3d.dtype
    )
    grid = (
        triton.cdiv(in_w, SCALE2_CONTIG_BLOCK_W),
        min(rows, CORE_NUM),
    )
    if align_corners:
        upsample_linear1d_backward_align_true_scale2_contig_kernel[grid](
            grad_out_2d,
            grad_in_2d,
            rows,
            IN_W=in_w,
            OUT_W=out_w,
            BLOCK_W=SCALE2_CONTIG_BLOCK_W,
            BLOCK_2W=SCALE2_CONTIG_BLOCK_2W,
        )
    else:
        upsample_linear1d_backward_align_false_scale2_contig_kernel[grid](
            grad_out_2d,
            grad_in_2d,
            rows,
            IN_W=in_w,
            OUT_W=out_w,
            BLOCK_W=SCALE2_CONTIG_BLOCK_W,
            BLOCK_2W=SCALE2_CONTIG_BLOCK_2W,
        )
    return grad_in_2d.view(n, c, in_w)


def upsample_linear1d_backward(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    align_corners: bool,
    scale_factors=None,
) -> torch.Tensor:
    logger.debug("GEMS_ASCEND UPSAMPLE_LINEAR1D_BACKWARD")
    assert grad_output.device.type == device

    n, c, in_w = _normalize_input_size(input_size)
    if output_size is not None:
        out_w = output_size[0]
    else:
        assert scale_factors is not None
        out_w = int(in_w * scale_factors[0])

    assert grad_output.shape[-1] == out_w

    grad_out_3d = grad_output.contiguous().view(n, c, out_w)
    rows = n * c

    with _device_guard(grad_output):
        if _can_use_downsample_view_copy_path(
            grad_output, rows, in_w, out_w, align_corners
        ):
            grad_in = _upsample_linear1d_backward_downsample_view_copy(
                grad_out_3d,
                n,
                c,
                in_w,
                out_w,
            )
        elif _can_use_scale2_conv_path(grad_output, rows, in_w, out_w, align_corners):
            grad_in = _upsample_linear1d_backward_scale2_conv(
                grad_out_3d,
                n,
                c,
                in_w,
                out_w,
            )
        elif _can_use_align_true_scale2_conv_path(
            grad_output, rows, in_w, out_w, align_corners
        ):
            grad_in = _upsample_linear1d_backward_align_true_scale2_conv(
                grad_out_3d,
                n,
                c,
                in_w,
                out_w,
            )
        elif (out_w * 2 == in_w or out_w == in_w * 2) and _can_use_dot_path(
            grad_output, rows, in_w, out_w, align_corners
        ):
            grad_in = _upsample_linear1d_backward_dot(
                grad_out_3d,
                n,
                c,
                rows,
                in_w,
                out_w,
                align_corners,
            )
        elif _can_use_window_dot_path(grad_output, rows, in_w, out_w, align_corners):
            grad_in = _upsample_linear1d_backward_window_dot(
                grad_out_3d,
                n,
                c,
                rows,
                in_w,
                out_w,
            )
        elif _can_use_scale2_contig_path(grad_output, rows, in_w, out_w, align_corners):
            grad_in = _upsample_linear1d_backward_scale2_contig(
                grad_out_3d,
                n,
                c,
                rows,
                in_w,
                out_w,
                align_corners,
            )
        elif _can_use_align_true_window_dot_path(
            grad_output, in_w, out_w, align_corners
        ):
            grad_in = _upsample_linear1d_backward_align_true_window_dot(
                grad_out_3d,
                n,
                c,
                rows,
                in_w,
                out_w,
            )
        elif not align_corners and out_w * 2 == in_w:
            grad_in = torch.empty(
                (n, c, in_w), device=grad_output.device, dtype=grad_output.dtype
            )
            block_w, rows_per_block = _select_tiles(in_w, grad_output.element_size())
            row_blocks = triton.cdiv(rows, rows_per_block)
            grid = (triton.cdiv(in_w, block_w), min(row_blocks, CORE_NUM))

            upsample_linear1d_backward_align_false_scale_kernel[grid](
                grad_out_3d,
                grad_in,
                rows,
                in_w,
                out_w,
                SCALE_MODE=0,
                BLOCK_W=block_w,
                ROWS_PER_BLOCK=rows_per_block,
            )
        elif not align_corners and out_w == in_w * 2:
            grad_in = torch.empty(
                (n, c, in_w), device=grad_output.device, dtype=grad_output.dtype
            )
            block_w, rows_per_block = _select_tiles(in_w, grad_output.element_size())
            if in_w <= MIN_BLOCK_W and rows <= MIN_BLOCK_W:
                rows_per_block = min(2, rows)
            row_blocks = triton.cdiv(rows, rows_per_block)
            grid = (triton.cdiv(in_w, block_w), min(row_blocks, CORE_NUM))

            upsample_linear1d_backward_align_false_scale_kernel[grid](
                grad_out_3d,
                grad_in,
                rows,
                in_w,
                out_w,
                SCALE_MODE=1,
                BLOCK_W=block_w,
                ROWS_PER_BLOCK=rows_per_block,
            )
        else:
            grad_in = torch.empty(
                (n, c, in_w), device=grad_output.device, dtype=grad_output.dtype
            )
            block_w, rows_per_block = _select_tiles(in_w, grad_output.element_size())
            row_blocks = triton.cdiv(rows, rows_per_block)
            grid = (triton.cdiv(in_w, block_w), min(row_blocks, CORE_NUM))

            upsample_linear1d_backward_kernel[grid](
                grad_out_3d,
                grad_in,
                rows,
                in_w,
                out_w,
                align_corners,
                BLOCK_W=block_w,
                ROWS_PER_BLOCK=rows_per_block,
            )

    return grad_in
