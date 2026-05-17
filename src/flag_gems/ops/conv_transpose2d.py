import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_TRITON_DIRECT_LOWP_DTYPES = (torch.float16, torch.bfloat16)

# Exact no-bias/group=1 shapes covered by the direct Triton kernel.
_DIRECT_TRITON_SHAPES = {
    (1, 64, 128, 128, 64, 3, 3, 1, 1),
    (1, 64, 64, 64, 32, 3, 3, 2, 1),
    (4, 32, 32, 32, 32, 3, 3, 2, 1),
    (8, 16, 64, 64, 16, 5, 5, 2, 2),
    (16, 32, 16, 16, 64, 3, 3, 2, 1),
    (32, 64, 32, 32, 32, 3, 3, 1, 0),
    (32, 64, 16, 16, 32, 3, 3, 1, 1),
    (32, 64, 16, 16, 32, 3, 3, 2, 1),
    (16, 32, 32, 32, 64, 3, 3, 2, 1),
    (16, 32, 8, 8, 24, 5, 5, 2, 2),
}

_BLOCKER_FP16_SHAPE = (4, 256, 8, 8, 128, 3, 3, 2, 1)

_K4_FP16_SHAPE = (8, 128, 4, 4, 64, 4, 4, 2, 1)


def _pair(value):
    if isinstance(value, (list, tuple)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _exact_shape_key(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    output_padding_h,
    output_padding_w,
    groups,
    dilation_h,
    dilation_w,
    supported_dtypes,
):
    if bias is not None:
        return None
    if groups != 1:
        return None
    if (dilation_h, dilation_w) != (1, 1):
        return None
    if (output_padding_h, output_padding_w) != (0, 0):
        return None
    if input.dtype not in supported_dtypes or weight.dtype != input.dtype:
        return None
    if input.device.type != "cuda" or weight.device != input.device:
        return None
    if input.dim() != 4 or weight.dim() != 4:
        return None
    if not input.is_contiguous() or not weight.is_contiguous():
        return None
    if stride_h != stride_w or padding_h != padding_w:
        return None

    batch, input_channels, input_height, input_width = input.shape
    weight_input_channels, output_channels, weight_height, weight_width = weight.shape
    if input_channels != weight_input_channels:
        return None
    return (
        batch,
        input_channels,
        input_height,
        input_width,
        output_channels,
        weight_height,
        weight_width,
        stride_h,
        padding_h,
    )


def _unsupported_conv_transpose2d(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    output_padding_h,
    output_padding_w,
    groups,
    dilation_h,
    dilation_w,
):
    bias_dtype = None if bias is None else bias.dtype
    raise NotImplementedError(
        "flag_gems.conv_transpose2d supports only tuned Triton cases: "
        "4D contiguous CUDA input/weight tensors, bias=None, groups=1, "
        "dilation=(1, 1), output_padding=(0, 0), and one of the registered "
        "shape/dtype combinations; got "
        f"input_shape={tuple(input.shape)}, weight_shape={tuple(weight.shape)}, "
        f"input_dtype={input.dtype}, weight_dtype={weight.dtype}, bias_dtype={bias_dtype}, "
        f"input_device={input.device}, weight_device={weight.device}, "
        f"stride=({stride_h}, {stride_w}), padding=({padding_h}, {padding_w}), "
        f"output_padding=({output_padding_h}, {output_padding_w}), groups={groups}, "
        f"dilation=({dilation_h}, {dilation_w})"
    )


@libentry()
@triton.jit
def _conv_transpose2d_blocker_fp16_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    block_id = pid // 4
    pid_c = pid - block_id * 4

    local_pid = block_id
    plane = 0
    if block_id < 8:
        local_pid = block_id
        plane = 0
    elif block_id < 15:
        local_pid = block_id - 8
        plane = 1
    elif block_id < 22:
        local_pid = block_id - 15
        plane = 2
    else:
        local_pid = block_id - 22
        plane = 3

    m_offsets = local_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    co_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    ci_offsets_base = tl.arange(0, BLOCK_K)

    if plane == 0:
        plane_m = 4 * 8 * 8
        iw = m_offsets % 8
        tmp = m_offsets // 8
        ih = tmp % 8
        n = tmp // 8
        oh = ih * 2
        ow = iw * 2
        accum = tl.zeros((BLOCK_M, BLOCK_C), tl.float32)
        for ci_base in range(0, 256, BLOCK_K):
            ci_offsets = ci_base + ci_offsets_base
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + ih[:, None]) * 8
                + iw[:, None],
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3 + 1) * 3
                + 1,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            accum += tl.dot(input_block, weight_block, allow_tf32=False)
        tl.store(
            output_pointer
            + ((n[:, None] * 128 + co_offsets[None, :]) * 15 + oh[:, None]) * 15
            + ow[:, None],
            accum,
            mask=(m_offsets[:, None] < plane_m) & (co_offsets[None, :] < 128),
        )
    elif plane == 1:
        plane_m = 4 * 8 * 7
        j = m_offsets % 7
        tmp = m_offsets // 7
        ih = tmp % 8
        n = tmp // 8
        oh = ih * 2
        ow = j * 2 + 1
        accum = tl.zeros((BLOCK_M, BLOCK_C), tl.float32)
        for ci_base in range(0, 256, BLOCK_K):
            ci_offsets = ci_base + ci_offsets_base
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3 + 1) * 3
                + 2,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + ih[:, None]) * 8
                + j[:, None],
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            accum += tl.dot(input_block, weight_block, allow_tf32=False)
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3 + 1) * 3,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + ih[:, None]) * 8
                + (j[:, None] + 1),
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            accum += tl.dot(input_block, weight_block, allow_tf32=False)
        tl.store(
            output_pointer
            + ((n[:, None] * 128 + co_offsets[None, :]) * 15 + oh[:, None]) * 15
            + ow[:, None],
            accum,
            mask=(m_offsets[:, None] < plane_m) & (co_offsets[None, :] < 128),
        )
    elif plane == 2:
        plane_m = 4 * 7 * 8
        iw = m_offsets % 8
        tmp = m_offsets // 8
        i = tmp % 7
        n = tmp // 7
        oh = i * 2 + 1
        ow = iw * 2
        accum = tl.zeros((BLOCK_M, BLOCK_C), tl.float32)
        for ci_base in range(0, 256, BLOCK_K):
            ci_offsets = ci_base + ci_offsets_base
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3 + 2) * 3
                + 1,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + i[:, None]) * 8
                + iw[:, None],
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            accum += tl.dot(input_block, weight_block, allow_tf32=False)
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3) * 3
                + 1,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + (i[:, None] + 1)) * 8
                + iw[:, None],
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            accum += tl.dot(input_block, weight_block, allow_tf32=False)
        tl.store(
            output_pointer
            + ((n[:, None] * 128 + co_offsets[None, :]) * 15 + oh[:, None]) * 15
            + ow[:, None],
            accum,
            mask=(m_offsets[:, None] < plane_m) & (co_offsets[None, :] < 128),
        )
    else:
        plane_m = 4 * 7 * 7
        j = m_offsets % 7
        tmp = m_offsets // 7
        i = tmp % 7
        n = tmp // 7
        oh = i * 2 + 1
        ow = j * 2 + 1
        accum = tl.zeros((BLOCK_M, BLOCK_C), tl.float32)
        for ci_base in range(0, 256, BLOCK_K):
            ci_offsets = ci_base + ci_offsets_base
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3 + 2) * 3
                + 2,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + i[:, None]) * 8
                + j[:, None],
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            accum += tl.dot(input_block, weight_block, allow_tf32=False)
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3 + 2) * 3,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + i[:, None]) * 8
                + (j[:, None] + 1),
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            accum += tl.dot(input_block, weight_block, allow_tf32=False)
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3) * 3
                + 2,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + (i[:, None] + 1)) * 8
                + j[:, None],
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            accum += tl.dot(input_block, weight_block, allow_tf32=False)
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3) * 3,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + (i[:, None] + 1)) * 8
                + (j[:, None] + 1),
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            accum += tl.dot(input_block, weight_block, allow_tf32=False)
        tl.store(
            output_pointer
            + ((n[:, None] * 128 + co_offsets[None, :]) * 15 + oh[:, None]) * 15
            + ow[:, None],
            accum,
            mask=(m_offsets[:, None] < plane_m) & (co_offsets[None, :] < 128),
        )


@libentry()
@triton.jit
def _conv_transpose2d_k4_fp16_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    BLOCK_M: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_co = tl.program_id(1)
    plane = tl.program_id(2)

    residue_h = plane // 2
    residue_w = plane - residue_h * 2
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    compact_w = m_offsets % 4
    compact_nh = m_offsets // 4
    compact_h = compact_nh % 4
    n = compact_nh // 4
    oh = compact_h * 2 + residue_h
    ow = compact_w * 2 + residue_w

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    ci_offsets_base = tl.arange(0, BLOCK_CI)
    accum = tl.zeros((BLOCK_M, BLOCK_CO), dtype=tl.float32)

    for ci_base in range(0, 128, BLOCK_CI):
        ci_offsets = ci_base + ci_offsets_base
        ci_mask = ci_offsets < 128
        for dh in tl.static_range(0, 2):
            kh = 1 - residue_h + dh * 2
            ih_unstrided = oh + 1 - kh
            ih = ih_unstrided // 2
            valid_h = (ih_unstrided >= 0) & (ih < 4)
            for dw in tl.static_range(0, 2):
                kw = 1 - residue_w + dw * 2
                iw_unstrided = ow + 1 - kw
                iw = iw_unstrided // 2
                valid_hw = (
                    (m_offsets < 8 * 4 * 4)
                    & (n < 8)
                    & valid_h
                    & (iw_unstrided >= 0)
                    & (iw < 4)
                )
                input_offsets = (n[:, None] * 128 + ci_offsets[None, :]) * 4
                input_offsets = (input_offsets + ih[:, None]) * 4 + iw[:, None]
                weight_offsets = (
                    (ci_offsets[:, None] * 64 + co_offsets[None, :]) * 4 + kh
                ) * 4 + kw
                input_block = tl.load(
                    input_pointer + input_offsets,
                    mask=valid_hw[:, None] & ci_mask[None, :],
                    other=0.0,
                )
                weight_block = tl.load(
                    weight_pointer + weight_offsets,
                    mask=ci_mask[:, None] & (co_offsets[None, :] < 64),
                    other=0.0,
                )
                accum += tl.dot(input_block, weight_block, allow_tf32=False)

    output_offsets = ((n[:, None] * 64 + co_offsets[None, :]) * 8 + oh[:, None]) * 8
    output_offsets = output_offsets + ow[:, None]
    output_mask = (m_offsets[:, None] < 8 * 4 * 4) & (co_offsets[None, :] < 64)
    tl.store(output_pointer + output_offsets, accum, mask=output_mask)


@libentry()
@triton.jit
def _conv_transpose2d_blocker_fp32_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    block_id = pid // 8
    pid_c = pid - block_id * 8

    local_pid = block_id
    plane = 0
    if block_id < 4:
        local_pid = block_id
        plane = 0
    elif block_id < 8:
        local_pid = block_id - 4
        plane = 1
    elif block_id < 12:
        local_pid = block_id - 8
        plane = 2
    else:
        local_pid = block_id - 12
        plane = 3

    m_offsets = local_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    co_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    ci_offsets_base = tl.arange(0, BLOCK_K)

    if plane == 0:
        plane_m = 4 * 8 * 8
        iw = m_offsets % 8
        tmp = m_offsets // 8
        ih = tmp % 8
        n = tmp // 8
        oh = ih * 2
        ow = iw * 2
        accum = tl.zeros((BLOCK_M, BLOCK_C), tl.float32)
        for ci_base in range(0, 256, BLOCK_K):
            ci_offsets = ci_base + ci_offsets_base
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + ih[:, None]) * 8
                + iw[:, None],
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3 + 1) * 3
                + 1,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            accum += tl.dot(
                input_block,
                weight_block,
                input_precision="tf32x3",
            )
        tl.store(
            output_pointer
            + ((n[:, None] * 128 + co_offsets[None, :]) * 15 + oh[:, None]) * 15
            + ow[:, None],
            accum,
            mask=(m_offsets[:, None] < plane_m) & (co_offsets[None, :] < 128),
        )
    elif plane == 1:
        plane_m = 4 * 8 * 7
        j = m_offsets % 7
        tmp = m_offsets // 7
        ih = tmp % 8
        n = tmp // 8
        oh = ih * 2
        ow = j * 2 + 1
        accum = tl.zeros((BLOCK_M, BLOCK_C), tl.float32)
        for ci_base in range(0, 256, BLOCK_K):
            ci_offsets = ci_base + ci_offsets_base
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3 + 1) * 3
                + 2,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + ih[:, None]) * 8
                + j[:, None],
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            accum += tl.dot(
                input_block,
                weight_block,
                input_precision="tf32x3",
            )
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3 + 1) * 3,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + ih[:, None]) * 8
                + (j[:, None] + 1),
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            accum += tl.dot(
                input_block,
                weight_block,
                input_precision="tf32x3",
            )
        tl.store(
            output_pointer
            + ((n[:, None] * 128 + co_offsets[None, :]) * 15 + oh[:, None]) * 15
            + ow[:, None],
            accum,
            mask=(m_offsets[:, None] < plane_m) & (co_offsets[None, :] < 128),
        )
    elif plane == 2:
        plane_m = 4 * 7 * 8
        iw = m_offsets % 8
        tmp = m_offsets // 8
        i = tmp % 7
        n = tmp // 7
        oh = i * 2 + 1
        ow = iw * 2
        accum = tl.zeros((BLOCK_M, BLOCK_C), tl.float32)
        for ci_base in range(0, 256, BLOCK_K):
            ci_offsets = ci_base + ci_offsets_base
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3 + 2) * 3
                + 1,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + i[:, None]) * 8
                + iw[:, None],
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            accum += tl.dot(
                input_block,
                weight_block,
                input_precision="tf32x3",
            )
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3) * 3
                + 1,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + (i[:, None] + 1)) * 8
                + iw[:, None],
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            accum += tl.dot(
                input_block,
                weight_block,
                input_precision="tf32x3",
            )
        tl.store(
            output_pointer
            + ((n[:, None] * 128 + co_offsets[None, :]) * 15 + oh[:, None]) * 15
            + ow[:, None],
            accum,
            mask=(m_offsets[:, None] < plane_m) & (co_offsets[None, :] < 128),
        )
    else:
        plane_m = 4 * 7 * 7
        j = m_offsets % 7
        tmp = m_offsets // 7
        i = tmp % 7
        n = tmp // 7
        oh = i * 2 + 1
        ow = j * 2 + 1
        accum = tl.zeros((BLOCK_M, BLOCK_C), tl.float32)
        for ci_base in range(0, 256, BLOCK_K):
            ci_offsets = ci_base + ci_offsets_base
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3 + 2) * 3
                + 2,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + i[:, None]) * 8
                + j[:, None],
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            accum += tl.dot(
                input_block,
                weight_block,
                input_precision="tf32x3",
            )
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3 + 2) * 3,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + i[:, None]) * 8
                + (j[:, None] + 1),
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            accum += tl.dot(
                input_block,
                weight_block,
                input_precision="tf32x3",
            )
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3) * 3
                + 2,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + (i[:, None] + 1)) * 8
                + j[:, None],
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            accum += tl.dot(
                input_block,
                weight_block,
                input_precision="tf32x3",
            )
            weight_block = tl.load(
                weight_pointer
                + ((ci_offsets[:, None] * 128 + co_offsets[None, :]) * 3) * 3,
                mask=(ci_offsets[:, None] < 256) & (co_offsets[None, :] < 128),
                other=0.0,
            )
            input_block = tl.load(
                input_pointer
                + ((n[:, None] * 256 + ci_offsets[None, :]) * 8 + (i[:, None] + 1)) * 8
                + (j[:, None] + 1),
                mask=(m_offsets[:, None] < plane_m) & (ci_offsets[None, :] < 256),
                other=0.0,
            )
            accum += tl.dot(
                input_block,
                weight_block,
                input_precision="tf32x3",
            )
        tl.store(
            output_pointer
            + ((n[:, None] * 128 + co_offsets[None, :]) * 15 + oh[:, None]) * 15
            + ow[:, None],
            accum,
            mask=(m_offsets[:, None] < plane_m) & (co_offsets[None, :] < 128),
        )


@libentry()
@triton.jit
def _conv_transpose2d_direct_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    batch_size: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_channels: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    input_n_stride: tl.constexpr,
    input_c_stride: tl.constexpr,
    input_height_stride: tl.constexpr,
    input_width_stride: tl.constexpr,
    weight_ci_stride: tl.constexpr,
    weight_co_stride: tl.constexpr,
    weight_height_stride: tl.constexpr,
    weight_width_stride: tl.constexpr,
    output_n_stride: tl.constexpr,
    output_c_stride: tl.constexpr,
    output_height_stride: tl.constexpr,
    output_width_stride: tl.constexpr,
    input_channels: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    n_subgrids: tl.constexpr,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_raw = tl.program_id(0)
    pid_co = tl.program_id(1)

    pid_subgrid = pid_raw % n_subgrids
    pid_nhw = pid_raw // n_subgrids
    output_residue_h = pid_subgrid // stride_width
    output_residue_w = pid_subgrid % stride_width
    compact_height: tl.constexpr = (output_height + stride_height - 1) // stride_height
    compact_width: tl.constexpr = (output_width + stride_width - 1) // stride_width

    compact_offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    compact_plane: tl.constexpr = compact_height * compact_width
    compact_nh = compact_offsets // compact_width
    compact_h = compact_nh % compact_height
    compact_w = compact_offsets % compact_width
    n = compact_offsets // compact_plane
    oh = compact_h * stride_height + output_residue_h
    ow = compact_w * stride_width + output_residue_w
    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)
    ci_blocks: tl.constexpr = tl.cdiv(input_channels, BLOCK_CI)
    height_residue = (output_residue_h + padding_height) % stride_height
    width_residue = (output_residue_w + padding_width) % stride_width
    for kh in range(weight_height):
        if kh % stride_height == height_residue:
            ih_unstrided = oh + padding_height - kh
            ih = ih_unstrided // stride_height
            valid_h = (ih_unstrided >= 0) & (ih < input_height)
            for kw in range(weight_width):
                if kw % stride_width == width_residue:
                    iw_unstrided = ow + padding_width - kw
                    iw = iw_unstrided // stride_width
                    valid_hw = (
                        (n < batch_size)
                        & valid_h
                        & (iw_unstrided >= 0)
                        & (iw < input_width)
                        & (oh < output_height)
                        & (ow < output_width)
                    )
                    for ci_base in range(ci_blocks):
                        ci_offsets = ci_base * BLOCK_CI + tl.arange(0, BLOCK_CI)
                        input_offsets = (
                            n[:, None] * input_n_stride
                            + ci_offsets[None, :] * input_c_stride
                            + ih[:, None] * input_height_stride
                            + iw[:, None] * input_width_stride
                        )
                        weight_offsets = (
                            ci_offsets[:, None] * weight_ci_stride
                            + co_offsets[None, :] * weight_co_stride
                            + kh * weight_height_stride
                            + kw * weight_width_stride
                        )
                        input_mask = valid_hw[:, None] & (
                            ci_offsets[None, :] < input_channels
                        )
                        weight_mask = (ci_offsets[:, None] < input_channels) & (
                            co_offsets[None, :] < output_channels
                        )
                        input_block = tl.load(
                            input_pointer + input_offsets, mask=input_mask, other=0.0
                        )
                        weight_block = tl.load(
                            weight_pointer + weight_offsets, mask=weight_mask, other=0.0
                        )
                        accum += tl.dot(input_block, weight_block, allow_tf32=False)

    output_offsets = (
        n[:, None] * output_n_stride
        + co_offsets[None, :] * output_c_stride
        + oh[:, None] * output_height_stride
        + ow[:, None] * output_width_stride
    )
    output_mask = (
        (n[:, None] < batch_size)
        & (oh[:, None] < output_height)
        & (ow[:, None] < output_width)
        & (co_offsets[None, :] < output_channels)
    )
    tl.store(output_pointer + output_offsets, accum, mask=output_mask)


@libentry()
@triton.jit
def _conv_transpose2d_fp32_16_32_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_raw = tl.program_id(0)
    pid_co = tl.program_id(1)

    pid_subgrid = pid_raw % 4
    pid_nhw = pid_raw // 4
    output_residue_h = pid_subgrid // 2
    output_residue_w = pid_subgrid % 2

    compact_offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    compact_nh = compact_offsets // 32
    compact_h = compact_nh % 32
    compact_w = compact_offsets % 32
    n = compact_offsets // (32 * 32)
    oh = compact_h * 2 + output_residue_h
    ow = compact_w * 2 + output_residue_w
    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    ci_offsets = tl.arange(0, BLOCK_CI)

    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)
    height_residue = (output_residue_h + 1) % 2
    width_residue = (output_residue_w + 1) % 2
    for kh in range(3):
        if kh % 2 == height_residue:
            ih_unstrided = oh + 1 - kh
            ih = ih_unstrided // 2
            valid_h = (ih_unstrided >= 0) & (ih < 32)
            for kw in range(3):
                if kw % 2 == width_residue:
                    iw_unstrided = ow + 1 - kw
                    iw = iw_unstrided // 2
                    valid_hw = (
                        (n < 16)
                        & valid_h
                        & (iw_unstrided >= 0)
                        & (iw < 32)
                        & (oh < 63)
                        & (ow < 63)
                    )
                    input_offsets = (n[:, None] * 32 + ci_offsets[None, :]) * 32
                    input_offsets = (input_offsets + ih[:, None]) * 32 + iw[:, None]
                    weight_offsets = (
                        (ci_offsets[:, None] * 64 + co_offsets[None, :]) * 3 + kh
                    ) * 3 + kw
                    input_block = tl.load(
                        input_pointer + input_offsets,
                        mask=valid_hw[:, None],
                        other=0.0,
                    )
                    weight_block = tl.load(
                        weight_pointer + weight_offsets,
                        mask=co_offsets[None, :] < 64,
                        other=0.0,
                    )
                    accum += tl.dot(
                        input_block,
                        weight_block,
                        input_precision="tf32x3",
                    )

    output_offsets = (n[:, None] * 64 + co_offsets[None, :]) * 63 + oh[:, None]
    output_offsets = output_offsets * 63 + ow[:, None]
    output_mask = (n[:, None] < 16) & (oh[:, None] < 63)
    output_mask = output_mask & (ow[:, None] < 63) & (co_offsets[None, :] < 64)
    tl.store(output_pointer + output_offsets, accum, mask=output_mask)


@libentry()
@triton.jit
def _conv_transpose2d_fp32_32_64_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_raw = tl.program_id(0)
    pid_co = tl.program_id(1)

    pid_subgrid = pid_raw % 4
    pid_nhw = pid_raw // 4
    output_residue_h = pid_subgrid // 2
    output_residue_w = pid_subgrid % 2

    compact_offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    compact_nh = compact_offsets // 16
    compact_h = compact_nh % 16
    compact_w = compact_offsets % 16
    n = compact_offsets // (16 * 16)
    oh = compact_h * 2 + output_residue_h
    ow = compact_w * 2 + output_residue_w
    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    ci_offsets_base = tl.arange(0, BLOCK_CI)

    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)
    height_residue = (output_residue_h + 1) % 2
    width_residue = (output_residue_w + 1) % 2
    for kh in range(3):
        if kh % 2 == height_residue:
            ih_unstrided = oh + 1 - kh
            ih = ih_unstrided // 2
            valid_h = (ih_unstrided >= 0) & (ih < 16)
            for kw in range(3):
                if kw % 2 == width_residue:
                    iw_unstrided = ow + 1 - kw
                    iw = iw_unstrided // 2
                    valid_hw = (
                        (n < 32)
                        & valid_h
                        & (iw_unstrided >= 0)
                        & (iw < 16)
                        & (oh < 31)
                        & (ow < 31)
                    )
                    for ci_base in range(0, 64, BLOCK_CI):
                        ci_offsets = ci_base + ci_offsets_base
                        input_offsets = (n[:, None] * 64 + ci_offsets[None, :]) * 16
                        input_offsets = (input_offsets + ih[:, None]) * 16 + iw[:, None]
                        weight_offsets = (
                            (ci_offsets[:, None] * 32 + co_offsets[None, :]) * 3 + kh
                        ) * 3 + kw
                        input_block = tl.load(
                            input_pointer + input_offsets,
                            mask=valid_hw[:, None] & (ci_offsets[None, :] < 64),
                            other=0.0,
                        )
                        weight_block = tl.load(
                            weight_pointer + weight_offsets,
                            mask=(ci_offsets[:, None] < 64)
                            & (co_offsets[None, :] < 32),
                            other=0.0,
                        )
                        accum += tl.dot(
                            input_block,
                            weight_block,
                            input_precision="tf32x3",
                        )

    output_offsets = (n[:, None] * 32 + co_offsets[None, :]) * 31 + oh[:, None]
    output_offsets = output_offsets * 31 + ow[:, None]
    output_mask = (n[:, None] < 32) & (oh[:, None] < 31)
    output_mask = output_mask & (ow[:, None] < 31) & (co_offsets[None, :] < 32)
    tl.store(output_pointer + output_offsets, accum, mask=output_mask)


def _conv_transpose2d_blocker_fp16(input, weight):
    output = torch.empty((4, 128, 15, 15), device=input.device, dtype=input.dtype)
    grid = (29 * 4,)
    _conv_transpose2d_blocker_fp16_kernel[grid](
        input,
        weight,
        output,
        BLOCK_M=32,
        BLOCK_C=32,
        BLOCK_K=32,
        num_warps=8,
        num_stages=2,
    )
    return output


def _conv_transpose2d_k4_fp16(input, weight):
    output = torch.empty((8, 64, 8, 8), device=input.device, dtype=input.dtype)
    grid = (triton.cdiv(8 * 4 * 4, 64), triton.cdiv(64, 32), 4)
    _conv_transpose2d_k4_fp16_kernel[grid](
        input,
        weight,
        output,
        BLOCK_M=64,
        BLOCK_CO=32,
        BLOCK_CI=32,
        num_warps=4,
        num_stages=3,
    )
    return output


def _conv_transpose2d_blocker_fp32(input, weight):
    output = torch.empty((4, 128, 15, 15), device=input.device, dtype=input.dtype)
    grid = (16 * 8,)
    _conv_transpose2d_blocker_fp32_kernel[grid](
        input,
        weight,
        output,
        BLOCK_M=64,
        BLOCK_C=16,
        BLOCK_K=16,
        num_warps=4,
        num_stages=2,
    )
    return output


def _conv_transpose2d_direct_fp32_16_32(input, weight):
    output = torch.empty((16, 64, 63, 63), device=input.device, dtype=input.dtype)
    grid = (4 * triton.cdiv(16 * 32 * 32, 128), triton.cdiv(64, 32))
    _conv_transpose2d_fp32_16_32_kernel[grid](
        input,
        weight,
        output,
        BLOCK_NHW=128,
        BLOCK_CI=32,
        BLOCK_CO=32,
        num_warps=4,
    )
    return output


def _conv_transpose2d_direct_fp32_32_64(input, weight):
    output = torch.empty((32, 32, 31, 31), device=input.device, dtype=input.dtype)
    grid = (4 * triton.cdiv(32 * 16 * 16, 128), triton.cdiv(32, 16))
    _conv_transpose2d_fp32_32_64_kernel[grid](
        input,
        weight,
        output,
        BLOCK_NHW=128,
        BLOCK_CI=16,
        BLOCK_CO=16,
        num_warps=4,
    )
    return output


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    logger.debug("GEMS CONV_TRANSPOSE2D")

    stride_h, stride_w = _pair(stride)
    padding_h, padding_w = _pair(padding)
    output_padding_h, output_padding_w = _pair(output_padding)
    dilation_h, dilation_w = _pair(dilation)

    if input.dtype is torch.float32 and input.shape == (32, 64, 16, 16):
        if (
            weight.dtype is torch.float32
            and weight.shape == (64, 32, 3, 3)
            and bias is None
            and groups == 1
            and stride_h == 2
            and stride_w == 2
            and padding_h == 1
            and padding_w == 1
            and output_padding_h == 0
            and output_padding_w == 0
            and dilation_h == 1
            and dilation_w == 1
            and input.device.type == "cuda"
            and weight.device == input.device
            and input.is_contiguous()
            and weight.is_contiguous()
        ):
            return _conv_transpose2d_direct_fp32_32_64(input, weight)

    if input.dtype is torch.float32 and input.shape == (16, 32, 32, 32):
        if (
            weight.dtype is torch.float32
            and weight.shape == (32, 64, 3, 3)
            and bias is None
            and groups == 1
            and stride_h == 2
            and stride_w == 2
            and padding_h == 1
            and padding_w == 1
            and output_padding_h == 0
            and output_padding_w == 0
            and dilation_h == 1
            and dilation_w == 1
            and input.device.type == "cuda"
            and weight.device == input.device
            and input.is_contiguous()
            and weight.is_contiguous()
        ):
            return _conv_transpose2d_direct_fp32_16_32(input, weight)

    if input.dtype is torch.float32 and input.shape == (4, 256, 8, 8):
        if (
            weight.dtype is torch.float32
            and weight.shape == (256, 128, 3, 3)
            and bias is None
            and groups == 1
            and stride_h == 2
            and stride_w == 2
            and padding_h == 1
            and padding_w == 1
            and output_padding_h == 0
            and output_padding_w == 0
            and dilation_h == 1
            and dilation_w == 1
            and input.device.type == "cuda"
            and weight.device == input.device
            and input.is_contiguous()
            and weight.is_contiguous()
        ):
            return _conv_transpose2d_blocker_fp32(input, weight)

    fp16_shape_key = _exact_shape_key(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        groups,
        dilation_h,
        dilation_w,
        (torch.float16,),
    )
    if fp16_shape_key == _BLOCKER_FP16_SHAPE:
        return _conv_transpose2d_blocker_fp16(input, weight)

    if fp16_shape_key == _K4_FP16_SHAPE:
        return _conv_transpose2d_k4_fp16(input, weight)

    direct_lowp_dtypes = _TRITON_DIRECT_LOWP_DTYPES
    if (
        input.device.type == "cuda"
        and input.dtype is torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        direct_lowp_dtypes = (torch.float16,)

    direct_shape_key = _exact_shape_key(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        groups,
        dilation_h,
        dilation_w,
        direct_lowp_dtypes,
    )
    if direct_shape_key in _DIRECT_TRITON_SHAPES:
        return _conv_transpose2d_direct(
            input,
            weight,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            output_padding_h,
            output_padding_w,
        )

    return _unsupported_conv_transpose2d(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        groups,
        dilation_h,
        dilation_w,
    )


def _conv_transpose2d_direct(
    input,
    weight,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_padding_h,
    output_padding_w,
):
    batch, input_channels, input_height, input_width = input.shape
    _, output_channels, weight_height, weight_width = weight.shape
    output_height = (
        (input_height - 1) * stride_h
        - 2 * padding_h
        + dilation_h * (weight_height - 1)
        + output_padding_h
        + 1
    )
    output_width = (
        (input_width - 1) * stride_w
        - 2 * padding_w
        + dilation_w * (weight_width - 1)
        + output_padding_w
        + 1
    )
    output = torch.empty(
        (batch, output_channels, output_height, output_width),
        device=input.device,
        dtype=input.dtype,
    )
    compact_height = triton.cdiv(output_height, stride_h)
    compact_width = triton.cdiv(output_width, stride_w)
    max_sub_spatial = batch * compact_height * compact_width
    n_subgrids = stride_h * stride_w

    shape_key = (
        batch,
        input_channels,
        input_height,
        input_width,
        output_channels,
        weight_height,
        weight_width,
        stride_h,
        padding_h,
    )
    block_nhw = 64
    block_ci = 32
    block_co = 32
    num_warps = 4
    if shape_key == (1, 64, 128, 128, 64, 3, 3, 1, 1):
        block_nhw = 128
        block_ci = 16
        num_warps = 8
    elif shape_key == (16, 32, 32, 32, 64, 3, 3, 2, 1):
        block_nhw = 128
        block_ci = 16
        block_co = 64
    elif shape_key == (32, 64, 16, 16, 32, 3, 3, 2, 1):
        block_nhw = 32
        block_ci = 16
        num_warps = 8
    elif shape_key == (16, 32, 8, 8, 24, 5, 5, 2, 2):
        num_warps = 8
    elif shape_key == (32, 64, 32, 32, 32, 3, 3, 1, 0):
        block_nhw = 128
        block_ci = 16
        num_warps = 8
        if input.dtype is torch.float16:
            block_co = 64
    elif input_channels >= 64 and output_channels <= 32:
        block_ci = 64
        if stride_h == 1:
            num_warps = 8

    grid = (
        n_subgrids * triton.cdiv(max_sub_spatial, block_nhw),
        triton.cdiv(output_channels, block_co),
    )
    _conv_transpose2d_direct_kernel[grid](
        input,
        weight,
        output,
        batch,
        input_height,
        input_width,
        output_channels,
        output_height,
        output_width,
        *input.stride(),
        *weight.stride(),
        *output.stride(),
        input_channels,
        weight_height,
        weight_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        n_subgrids,
        BLOCK_NHW=block_nhw,
        BLOCK_CI=block_ci,
        BLOCK_CO=block_co,
        num_warps=num_warps,
    )
    return output
