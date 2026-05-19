import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_TRITON_DIRECT_LOWP_DTYPES = (torch.float16, torch.bfloat16)

# Schedule-specialized high-impact rows kept as exact microkernels.
_K3_S2_P1_N4_CI256_CO128_HW8_SHAPE = (4, 256, 8, 8, 128, 3, 3, 2, 1)
_K4_S2_P1_N8_CI128_CO64_HW4_SHAPE = (8, 128, 4, 4, 64, 4, 4, 2, 1)
_K3_S2_P1_N16_CI32_CO64_HW16_SHAPE = (16, 32, 16, 16, 64, 3, 3, 2, 1)
_K3_S2_P1_N16_CI32_CO64_HW32_SHAPE = (16, 32, 32, 32, 64, 3, 3, 2, 1)
_K3_S2_P1_N32_CI64_CO32_HW16_SHAPE = (32, 64, 16, 16, 32, 3, 3, 2, 1)
_K3_S1_P1_N1_CI64_CO64_HW128_SHAPE = (1, 64, 128, 128, 64, 3, 3, 1, 1)
_K3_S2_P1_N1_CI64_CO32_HW64_SHAPE = (1, 64, 64, 64, 32, 3, 3, 2, 1)
_K3_S2_P1_N4_CI32_CO32_HW32_SHAPE = (4, 32, 32, 32, 32, 3, 3, 2, 1)
_K5_S2_P2_N8_CI16_CO16_HW64_SHAPE = (8, 16, 64, 64, 16, 5, 5, 2, 2)
_K5_S2_P2_N16_CI32_CO24_HW8_SHAPE = (16, 32, 8, 8, 24, 5, 5, 2, 2)
_K3_S1_P0_N32_CI64_CO32_HW32_SHAPE = (32, 64, 32, 32, 32, 3, 3, 1, 0)

_GENERAL_TRITON_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_DIRECT_TILED_FAMILY_MAX_CHANNELS = 256
_DIRECT_TILED_FAMILY_MAX_KERNEL = 5
_DIRECT_TILED_FAMILY_MAX_STRIDE = 4
_DIRECT_TILED_OUTPUT_PADDING_MIN_INPUT_ELEMENTS = 1024
_DIRECT_TILED_DEFAULT_SCHEDULE = (64, 32, 32, 4)
_DIRECT_TILED_EXACT_SCHEDULES = {
    (_K3_S1_P1_N1_CI64_CO64_HW128_SHAPE, torch.float32): (256, 16, 32, 4),
    (_K3_S1_P1_N1_CI64_CO64_HW128_SHAPE, torch.float16): (128, 16, 32, 8),
    (_K3_S1_P1_N1_CI64_CO64_HW128_SHAPE, torch.bfloat16): (128, 16, 32, 8),
    (_K3_S2_P1_N1_CI64_CO32_HW64_SHAPE, None): (128, 16, 16, 4),
    (_K3_S2_P1_N4_CI32_CO32_HW32_SHAPE, None): (64, 16, 16, 4),
    (_K5_S2_P2_N8_CI16_CO16_HW64_SHAPE, None): (128, 16, 16, 8),
    (_K3_S2_P1_N16_CI32_CO64_HW16_SHAPE, None): (32, 16, 16, 8),
    (_K3_S2_P1_N16_CI32_CO64_HW32_SHAPE, None): (128, 16, 64, 4),
    (_K3_S2_P1_N32_CI64_CO32_HW16_SHAPE, None): (32, 16, 32, 8),
    (_K5_S2_P2_N16_CI32_CO24_HW8_SHAPE, None): (64, 32, 32, 8),
    (_K3_S1_P0_N32_CI64_CO32_HW32_SHAPE, None): (256, 16, 32, 8),
}


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


def _direct_tiled_family_shape_key(
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
    if bias is not None or groups != 1:
        return None
    if (dilation_h, dilation_w) != (1, 1):
        return None
    if input.dtype not in _GENERAL_TRITON_DTYPES or weight.dtype != input.dtype:
        return None
    if input.device.type != "cuda" or weight.device != input.device:
        return None
    if input.dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return None
    if input.dim() != 4 or weight.dim() != 4:
        return None
    if not input.is_contiguous() or not weight.is_contiguous():
        return None
    if stride_h != stride_w or padding_h != padding_w:
        return None
    if output_padding_h != output_padding_w:
        return None
    if stride_h <= 0 or stride_h > _DIRECT_TILED_FAMILY_MAX_STRIDE:
        return None
    if padding_h < 0 or output_padding_h < 0:
        return None

    batch, input_channels, input_height, input_width = input.shape
    weight_input_channels, output_channels, weight_height, weight_width = weight.shape
    if batch <= 0 or input_height <= 0 or input_width <= 0:
        return None
    if input_channels != weight_input_channels:
        return None
    if input_channels < 16 or output_channels < 16:
        return None
    if (
        input_channels > _DIRECT_TILED_FAMILY_MAX_CHANNELS
        or output_channels > _DIRECT_TILED_FAMILY_MAX_CHANNELS
    ):
        return None
    if (
        weight_height <= 0
        or weight_width <= 0
        or weight_height > _DIRECT_TILED_FAMILY_MAX_KERNEL
        or weight_width > _DIRECT_TILED_FAMILY_MAX_KERNEL
    ):
        return None
    output_height = (
        (input_height - 1) * stride_h - 2 * padding_h + weight_height + output_padding_h
    )
    output_width = (
        (input_width - 1) * stride_w - 2 * padding_w + weight_width + output_padding_w
    )
    if output_height <= 0 or output_width <= 0:
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


def _can_use_direct_tiled_family(
    input,
    direct_tiled_family_shape_key,
    output_padding_h,
):
    if direct_tiled_family_shape_key is None:
        return False
    (
        batch,
        input_channels,
        input_height,
        input_width,
        output_channels,
        weight_height,
        weight_width,
        stride_h,
        _padding_h,
    ) = direct_tiled_family_shape_key

    if output_padding_h == 0 and stride_h <= 2:
        return True
    input_elements = batch * input_height * input_width
    if (
        input.dtype in _GENERAL_TRITON_DTYPES
        and stride_h == 2
        and output_padding_h == 1
        and weight_height == 3
        and weight_width == 3
        and input_channels >= 64
        and output_channels <= 64
        and input_elements >= _DIRECT_TILED_OUTPUT_PADDING_MIN_INPUT_ELEMENTS
    ):
        return True
    if stride_h >= 3 and output_padding_h == 0:
        if weight_height >= 5 or weight_width >= 5:
            return True
        if input.dtype in _TRITON_DIRECT_LOWP_DTYPES:
            return True
    return False


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
        "flag_gems.conv_transpose2d supports 3D or 4D CUDA input tensors "
        "and 4D CUDA weight tensors with float32, float16, or bfloat16 dtype; got "
        f"input_shape={tuple(input.shape)}, weight_shape={tuple(weight.shape)}, "
        f"input_dtype={input.dtype}, weight_dtype={weight.dtype}, bias_dtype={bias_dtype}, "
        f"input_device={input.device}, weight_device={weight.device}, "
        f"stride=({stride_h}, {stride_w}), padding=({padding_h}, {padding_w}), "
        f"output_padding=({output_padding_h}, {output_padding_w}), groups={groups}, "
        f"dilation=({dilation_h}, {dilation_w})"
    )


def _validate_conv_transpose2d_args(
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
    if input.device.type != "cuda" or weight.device != input.device:
        return False
    if input.dim() != 4 or weight.dim() != 4:
        return False
    if not input.is_contiguous() or not weight.is_contiguous():
        return False
    if input.dtype not in _GENERAL_TRITON_DTYPES or weight.dtype != input.dtype:
        return False
    if input.dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return False
    if bias is not None:
        if bias.device != input.device or bias.dtype != input.dtype:
            return False
        if bias.dim() != 1 or not bias.is_contiguous():
            return False
    if groups <= 0:
        raise RuntimeError("groups must be a positive integer")
    if stride_h <= 0 or stride_w <= 0:
        raise RuntimeError("non-positive stride is not supported")
    if dilation_h <= 0 or dilation_w <= 0:
        raise RuntimeError("dilation should be greater than zero")
    if padding_h < 0 or padding_w < 0:
        raise RuntimeError("negative padding is not supported")
    if output_padding_h < 0 or output_padding_w < 0:
        raise RuntimeError("negative output_padding is not supported")
    if output_padding_h >= stride_h and output_padding_h >= dilation_h:
        raise RuntimeError(
            "output padding must be smaller than either stride or dilation"
        )
    if output_padding_w >= stride_w and output_padding_w >= dilation_w:
        raise RuntimeError(
            "output padding must be smaller than either stride or dilation"
        )

    input_channels = input.shape[1]
    weight_input_channels = weight.shape[0]
    output_channels_per_group = weight.shape[1]
    weight_height = weight.shape[2]
    weight_width = weight.shape[3]
    if (
        input_channels <= 0
        or output_channels_per_group <= 0
        or weight_height <= 0
        or weight_width <= 0
    ):
        raise RuntimeError(
            "non-empty input channels and weight dimensions are required"
        )
    if input_channels != weight_input_channels:
        raise RuntimeError(
            "expected input channel dimension to match weight input channels"
        )
    if input_channels % groups != 0:
        raise RuntimeError("input channels must be divisible by groups")
    output_channels = output_channels_per_group * groups
    if bias is not None and bias.numel() != output_channels:
        raise RuntimeError("expected bias to have one element per output channel")

    input_height = input.shape[2]
    input_width = input.shape[3]
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
    if output_height <= 0 or output_width <= 0:
        raise RuntimeError("calculated output size is too small")
    return True


def _can_use_scatter_no_overlap(
    input,
    weight,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    groups,
):
    batch, input_channels, input_height, input_width = input.shape
    _, output_channels_per_group, weight_height, weight_width = weight.shape
    if batch <= 0 or input_height <= 0 or input_width <= 0:
        return False
    if stride_h < 3 or stride_w < 3:
        return False

    effective_kernel_h = (weight_height - 1) * dilation_h + 1
    effective_kernel_w = (weight_width - 1) * dilation_w + 1
    if stride_h < effective_kernel_h or stride_w < effective_kernel_w:
        return False

    input_channels_per_group = input_channels // groups
    if input_channels_per_group > 128 or output_channels_per_group > 128:
        return False
    return weight_height * weight_width <= 25


@libentry()
@triton.jit
def _conv_transpose2d_k3_s2_p1_n4_ci256_co128_hw8_lowp_kernel(
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
def _conv_transpose2d_k4_s2_p1_n8_ci128_co64_hw4_lowp_kernel(
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
def _conv_transpose2d_k3_s2_p1_n4_ci256_co128_hw8_fp32_kernel(
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
                        accum += tl.dot(
                            input_block,
                            weight_block,
                            input_precision="tf32x3",
                        )

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
def _conv_transpose2d_k3_s1_p1_n1_ci64_co64_hw128_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    pid_co = tl.program_id(1)

    offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    oh = offsets // 128
    ow = offsets - oh * 128
    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    ci_offsets_base = tl.arange(0, BLOCK_CI)

    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)
    for kh in range(3):
        ih = oh + kh - 1
        valid_h = (ih >= 0) & (ih < 128)
        weight_kh: tl.constexpr = 2 - kh
        for kw in range(3):
            iw = ow + kw - 1
            valid_hw = valid_h & (iw >= 0) & (iw < 128)
            weight_kw: tl.constexpr = 2 - kw
            for ci_base in range(0, 64, BLOCK_CI):
                ci_offsets = ci_base + ci_offsets_base
                input_offsets = (ci_offsets[None, :] * 128 + ih[:, None]) * 128
                input_offsets = input_offsets + iw[:, None]
                weight_offsets = (
                    (ci_offsets[:, None] * 64 + co_offsets[None, :]) * 3 + weight_kh
                ) * 3 + weight_kw
                input_block = tl.load(
                    input_pointer + input_offsets,
                    mask=valid_hw[:, None],
                    other=0.0,
                )
                weight_block = tl.load(weight_pointer + weight_offsets)
                accum += tl.dot(
                    input_block,
                    weight_block,
                    input_precision="tf32x3",
                )

    output_offsets = (co_offsets[None, :] * 128 + oh[:, None]) * 128 + ow[:, None]
    tl.store(output_pointer + output_offsets, accum)


@libentry()
@triton.jit
def _conv_transpose2d_k3_s2_p1_n16_ci32_co64_hw16_fp32_kernel(
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
    output_residue_w = pid_subgrid - output_residue_h * 2

    compact_offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    compact_nh = compact_offsets // 16
    compact_h = compact_nh % 16
    compact_w = compact_offsets - compact_nh * 16
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
                        (n < 16)
                        & valid_h
                        & (iw_unstrided >= 0)
                        & (iw < 16)
                        & (oh < 31)
                        & (ow < 31)
                    )
                    for ci_base in range(0, 32, BLOCK_CI):
                        ci_offsets = ci_base + ci_offsets_base
                        ci_mask = ci_offsets < 32
                        input_offsets = (n[:, None] * 32 + ci_offsets[None, :]) * 16
                        input_offsets = (input_offsets + ih[:, None]) * 16
                        input_offsets = input_offsets + iw[:, None]
                        weight_offsets = (
                            (ci_offsets[:, None] * 64 + co_offsets[None, :]) * 3 + kh
                        ) * 3 + kw
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
                        accum += tl.dot(
                            input_block,
                            weight_block,
                            input_precision="tf32x3",
                        )

    output_offsets = (n[:, None] * 64 + co_offsets[None, :]) * 31 + oh[:, None]
    output_offsets = output_offsets * 31 + ow[:, None]
    output_mask = (n[:, None] < 16) & (oh[:, None] < 31)
    output_mask = output_mask & (ow[:, None] < 31) & (co_offsets[None, :] < 64)
    tl.store(output_pointer + output_offsets, accum, mask=output_mask)


@libentry()
@triton.jit
def _conv_transpose2d_k3_s2_p1_n16_ci32_co64_hw32_fp32_kernel(
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
def _conv_transpose2d_residue_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    batch_size: tl.constexpr,
    input_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_channels: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    output_channels_per_group: tl.constexpr,
    input_channels_per_group: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    has_bias: tl.constexpr,
    n_subgrids: tl.constexpr,
    co_blocks_per_group: tl.constexpr,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_raw = tl.program_id(0)
    pid_gco = tl.program_id(1)

    pid_subgrid = pid_raw % n_subgrids
    pid_nhw = pid_raw // n_subgrids
    output_residue_h = pid_subgrid // stride_width
    output_residue_w = pid_subgrid % stride_width
    compact_height: tl.constexpr = (output_height + stride_height - 1) // stride_height
    compact_width: tl.constexpr = (output_width + stride_width - 1) // stride_width
    compact_plane: tl.constexpr = compact_height * compact_width

    compact_offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    compact_nh = compact_offsets // compact_width
    compact_h = compact_nh % compact_height
    compact_w = compact_offsets % compact_width
    n = compact_offsets // compact_plane
    oh = compact_h * stride_height + output_residue_h
    ow = compact_w * stride_width + output_residue_w

    group = pid_gco // co_blocks_per_group
    pid_co_in_group = pid_gco - group * co_blocks_per_group
    co_in_offsets = pid_co_in_group * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_offsets = group * output_channels_per_group + co_in_offsets

    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)
    if has_bias:
        bias_values = tl.load(
            bias_pointer + co_offsets,
            mask=co_in_offsets < output_channels_per_group,
            other=0.0,
        ).to(tl.float32)
        accum += bias_values[None, :]

    ci_blocks: tl.constexpr = tl.cdiv(input_channels_per_group, BLOCK_CI)
    height_residue = (output_residue_h + padding_height) % stride_height
    width_residue = (output_residue_w + padding_width) % stride_width
    for kh in range(weight_height):
        kh_residue: tl.constexpr = (kh * dilation_height) % stride_height
        if kh_residue == height_residue:
            ih_unstrided = oh + padding_height - kh * dilation_height
            ih = ih_unstrided // stride_height
            valid_h = (n < batch_size) & (ih_unstrided >= 0) & (ih < input_height)
            for kw in range(weight_width):
                kw_residue: tl.constexpr = (kw * dilation_width) % stride_width
                if kw_residue == width_residue:
                    iw_unstrided = ow + padding_width - kw * dilation_width
                    iw = iw_unstrided // stride_width
                    valid_hw = (
                        valid_h
                        & (iw_unstrided >= 0)
                        & (iw < input_width)
                        & (oh < output_height)
                        & (ow < output_width)
                    )
                    for ci_base in range(ci_blocks):
                        ci_in_offsets = ci_base * BLOCK_CI + tl.arange(0, BLOCK_CI)
                        ci_offsets = group * input_channels_per_group + ci_in_offsets
                        input_offsets = (
                            n[:, None] * input_channels + ci_offsets[None, :]
                        ) * input_height
                        input_offsets = (
                            input_offsets + ih[:, None]
                        ) * input_width + iw[:, None]
                        weight_offsets = (
                            ci_offsets[:, None] * output_channels_per_group
                            + co_in_offsets[None, :]
                        ) * weight_height
                        weight_offsets = (weight_offsets + kh) * weight_width + kw
                        input_mask = valid_hw[:, None] & (
                            ci_in_offsets[None, :] < input_channels_per_group
                        )
                        weight_mask = (
                            ci_in_offsets[:, None] < input_channels_per_group
                        ) & (co_in_offsets[None, :] < output_channels_per_group)
                        input_block = tl.load(
                            input_pointer + input_offsets, mask=input_mask, other=0.0
                        )
                        weight_block = tl.load(
                            weight_pointer + weight_offsets, mask=weight_mask, other=0.0
                        )
                        accum += tl.dot(
                            input_block,
                            weight_block,
                            input_precision="tf32x3",
                        )

    output_offsets = n[:, None] * output_channels + co_offsets[None, :]
    output_offsets = (output_offsets * output_height + oh[:, None]) * output_width
    output_offsets = output_offsets + ow[:, None]
    output_mask = (
        (n[:, None] < batch_size)
        & (oh[:, None] < output_height)
        & (ow[:, None] < output_width)
        & (co_in_offsets[None, :] < output_channels_per_group)
        & (co_offsets[None, :] < output_channels)
    )
    tl.store(output_pointer + output_offsets, accum, mask=output_mask)


@libentry()
@triton.jit
def _conv_transpose2d_general_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    total_elements: tl.constexpr,
    batch_size: tl.constexpr,
    input_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_channels: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    output_channels_per_group: tl.constexpr,
    input_channels_per_group: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    tmp = offsets // output_width
    ow = offsets - tmp * output_width
    tmp2 = tmp // output_height
    oh = tmp - tmp2 * output_height
    n = tmp2 // output_channels
    co = tmp2 - n * output_channels

    group = co // output_channels_per_group
    co_in_group = co - group * output_channels_per_group
    accum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    if has_bias:
        bias = tl.load(bias_pointer + co, mask=mask, other=0.0).to(tl.float32)
        accum += bias

    for ci_in_group in tl.range(0, input_channels_per_group):
        ci = group * input_channels_per_group + ci_in_group
        for kh in tl.static_range(0, weight_height):
            ih_unstrided = oh + padding_height - kh * dilation_height
            ih = ih_unstrided // stride_height
            valid_h = (ih_unstrided % stride_height == 0) & (ih >= 0)
            valid_h = valid_h & (ih < input_height)
            for kw in tl.static_range(0, weight_width):
                iw_unstrided = ow + padding_width - kw * dilation_width
                iw = iw_unstrided // stride_width
                valid = mask & valid_h
                valid = valid & (iw_unstrided % stride_width == 0)
                valid = valid & (iw >= 0) & (iw < input_width)

                input_offsets = (n * input_channels + ci) * input_height + ih
                input_offsets = input_offsets * input_width + iw
                weight_offsets = (
                    ci * output_channels_per_group + co_in_group
                ) * weight_height
                weight_offsets = (weight_offsets + kh) * weight_width + kw
                input_values = tl.load(
                    input_pointer + input_offsets, mask=valid, other=0.0
                ).to(tl.float32)
                weight_values = tl.load(
                    weight_pointer + weight_offsets, mask=valid, other=0.0
                ).to(tl.float32)
                accum += input_values * weight_values

    tl.store(output_pointer + offsets, accum, mask=mask)


@libentry()
@triton.jit
def _conv_transpose2d_residue_static_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    batch_size: tl.constexpr,
    input_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_channels: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    compact_height: tl.constexpr,
    compact_width: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    output_channels_per_group: tl.constexpr,
    input_channels_per_group: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    has_bias: tl.constexpr,
    output_residue_h: tl.constexpr,
    output_residue_w: tl.constexpr,
    co_blocks_per_group: tl.constexpr,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    pid_gco = tl.program_id(1)

    compact_offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    compact_plane: tl.constexpr = compact_height * compact_width
    compact_nh = compact_offsets // compact_width
    compact_h = compact_nh % compact_height
    compact_w = compact_offsets % compact_width
    n = compact_offsets // compact_plane
    oh = compact_h * stride_height + output_residue_h
    ow = compact_w * stride_width + output_residue_w

    group = pid_gco // co_blocks_per_group
    pid_co_in_group = pid_gco - group * co_blocks_per_group
    co_in_offsets = pid_co_in_group * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_offsets = group * output_channels_per_group + co_in_offsets

    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)
    if has_bias:
        bias_values = tl.load(
            bias_pointer + co_offsets,
            mask=co_in_offsets < output_channels_per_group,
            other=0.0,
        ).to(tl.float32)
        accum += bias_values[None, :]

    ci_blocks: tl.constexpr = tl.cdiv(input_channels_per_group, BLOCK_CI)
    height_residue: tl.constexpr = (output_residue_h + padding_height) % stride_height
    width_residue: tl.constexpr = (output_residue_w + padding_width) % stride_width
    for kh in tl.static_range(0, weight_height):
        if (kh * dilation_height) % stride_height == height_residue:
            ih_unstrided = oh + padding_height - kh * dilation_height
            ih = ih_unstrided // stride_height
            valid_h = (n < batch_size) & (ih_unstrided >= 0) & (ih < input_height)
            for kw in tl.static_range(0, weight_width):
                if (kw * dilation_width) % stride_width == width_residue:
                    iw_unstrided = ow + padding_width - kw * dilation_width
                    iw = iw_unstrided // stride_width
                    valid_hw = (
                        valid_h
                        & (iw_unstrided >= 0)
                        & (iw < input_width)
                        & (oh < output_height)
                        & (ow < output_width)
                    )
                    for ci_base in range(ci_blocks):
                        ci_in_offsets = ci_base * BLOCK_CI + tl.arange(0, BLOCK_CI)
                        ci_offsets = group * input_channels_per_group + ci_in_offsets
                        input_offsets = (
                            n[:, None] * input_channels + ci_offsets[None, :]
                        ) * input_height
                        input_offsets = (
                            input_offsets + ih[:, None]
                        ) * input_width + iw[:, None]
                        weight_offsets = (
                            ci_offsets[:, None] * output_channels_per_group
                            + co_in_offsets[None, :]
                        ) * weight_height
                        weight_offsets = (weight_offsets + kh) * weight_width + kw
                        input_mask = valid_hw[:, None] & (
                            ci_in_offsets[None, :] < input_channels_per_group
                        )
                        weight_mask = (
                            ci_in_offsets[:, None] < input_channels_per_group
                        ) & (co_in_offsets[None, :] < output_channels_per_group)
                        input_block = tl.load(
                            input_pointer + input_offsets, mask=input_mask, other=0.0
                        )
                        weight_block = tl.load(
                            weight_pointer + weight_offsets, mask=weight_mask, other=0.0
                        )
                        accum += tl.dot(
                            input_block,
                            weight_block,
                            input_precision="tf32x3",
                        )

    output_offsets = n[:, None] * output_channels + co_offsets[None, :]
    output_offsets = (output_offsets * output_height + oh[:, None]) * output_width
    output_offsets = output_offsets + ow[:, None]
    output_mask = (
        (n[:, None] < batch_size)
        & (oh[:, None] < output_height)
        & (ow[:, None] < output_width)
        & (co_in_offsets[None, :] < output_channels_per_group)
        & (co_offsets[None, :] < output_channels)
    )
    tl.store(output_pointer + output_offsets, accum, mask=output_mask)


@libentry()
@triton.jit
def _conv_transpose2d_scatter_init_kernel(
    bias_pointer,
    output_pointer,
    total_elements: tl.constexpr,
    output_channels: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    values = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    if has_bias:
        spatial_size: tl.constexpr = output_height * output_width
        co = (offsets // spatial_size) % output_channels
        values = tl.load(bias_pointer + co, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_pointer + offsets, values, mask=mask)


@libentry()
@triton.jit
def _conv_transpose2d_scatter_no_overlap_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    batch_size: tl.constexpr,
    input_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_channels: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    output_channels_per_group: tl.constexpr,
    input_channels_per_group: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_gkk = tl.program_id(2)

    kw = pid_gkk % weight_width
    tmp = pid_gkk // weight_width
    kh = tmp % weight_height
    group = tmp // weight_height

    nhw_offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    iw = nhw_offsets % input_width
    tmp = nhw_offsets // input_width
    ih = tmp % input_height
    n = tmp // input_height

    oh = ih * stride_height - padding_height + kh * dilation_height
    ow = iw * stride_width - padding_width + kw * dilation_width
    valid_nhw = (nhw_offsets < batch_size * input_height * input_width) & (
        n < batch_size
    )
    valid_nhw = valid_nhw & (oh >= 0) & (oh < output_height)
    valid_nhw = valid_nhw & (ow >= 0) & (ow < output_width)

    co_in_group = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co = group * output_channels_per_group + co_in_group
    ci_in_group_base = tl.arange(0, BLOCK_CI)

    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)
    ci_blocks: tl.constexpr = tl.cdiv(input_channels_per_group, BLOCK_CI)
    for ci_block in range(ci_blocks):
        ci_in_group = ci_block * BLOCK_CI + ci_in_group_base
        ci = group * input_channels_per_group + ci_in_group
        input_offsets = (n[:, None] * input_channels + ci[None, :]) * input_height
        input_offsets = (input_offsets + ih[:, None]) * input_width + iw[:, None]
        weight_offsets = (
            ci[:, None] * output_channels_per_group + co_in_group[None, :]
        ) * weight_height
        weight_offsets = (weight_offsets + kh) * weight_width + kw

        ci_mask = ci_in_group < input_channels_per_group
        co_mask = co_in_group < output_channels_per_group
        input_block = tl.load(
            input_pointer + input_offsets,
            mask=valid_nhw[:, None] & ci_mask[None, :],
            other=0.0,
        )
        weight_block = tl.load(
            weight_pointer + weight_offsets,
            mask=ci_mask[:, None] & co_mask[None, :],
            other=0.0,
        )
        accum += tl.dot(
            input_block,
            weight_block,
            input_precision="tf32x3",
        )

    if has_bias:
        bias = tl.load(
            bias_pointer + co,
            mask=co_in_group < output_channels_per_group,
            other=0.0,
        ).to(tl.float32)
        accum += bias[None, :]

    output_offsets = (n[:, None] * output_channels + co[None, :]) * output_height
    output_offsets = (output_offsets + oh[:, None]) * output_width + ow[:, None]
    output_mask = valid_nhw[:, None] & (
        co_in_group[None, :] < output_channels_per_group
    )
    tl.store(output_pointer + output_offsets, accum, mask=output_mask)


@libentry()
@triton.jit
def _conv_transpose2d_k3_s2_p1_n32_ci64_co32_hw16_fp32_kernel(
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


def _conv_transpose2d_k3_s2_p1_n4_ci256_co128_hw8_lowp(input, weight):
    output = torch.empty((4, 128, 15, 15), device=input.device, dtype=input.dtype)
    grid = (29 * 4,)
    _conv_transpose2d_k3_s2_p1_n4_ci256_co128_hw8_lowp_kernel[grid](
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


def _conv_transpose2d_k4_s2_p1_n8_ci128_co64_hw4_lowp(input, weight):
    output = torch.empty((8, 64, 8, 8), device=input.device, dtype=input.dtype)
    grid = (triton.cdiv(8 * 4 * 4, 64), triton.cdiv(64, 32), 4)
    _conv_transpose2d_k4_s2_p1_n8_ci128_co64_hw4_lowp_kernel[grid](
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


def _conv_transpose2d_k3_s2_p1_n4_ci256_co128_hw8_fp32(input, weight):
    output = torch.empty((4, 128, 15, 15), device=input.device, dtype=input.dtype)
    grid = (16 * 8,)
    _conv_transpose2d_k3_s2_p1_n4_ci256_co128_hw8_fp32_kernel[grid](
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


def _conv_transpose2d_direct_k3_s2_p1_n16_ci32_co64_hw16_fp32(input, weight):
    output = torch.empty((16, 64, 31, 31), device=input.device, dtype=input.dtype)
    grid = (4 * triton.cdiv(16 * 16 * 16, 64), triton.cdiv(64, 16))
    _conv_transpose2d_k3_s2_p1_n16_ci32_co64_hw16_fp32_kernel[grid](
        input,
        weight,
        output,
        BLOCK_NHW=64,
        BLOCK_CI=16,
        BLOCK_CO=16,
        num_warps=8,
        num_stages=2,
    )
    return output


def _conv_transpose2d_direct_k3_s2_p1_n16_ci32_co64_hw32_fp32(input, weight):
    output = torch.empty((16, 64, 63, 63), device=input.device, dtype=input.dtype)
    grid = (4 * triton.cdiv(16 * 32 * 32, 64), triton.cdiv(64, 32))
    _conv_transpose2d_k3_s2_p1_n16_ci32_co64_hw32_fp32_kernel[grid](
        input,
        weight,
        output,
        BLOCK_NHW=64,
        BLOCK_CI=32,
        BLOCK_CO=32,
        num_warps=4,
        num_stages=2,
    )
    return output


def _conv_transpose2d_direct_k3_s2_p1_n32_ci64_co32_hw16_fp32(input, weight):
    output = torch.empty((32, 32, 31, 31), device=input.device, dtype=input.dtype)
    grid = (4 * triton.cdiv(32 * 16 * 16, 128), triton.cdiv(32, 16))
    _conv_transpose2d_k3_s2_p1_n32_ci64_co32_hw16_fp32_kernel[grid](
        input,
        weight,
        output,
        BLOCK_NHW=128,
        BLOCK_CI=16,
        BLOCK_CO=16,
        num_warps=4,
    )
    return output


def _conv_transpose2d_k3_s1_p1_n1_ci64_co64_hw128(input, weight):
    output = torch.empty((1, 64, 128, 128), device=input.device, dtype=input.dtype)
    block_nhw = 128
    block_ci = 16
    block_co = 32
    num_warps = 8

    grid = (triton.cdiv(128 * 128, block_nhw), triton.cdiv(64, block_co))
    _conv_transpose2d_k3_s1_p1_n1_ci64_co64_hw128_kernel[grid](
        input,
        weight,
        output,
        BLOCK_NHW=block_nhw,
        BLOCK_CI=block_ci,
        BLOCK_CO=block_co,
        num_warps=num_warps,
    )
    return output


def _conv_transpose2d_scatter_no_overlap(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_padding_h,
    output_padding_w,
    groups,
):
    batch, input_channels, input_height, input_width = input.shape
    _, output_channels_per_group, weight_height, weight_width = weight.shape
    output_channels = output_channels_per_group * groups
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
    total_elements = output.numel()
    if total_elements == 0:
        return output

    init_block = 1024
    bias_pointer = bias if bias is not None else input
    _conv_transpose2d_scatter_init_kernel[(triton.cdiv(total_elements, init_block),)](
        bias_pointer,
        output,
        total_elements,
        output_channels,
        output_height,
        output_width,
        bias is not None,
        BLOCK_SIZE=init_block,
        num_warps=4,
    )

    input_channels_per_group = input_channels // groups
    if input_channels_per_group <= 16:
        block_ci = 16
    elif input_channels_per_group <= 64:
        block_ci = 64 if input.dtype is not torch.float32 else 32
    else:
        block_ci = 64
    block_co = 16 if output_channels_per_group <= 16 else 32
    block_nhw = 32 if input.dtype is torch.float32 else 64
    if output_channels_per_group >= 64:
        block_nhw = 32

    input_nhw = batch * input_height * input_width
    grid = (
        triton.cdiv(input_nhw, block_nhw),
        triton.cdiv(output_channels_per_group, block_co),
        groups * weight_height * weight_width,
    )
    _conv_transpose2d_scatter_no_overlap_kernel[grid](
        input,
        weight,
        bias_pointer,
        output,
        batch,
        input_channels,
        input_height,
        input_width,
        output_channels,
        output_height,
        output_width,
        weight_height,
        weight_width,
        output_channels_per_group,
        input_channels_per_group,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        bias is not None,
        BLOCK_NHW=block_nhw,
        BLOCK_CI=block_ci,
        BLOCK_CO=block_co,
        num_warps=4,
        num_stages=3,
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

    input_was_unbatched = input.dim() == 3
    if input_was_unbatched:
        input = input.unsqueeze(0)

    if not input.is_contiguous():
        input = input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    output = _conv_transpose2d_4d_dispatch(
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
    if input_was_unbatched:
        return output.squeeze(0)
    return output


def _try_conv_transpose2d_specialized_schedule(
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
    fp32_shape_key = _exact_shape_key(
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
        (torch.float32,),
    )
    if fp32_shape_key == _K3_S2_P1_N32_CI64_CO32_HW16_SHAPE:
        return _conv_transpose2d_direct_k3_s2_p1_n32_ci64_co32_hw16_fp32(input, weight)
    if fp32_shape_key == _K3_S2_P1_N16_CI32_CO64_HW32_SHAPE:
        return _conv_transpose2d_direct_k3_s2_p1_n16_ci32_co64_hw32_fp32(input, weight)
    if fp32_shape_key == _K3_S2_P1_N4_CI256_CO128_HW8_SHAPE:
        return _conv_transpose2d_k3_s2_p1_n4_ci256_co128_hw8_fp32(input, weight)
    if fp32_shape_key == _K3_S2_P1_N16_CI32_CO64_HW16_SHAPE:
        return _conv_transpose2d_direct_k3_s2_p1_n16_ci32_co64_hw16_fp32(input, weight)

    lowp_dtypes = None
    if input.dtype is torch.float16:
        lowp_dtypes = (torch.float16,)
    elif input.dtype is torch.bfloat16:
        lowp_dtypes = _TRITON_DIRECT_LOWP_DTYPES
        if input.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            lowp_dtypes = (torch.float16,)

    if lowp_dtypes is not None:
        lowp_shape_key = _exact_shape_key(
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
            lowp_dtypes,
        )
        if lowp_shape_key == _K3_S2_P1_N4_CI256_CO128_HW8_SHAPE:
            return _conv_transpose2d_k3_s2_p1_n4_ci256_co128_hw8_lowp(input, weight)
        if lowp_shape_key == _K4_S2_P1_N8_CI128_CO64_HW4_SHAPE:
            return _conv_transpose2d_k4_s2_p1_n8_ci128_co64_hw4_lowp(input, weight)
        if lowp_shape_key == _K3_S1_P1_N1_CI64_CO64_HW128_SHAPE:
            return _conv_transpose2d_k3_s1_p1_n1_ci64_co64_hw128(input, weight)

    return None


def _conv_transpose2d_4d_dispatch(
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
    specialized_output = _try_conv_transpose2d_specialized_schedule(
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
    if specialized_output is not None:
        return specialized_output

    direct_tiled_family_shape_key = _direct_tiled_family_shape_key(
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
    if _can_use_direct_tiled_family(
        input, direct_tiled_family_shape_key, output_padding_h
    ):
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

    if _validate_conv_transpose2d_args(
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
        if _can_use_scatter_no_overlap(
            input,
            weight,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            groups,
        ):
            return _conv_transpose2d_scatter_no_overlap(
                input,
                weight,
                bias,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
                output_padding_h,
                output_padding_w,
                groups,
            )
        return _conv_transpose2d_general(
            input,
            weight,
            bias,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            output_padding_h,
            output_padding_w,
            groups,
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


def _select_conv_transpose2d_direct_schedule(input_dtype, shape_key):
    exact_schedule = _DIRECT_TILED_EXACT_SCHEDULES.get((shape_key, input_dtype))
    if exact_schedule is not None:
        return exact_schedule
    exact_schedule = _DIRECT_TILED_EXACT_SCHEDULES.get((shape_key, None))
    if exact_schedule is not None:
        return exact_schedule

    (
        _batch,
        input_channels,
        _input_height,
        _input_width,
        output_channels,
        weight_height,
        weight_width,
        stride_h,
        _padding_h,
    ) = shape_key
    block_nhw, block_ci, block_co, num_warps = _DIRECT_TILED_DEFAULT_SCHEDULE

    if input_dtype is torch.bfloat16:
        if stride_h >= 3:
            block_nhw = 128
            block_ci = 16
            block_co = 16
            num_warps = 8
        elif input_channels >= 128:
            block_nhw = 256
            block_ci = 16
            block_co = 16
            num_warps = 8
        elif weight_height >= 5 or weight_width >= 5:
            block_nhw = 128
            block_ci = 16
        elif input_channels >= 64 and output_channels <= 32:
            block_ci = 64
            if stride_h == 1:
                num_warps = 8
    elif input_dtype is torch.float16:
        if stride_h >= 3:
            block_nhw = 128
            block_ci = 16
            block_co = 16
            num_warps = 8
        elif weight_height >= 5 or weight_width >= 5:
            block_nhw = 128
            block_ci = 16
        elif input_channels >= 64 and output_channels <= 32:
            block_ci = 64
            if stride_h == 1:
                num_warps = 8
    elif input_dtype is torch.float32 and (weight_height >= 5 or weight_width >= 5):
        block_ci = 16
    elif input_channels >= 64 and output_channels <= 32:
        block_ci = 64
        if stride_h == 1:
            num_warps = 8

    return block_nhw, block_ci, block_co, num_warps


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
    block_nhw, block_ci, block_co, num_warps = _select_conv_transpose2d_direct_schedule(
        input.dtype, shape_key
    )

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


def _conv_transpose2d_general(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_padding_h,
    output_padding_w,
    groups,
):
    return _conv_transpose2d_residue(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        output_padding_h,
        output_padding_w,
        groups,
    )


def _conv_transpose2d_residue(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_padding_h,
    output_padding_w,
    groups,
):
    batch, input_channels, input_height, input_width = input.shape
    _, output_channels_per_group, weight_height, weight_width = weight.shape
    output_channels = output_channels_per_group * groups
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
    total_elements = output.numel()
    if total_elements == 0:
        return output

    input_channels_per_group = input_channels // groups
    if (
        input.dtype in _TRITON_DIRECT_LOWP_DTYPES
        and weight_height >= 5
        and weight_width >= 5
        and stride_h == 2
        and stride_w == 2
        and dilation_h == 1
        and dilation_w == 1
        and input_channels_per_group >= 64
        and output_channels_per_group <= 32
    ):
        block_nhw = 256
        block_ci = 16
        block_co = 32
        co_blocks_per_group = triton.cdiv(output_channels_per_group, block_co)
        bias_pointer = bias if bias is not None else input
        for residue_h in range(stride_h):
            compact_height = (output_height + stride_h - 1 - residue_h) // stride_h
            for residue_w in range(stride_w):
                compact_width = (output_width + stride_w - 1 - residue_w) // stride_w
                grid = (
                    triton.cdiv(batch * compact_height * compact_width, block_nhw),
                    groups * co_blocks_per_group,
                )
                _conv_transpose2d_residue_static_kernel[grid](
                    input,
                    weight,
                    bias_pointer,
                    output,
                    batch,
                    input_channels,
                    input_height,
                    input_width,
                    output_channels,
                    output_height,
                    output_width,
                    compact_height,
                    compact_width,
                    weight_height,
                    weight_width,
                    output_channels_per_group,
                    input_channels_per_group,
                    stride_h,
                    stride_w,
                    padding_h,
                    padding_w,
                    dilation_h,
                    dilation_w,
                    bias is not None,
                    residue_h,
                    residue_w,
                    co_blocks_per_group,
                    BLOCK_NHW=block_nhw,
                    BLOCK_CI=block_ci,
                    BLOCK_CO=block_co,
                    num_warps=4,
                    num_stages=2,
                )
        return output

    block_nhw = 64
    block_ci = 32
    block_co = 32
    num_warps = 4
    if input.dtype is torch.float32:
        block_ci = 16
        block_co = 16
    elif input_channels_per_group <= 16:
        block_ci = 16
    if output_channels_per_group <= 16:
        block_co = 16
    if (
        weight_height >= 5
        and weight_width >= 5
        and stride_h == 2
        and stride_w == 2
        and input_channels_per_group >= 64
        and output_channels_per_group <= 32
    ):
        block_nhw = 128
        block_ci = 64 if input.dtype is not torch.float32 else 32
        block_co = 16
        num_warps = 8
    if stride_h * stride_w >= 4 and input.dtype is not torch.float32:
        block_nhw = 128
        num_warps = 8

    compact_height = triton.cdiv(output_height, stride_h)
    compact_width = triton.cdiv(output_width, stride_w)
    max_sub_spatial = batch * compact_height * compact_width
    n_subgrids = stride_h * stride_w
    co_blocks_per_group = triton.cdiv(output_channels_per_group, block_co)
    grid = (
        n_subgrids * triton.cdiv(max_sub_spatial, block_nhw),
        groups * co_blocks_per_group,
    )
    bias_pointer = bias if bias is not None else input
    _conv_transpose2d_residue_kernel[grid](
        input,
        weight,
        bias_pointer,
        output,
        batch,
        input_channels,
        input_height,
        input_width,
        output_channels,
        output_height,
        output_width,
        weight_height,
        weight_width,
        output_channels_per_group,
        input_channels // groups,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        bias is not None,
        n_subgrids,
        co_blocks_per_group,
        BLOCK_NHW=block_nhw,
        BLOCK_CI=block_ci,
        BLOCK_CO=block_co,
        num_warps=num_warps,
    )
    return output
