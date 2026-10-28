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

import logging
import weakref

import torch
import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except (ImportError, AttributeError):
    TensorDescriptor = None

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# MUSA SDK 5.2 / Triton 3.6 stable descriptor/TME experiment.
# Keep this fully optional: if the installed Triton wheel does not expose the
# descriptor API, all shapes continue on the existing packed-pointer kernels.
_HAS_STABLE_TENSOR_DESCRIPTOR = (
    TensorDescriptor is not None
    and hasattr(tl, "load_tensor_descriptor")
    and hasattr(tl, "store_tensor_descriptor")
)

_TME_3X3_BLOCK_M = 128
_TME_3X3_BLOCK_K = 64
_TME_3X3_BLOCK_N = 32

# 3x3 stride-1 VALID mixed-TME layout for the OW=126 benchmark family:
#   [0, 64)   -> M64
#   [64, 96)  -> M32
#   [96, OW)  -> packed fallback (30 columns when OW=126)
_TME_3X3_VALID_BLOCK_M64 = 64
_TME_3X3_VALID_BLOCK_M32 = 32
_TME_3X3_VALID_TME_WIDTH = 96


def conv2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


@libentry()
@triton.jit
def _nchw_to_channels_last_pad_kernel(
    x_ptr,
    y_ptr,
    batch,
    channels,
    in_h,
    in_w,
    padded_h,
    padded_w,
    pad_h,
    pad_w,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fuse NCHW -> channels-last conversion with symmetric zero padding.

    y has logical shape [N, C, padded_h, padded_w] and channels-last strides.
    Each program handles BLOCK_HW padded spatial positions and BLOCK_C
    channels.  Padding values are written as zeros in the same pass as the
    layout conversion, so no separate memset/pad operation is required.
    """

    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)

    hw_lane = tl.arange(0, BLOCK_HW)
    c_lane = tl.arange(0, BLOCK_C)

    flat_hw = pid_hw * BLOCK_HW + hw_lane
    total_hw = batch * padded_h * padded_w
    hw_mask = flat_hw < total_hw

    hw_per_batch = padded_h * padded_w
    n = flat_hw // hw_per_batch
    rem = flat_hw - n * hw_per_batch
    ph = rem // padded_w
    pw = rem - ph * padded_w

    c = pid_c * BLOCK_C + c_lane
    c_mask = c < channels

    ih = ph - pad_h
    iw = pw - pad_w

    inside = hw_mask & (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w)

    x_ptrs = (
        x_ptr
        + n[:, None] * x_stride_n
        + c[None, :] * x_stride_c
        + ih[:, None] * x_stride_h
        + iw[:, None] * x_stride_w
    )

    y_ptrs = (
        y_ptr
        + n[:, None] * y_stride_n
        + c[None, :] * y_stride_c
        + ph[:, None] * y_stride_h
        + pw[:, None] * y_stride_w
    )

    value = tl.load(
        x_ptrs,
        mask=inside[:, None] & c_mask[None, :],
        other=0.0,
    )

    tl.store(
        y_ptrs,
        value,
        mask=hw_mask[:, None] & c_mask[None, :],
    )


@triton.jit
def _conv2d_forward_impl(
    x_ptr,
    w_ptr,
    y_ptr,
    bias_ptr,
    batch,
    in_h,
    in_w,
    out_channels,
    out_h,
    out_w,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    w_stride_o,
    w_stride_i,
    w_stride_h,
    w_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    channels_per_group: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)

    # Flatten (N, OH, OW) into an M dimension.
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    no = m // out_w
    n = no // out_h
    oh = no - n * out_h
    ow = m - no * out_w

    out_per_group = out_channels // groups
    co = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Base address for one group of input channels.
    x_base = x_ptr + (n * x_stride_n + pid_g * channels_per_group * x_stride_c)[:, None]

    # Weight is OIHW.  The logical N dimension here is output channel.
    w_base = w_ptr + ((pid_g * out_per_group + co) * w_stride_o)[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_blocks: tl.constexpr = (channels_per_group + BLOCK_K - 1) // BLOCK_K

    for r in range(kernel_h * kernel_w * k_blocks):
        kb = r % k_blocks
        hw = r // k_blocks
        kh = hw // kernel_w
        kw = hw - kh * kernel_w

        ci = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        ih = oh * stride_h + kh * dilation_h - pad_h
        iw = ow * stride_w + kw * dilation_w - pad_w

        x_tile_ptrs = (
            x_base
            + ci[None, :] * x_stride_c
            + ih[:, None] * x_stride_h
            + iw[:, None] * x_stride_w
        )
        w_tile_ptrs = (
            w_base + ci[:, None] * w_stride_i + kh * w_stride_h + kw * w_stride_w
        )

        x_mask = (
            (n < batch)[:, None]
            & (ci < channels_per_group)[None, :]
            & (ih >= 0)[:, None]
            & (ih < in_h)[:, None]
            & (iw >= 0)[:, None]
            & (iw < in_w)[:, None]
        )
        w_mask = (ci < channels_per_group)[:, None] & (co < out_per_group)[None, :]

        x_tile = tl.load(x_tile_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_tile_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x_tile, w_tile, allow_tf32=False)

    if HAS_BIAS:
        bias_index = pid_g * out_per_group + co
        b = tl.load(
            bias_ptr + bias_index,
            mask=co < out_per_group,
            other=0.0,
        ).to(tl.float32)
        acc += b[None, :]

    y_ptrs = (
        y_ptr
        + n[:, None] * y_stride_n
        + (pid_g * out_per_group + co)[None, :] * y_stride_c
        + oh[:, None] * y_stride_h
        + ow[:, None] * y_stride_w
    )
    y_mask = (n < batch)[:, None] & (co < out_per_group)[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


_FORWARD_AUTOTUNE_KEY = [
    "batch",
    "channels_per_group",
    "in_h",
    "in_w",
    "out_channels",
    "out_h",
    "out_w",
    "kernel_h",
    "kernel_w",
    "stride_h",
    "stride_w",
    "pad_h",
    "pad_w",
    "groups",
]


# Packed-weight forward search space.
#
# Start from the backend's validated conv2d_forward configs, then add only
# conservative BLOCK_CO=64 candidates.  We deliberately avoid M64xCO64 and
# M128xCO32 because both create 4096-element accumulators, which previously
# triggered S5000/MUSA register-allocation failures.
_FORWARD_PACKED_AUTOTUNE_CONFIGS = list(runtime.get_tuned_config("conv2d_forward")) + [
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 32,
            "BLOCK_CO": 64,
            "BLOCK_CI": 16,
        },
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 32,
            "BLOCK_CO": 64,
            "BLOCK_CI": 32,
        },
        num_warps=4,
        num_stages=1,
    ),
]


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("conv2d_forward"),
    key=_FORWARD_AUTOTUNE_KEY,
)
@triton.jit
def conv2d_forward_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    bias_ptr,
    batch,
    in_h,
    in_w,
    out_channels,
    out_h,
    out_w,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    w_stride_o,
    w_stride_i,
    w_stride_h,
    w_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    channels_per_group: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    _conv2d_forward_impl(
        x_ptr,
        w_ptr,
        y_ptr,
        bias_ptr,
        batch,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        x_stride_n,
        x_stride_c,
        x_stride_h,
        x_stride_w,
        w_stride_o,
        w_stride_i,
        w_stride_h,
        w_stride_w,
        y_stride_n,
        y_stride_c,
        y_stride_h,
        y_stride_w,
        channels_per_group,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        True,
        BLOCK_NI_HO_WO,
        BLOCK_CI,
        BLOCK_CO,
    )


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("conv2d_forward"),
    key=_FORWARD_AUTOTUNE_KEY,
)
@triton.jit
def conv2d_forward_no_bias_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    batch,
    in_h,
    in_w,
    out_channels,
    out_h,
    out_w,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    w_stride_o,
    w_stride_i,
    w_stride_h,
    w_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    channels_per_group: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    # y_ptr is passed as a harmless dummy bias pointer.  HAS_BIAS=False means
    # that pointer is never dereferenced after specialization.
    _conv2d_forward_impl(
        x_ptr,
        w_ptr,
        y_ptr,
        y_ptr,
        batch,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        x_stride_n,
        x_stride_c,
        x_stride_h,
        x_stride_w,
        w_stride_o,
        w_stride_i,
        w_stride_h,
        w_stride_w,
        y_stride_n,
        y_stride_c,
        y_stride_h,
        y_stride_w,
        channels_per_group,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        False,
        BLOCK_NI_HO_WO,
        BLOCK_CI,
        BLOCK_CO,
    )


@triton.jit
def _conv2d_forward_packed_impl(
    x_ptr,
    packed_w_ptr,
    y_ptr,
    bias_ptr,
    batch,
    in_h,
    in_w,
    out_channels,
    out_h,
    out_w,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    pw_stride_h,
    pw_stride_w,
    pw_stride_i,
    pw_stride_o,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    channels_per_group: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)

    # Flatten (N, OH, OW) into the implicit-GEMM M dimension.
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    no = m // out_w
    n = no // out_h
    oh = no - n * out_h
    ow = m - no * out_w

    out_per_group = out_channels // groups
    co = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    global_co = pid_g * out_per_group + co

    x_base = x_ptr + (n * x_stride_n + pid_g * channels_per_group * x_stride_c)[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_blocks: tl.constexpr = (channels_per_group + BLOCK_K - 1) // BLOCK_K

    for r in range(kernel_h * kernel_w * k_blocks):
        kb = r % k_blocks
        hw = r // k_blocks
        kh = hw // kernel_w
        kw = hw - kh * kernel_w

        ci = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        ih = oh * stride_h + kh * dilation_h - pad_h
        iw = ow * stride_w + kw * dilation_w - pad_w

        x_tile_ptrs = (
            x_base
            + ci[None, :] * x_stride_c
            + ih[:, None] * x_stride_h
            + iw[:, None] * x_stride_w
        )

        # packed_w layout is [KH, KW, CI, CO_total].  For a fixed (kh, kw),
        # the logical B tile [CI, CO] is row-major with CO contiguous.
        w_tile_ptrs = (
            packed_w_ptr
            + kh * pw_stride_h
            + kw * pw_stride_w
            + ci[:, None] * pw_stride_i
            + global_co[None, :] * pw_stride_o
        )

        x_mask = (
            (n < batch)[:, None]
            & (ci < channels_per_group)[None, :]
            & (ih >= 0)[:, None]
            & (ih < in_h)[:, None]
            & (iw >= 0)[:, None]
            & (iw < in_w)[:, None]
        )
        w_mask = (ci < channels_per_group)[:, None] & (co < out_per_group)[None, :]

        x_tile = tl.load(x_tile_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_tile_ptrs, mask=w_mask, other=0.0)
        # S5000 diagnostic: permit TF32/SQMMA for FP32 packed forward.
        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    if HAS_BIAS:
        b = tl.load(
            bias_ptr + global_co,
            mask=co < out_per_group,
            other=0.0,
        ).to(tl.float32)
        acc += b[None, :]

    y_ptrs = (
        y_ptr
        + n[:, None] * y_stride_n
        + global_co[None, :] * y_stride_c
        + oh[:, None] * y_stride_h
        + ow[:, None] * y_stride_w
    )
    y_mask = (n < batch)[:, None] & (co < out_per_group)[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


@libentry()
@triton.jit
def conv2d_forward_tme_3x3s1_p1_fp16_bf16_kernel(
    x_desc,
    w_desc,
    y_desc,
    out_h,
    out_w,
    padded_h,
    padded_w,
    OW_START: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """Descriptor/TME kernel for large 3x3 stride-1 p1/p2/valid cases.

    Descriptor logical matrices:
      X = padded channels-last input viewed as [N*H*W, 64]
      W = packed HWIO weight viewed as [9*64, 32]
      Y = channels-last output viewed as [N*OH*OW, 32]

    One program computes 128 contiguous output-width positions:
        [128, 64] @ [64, 32] -> [128, 32]

    There are exactly nine descriptor-load/dot steps, one per 3x3 tap.
    """

    pid_row = tl.program_id(0)
    pid_w = tl.program_id(1)

    n = pid_row // out_h
    oh = pid_row - n * out_h
    ow0 = OW_START + pid_w * BLOCK_M

    acc = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )

    # Explicitly padded CL input means effective conv padding is zero.
    # For stride=1, a fixed (kh,kw) maps an OW tile to 32 consecutive rows
    # of the flattened NHWC matrix, which is ideal for a 2D TME load.
    for tap in tl.static_range(0, 9):
        kh = tap // 3
        kw = tap - kh * 3

        x_row = (n * padded_h + (oh + kh)) * padded_w + (ow0 + kw)
        w_row = tap * BLOCK_K

        x_tile = tl.load_tensor_descriptor(
            x_desc,
            [x_row, 0],
        )
        w_tile = tl.load_tensor_descriptor(
            w_desc,
            [w_row, 0],
        )

        acc += tl.dot(
            x_tile,
            w_tile,
        )

    if IS_BF16:
        out_tile = acc.to(tl.bfloat16)
    else:
        out_tile = acc.to(tl.float16)

    y_row = (n * out_h + oh) * out_w + ow0
    tl.store_tensor_descriptor(
        y_desc,
        [y_row, 0],
        out_tile,
    )


@libentry()
@triton.autotune(
    configs=_FORWARD_PACKED_AUTOTUNE_CONFIGS,
    key=_FORWARD_AUTOTUNE_KEY,
)
@triton.jit
def conv2d_forward_packed_kernel(
    x_ptr,
    packed_w_ptr,
    y_ptr,
    bias_ptr,
    batch,
    in_h,
    in_w,
    out_channels,
    out_h,
    out_w,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    pw_stride_h,
    pw_stride_w,
    pw_stride_i,
    pw_stride_o,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    channels_per_group: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    _conv2d_forward_packed_impl(
        x_ptr,
        packed_w_ptr,
        y_ptr,
        bias_ptr,
        batch,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        x_stride_n,
        x_stride_c,
        x_stride_h,
        x_stride_w,
        pw_stride_h,
        pw_stride_w,
        pw_stride_i,
        pw_stride_o,
        y_stride_n,
        y_stride_c,
        y_stride_h,
        y_stride_w,
        channels_per_group,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        True,
        BLOCK_NI_HO_WO,
        BLOCK_CI,
        BLOCK_CO,
    )


@libentry()
@triton.autotune(
    configs=_FORWARD_PACKED_AUTOTUNE_CONFIGS,
    key=_FORWARD_AUTOTUNE_KEY,
)
@triton.jit
def conv2d_forward_no_bias_packed_kernel(
    x_ptr,
    packed_w_ptr,
    y_ptr,
    batch,
    in_h,
    in_w,
    out_channels,
    out_h,
    out_w,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    pw_stride_h,
    pw_stride_w,
    pw_stride_i,
    pw_stride_o,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    channels_per_group: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    # y_ptr is a harmless dummy bias pointer. HAS_BIAS=False specializes the
    # bias load away.
    _conv2d_forward_packed_impl(
        x_ptr,
        packed_w_ptr,
        y_ptr,
        y_ptr,
        batch,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        x_stride_n,
        x_stride_c,
        x_stride_h,
        x_stride_w,
        pw_stride_h,
        pw_stride_w,
        pw_stride_i,
        pw_stride_o,
        y_stride_n,
        y_stride_c,
        y_stride_h,
        y_stride_w,
        channels_per_group,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        False,
        BLOCK_NI_HO_WO,
        BLOCK_CI,
        BLOCK_CO,
    )


@triton.jit
def _conv2d_forward_packed_region_impl(
    x_ptr,
    packed_w_ptr,
    y_ptr,
    bias_ptr,
    batch,
    in_h,
    in_w,
    out_channels,
    out_h,
    out_w,
    region_oh_start,
    region_ow_start,
    region_h,
    region_w,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    pw_stride_h,
    pw_stride_w,
    pw_stride_i,
    pw_stride_o,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    channels_per_group: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CHECK_SPATIAL_BOUNDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute one rectangular output region.

    CHECK_SPATIAL_BOUNDS=False is used only for a host-proven interior region,
    so all (ih, iw) generated for valid m lanes are guaranteed to lie inside
    [0, in_h) x [0, in_w).  This specializes the H/W predicates away.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)

    # Flatten (N, region_OH, region_OW) into M.
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    no = m // region_w
    n = no // region_h
    local_oh = no - n * region_h
    local_ow = m - no * region_w

    oh = region_oh_start + local_oh
    ow = region_ow_start + local_ow

    out_per_group = out_channels // groups
    co = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    global_co = pid_g * out_per_group + co

    x_base = x_ptr + (n * x_stride_n + pid_g * channels_per_group * x_stride_c)[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_blocks: tl.constexpr = (channels_per_group + BLOCK_K - 1) // BLOCK_K

    for r in range(kernel_h * kernel_w * k_blocks):
        kb = r % k_blocks
        hw = r // k_blocks
        kh = hw // kernel_w
        kw = hw - kh * kernel_w

        ci = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        ih = oh * stride_h + kh * dilation_h - pad_h
        iw = ow * stride_w + kw * dilation_w - pad_w

        x_tile_ptrs = (
            x_base
            + ci[None, :] * x_stride_c
            + ih[:, None] * x_stride_h
            + iw[:, None] * x_stride_w
        )

        w_tile_ptrs = (
            packed_w_ptr
            + kh * pw_stride_h
            + kw * pw_stride_w
            + ci[:, None] * pw_stride_i
            + global_co[None, :] * pw_stride_o
        )

        # The m-tail and CI-tail predicates are still required.  Only the
        # expensive per-(kh,kw) spatial bounds checks disappear in the
        # interior specialization.
        base_x_mask = (n < batch)[:, None] & (ci < channels_per_group)[None, :]

        if CHECK_SPATIAL_BOUNDS:
            x_mask = (
                base_x_mask
                & (ih >= 0)[:, None]
                & (ih < in_h)[:, None]
                & (iw >= 0)[:, None]
                & (iw < in_w)[:, None]
            )
        else:
            x_mask = base_x_mask

        w_mask = (ci < channels_per_group)[:, None] & (co < out_per_group)[None, :]

        x_tile = tl.load(x_tile_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_tile_ptrs, mask=w_mask, other=0.0)
        # S5000 diagnostic: permit TF32/SQMMA for FP32 packed forward.
        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    if HAS_BIAS:
        b = tl.load(
            bias_ptr + global_co,
            mask=co < out_per_group,
            other=0.0,
        ).to(tl.float32)
        acc += b[None, :]

    y_ptrs = (
        y_ptr
        + n[:, None] * y_stride_n
        + global_co[None, :] * y_stride_c
        + oh[:, None] * y_stride_h
        + ow[:, None] * y_stride_w
    )
    y_mask = (n < batch)[:, None] & (co < out_per_group)[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


@libentry()
@triton.autotune(
    configs=_FORWARD_PACKED_AUTOTUNE_CONFIGS,
    key=_FORWARD_AUTOTUNE_KEY,
)
@triton.jit
def conv2d_forward_packed_interior_kernel(
    x_ptr,
    packed_w_ptr,
    y_ptr,
    bias_ptr,
    batch,
    in_h,
    in_w,
    out_channels,
    out_h,
    out_w,
    region_oh_start,
    region_ow_start,
    region_h,
    region_w,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    pw_stride_h,
    pw_stride_w,
    pw_stride_i,
    pw_stride_o,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    channels_per_group: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """Row-wise packed-weight interior kernel.

    Grid:
      axis 0: batch * interior_h        -> one (N, OH) row/program
      axis 1: ceil(interior_w / BLOCK) -> contiguous OW tile
      axis 2: groups * CO blocks        -> group/output-channel tile

    Unlike the old flattened interior mapping, output lanes never execute
    ``m // region_w`` or ``m % region_w``.  Because the host has proven this
    rectangle is fully interior, valid OW lanes also need no input H/W bounds
    predicates.
    """

    pid_row = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_gc = tl.program_id(2)

    # Decode batch/row once per program, not once per output lane.
    n = pid_row // region_h
    local_oh = pid_row - n * region_h
    oh = region_oh_start + local_oh

    # Reuse the existing forward autotune M tile as a contiguous OW tile.
    ow_lane = tl.arange(0, BLOCK_NI_HO_WO)
    local_ow = pid_w * BLOCK_NI_HO_WO + ow_lane
    ow = region_ow_start + local_ow
    ow_mask = local_ow < region_w

    out_per_group = out_channels // groups
    co_blocks = (out_per_group + BLOCK_CO - 1) // BLOCK_CO
    pid_g = pid_gc // co_blocks
    pid_co = pid_gc - pid_g * co_blocks

    co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    global_co = pid_g * out_per_group + co

    x_base = x_ptr + n * x_stride_n + pid_g * channels_per_group * x_stride_c

    acc = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)
    k_blocks: tl.constexpr = (channels_per_group + BLOCK_CI - 1) // BLOCK_CI

    for r in range(kernel_h * kernel_w * k_blocks):
        kb = r % k_blocks
        hw = r // k_blocks
        kh = hw // kernel_w
        kw = hw - kh * kernel_w

        ci = kb * BLOCK_CI + tl.arange(0, BLOCK_CI)
        ih = oh * stride_h + kh * dilation_h - pad_h
        iw = ow * stride_w + kw * dilation_w - pad_w

        x_tile_ptrs = (
            x_base
            + ci[None, :] * x_stride_c
            + ih * x_stride_h
            + iw[:, None] * x_stride_w
        )

        w_tile_ptrs = (
            packed_w_ptr
            + kh * pw_stride_h
            + kw * pw_stride_w
            + ci[:, None] * pw_stride_i
            + global_co[None, :] * pw_stride_o
        )

        # Spatial H/W checks are intentionally absent on the interior path.
        # Only the last OW tile and CI tail need masking.
        x_mask = ow_mask[:, None] & (ci < channels_per_group)[None, :]
        w_mask = (ci < channels_per_group)[:, None] & (co < out_per_group)[None, :]

        x_tile = tl.load(x_tile_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_tile_ptrs, mask=w_mask, other=0.0)
        # S5000 diagnostic: permit TF32/SQMMA for FP32 packed forward.
        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    if HAS_BIAS:
        b = tl.load(
            bias_ptr + global_co,
            mask=co < out_per_group,
            other=0.0,
        ).to(tl.float32)
        acc += b[None, :]

    y_ptrs = (
        y_ptr
        + n * y_stride_n
        + global_co[None, :] * y_stride_c
        + oh * y_stride_h
        + ow[:, None] * y_stride_w
    )
    y_mask = ow_mask[:, None] & (co < out_per_group)[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


@libentry()
@triton.autotune(
    configs=_FORWARD_PACKED_AUTOTUNE_CONFIGS,
    key=_FORWARD_AUTOTUNE_KEY,
)
@triton.jit
def conv2d_forward_packed_boundary_kernel(
    x_ptr,
    packed_w_ptr,
    y_ptr,
    bias_ptr,
    batch,
    in_h,
    in_w,
    out_channels,
    out_h,
    out_w,
    region_oh_start,
    region_ow_start,
    region_h,
    region_w,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    pw_stride_h,
    pw_stride_w,
    pw_stride_i,
    pw_stride_o,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    channels_per_group: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    _conv2d_forward_packed_region_impl(
        x_ptr,
        packed_w_ptr,
        y_ptr,
        bias_ptr,
        batch,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        region_oh_start,
        region_ow_start,
        region_h,
        region_w,
        x_stride_n,
        x_stride_c,
        x_stride_h,
        x_stride_w,
        pw_stride_h,
        pw_stride_w,
        pw_stride_i,
        pw_stride_o,
        y_stride_n,
        y_stride_c,
        y_stride_h,
        y_stride_w,
        channels_per_group,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        HAS_BIAS,
        True,
        BLOCK_NI_HO_WO,
        BLOCK_CI,
        BLOCK_CO,
    )


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("conv2d_backward_weight"),
    key=[
        "batch",
        "in_h",
        "in_w",
        "kernel_h",
        "kernel_w",
        "channels_per_group",
        "stride_h",
        "stride_w",
        "out_h",
        "out_w",
        "out_per_group",
        "pad_h",
        "pad_w",
    ],
)
@triton.jit
def conv2d_backward_kernel_weight(
    x_ptr,
    dy_ptr,
    dw_ptr,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    dw_stride_o,
    dw_stride_i,
    dw_stride_h,
    dw_stride_w,
    dy_stride_n,
    dy_stride_c,
    dy_stride_h,
    dy_stride_w,
    in_h,
    in_w,
    kernel_h,
    kernel_w,
    channels_per_group,
    batch,
    stride_h,
    stride_w,
    out_h,
    out_w,
    out_per_group,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    BLOCK_NO: tl.constexpr,
    BLOCK_CI_HK_WK: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_o = tl.program_id(2)

    flat_k = pid_k * BLOCK_CI_HK_WK + tl.arange(0, BLOCK_CI_HK_WK)
    ci_hw = flat_k // kernel_w
    ci = ci_hw // kernel_h
    kh = ci_hw - ci * kernel_h
    kw = flat_k - ci_hw * kernel_w

    co = pid_o * BLOCK_CO + tl.arange(0, BLOCK_CO)

    dy_group_base = dy_ptr + ((pid_g * out_per_group + co) * dy_stride_c)[None, :]

    dw_group_ptrs = (
        dw_ptr
        + (pid_g * out_per_group + co)[None, :] * dw_stride_o
        + ci[:, None] * dw_stride_i
        + kh[:, None] * dw_stride_h
        + kw[:, None] * dw_stride_w
    )

    x_group_base = (
        x_ptr + (pid_g * channels_per_group * x_stride_c + ci * x_stride_c)[:, None]
    )

    acc = tl.zeros((BLOCK_CI_HK_WK, BLOCK_CO), dtype=tl.float32)

    for oh_idx in range(0, out_h):
        for ow_idx in range(0, out_w):
            for n0 in range(0, batch, BLOCK_NO):
                ns = n0 + tl.arange(0, BLOCK_NO)

                dy_ptrs = (
                    dy_group_base
                    + ns[:, None] * dy_stride_n
                    + oh_idx * dy_stride_h
                    + ow_idx * dy_stride_w
                )
                dy_mask = (ns < batch)[:, None] & (co < out_per_group)[None, :]
                dy_tile = tl.load(dy_ptrs, mask=dy_mask, other=0.0)

                ih = kh * dilation_h - pad_h + oh_idx * stride_h
                iw = kw * dilation_w - pad_w + ow_idx * stride_w
                x_ptrs = (
                    x_group_base
                    + ns[None, :] * x_stride_n
                    + ih[:, None] * x_stride_h
                    + iw[:, None] * x_stride_w
                )
                x_mask = (
                    (ns < batch)[None, :]
                    & (ci < channels_per_group)[:, None]
                    & (kh < kernel_h)[:, None]
                    & (kw < kernel_w)[:, None]
                    & (ih >= 0)[:, None]
                    & (ih < in_h)[:, None]
                    & (iw >= 0)[:, None]
                    & (iw < in_w)[:, None]
                )
                x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
                acc += tl.dot(x_tile, dy_tile, allow_tf32=False)

    store_mask = (
        (ci < channels_per_group)[:, None]
        & (kh < kernel_h)[:, None]
        & (kw < kernel_w)[:, None]
        & (co < out_per_group)[None, :]
    )
    tl.store(dw_group_ptrs, acc, mask=store_mask)


# Pack only when each weight value is reused by many output positions.  This
# excludes the tiny cases that were already near launch-overhead limited in the
# benchmark while covering the large 128x128 / 210x210 cases.
_WEIGHT_PREPACK_MIN_M = 32768

# Split into interior + boundary rectangles only when most output pixels are
# interior.  This keeps tiny or boundary-heavy convolutions on the single
# packed kernel and avoids paying several extra launches for little benefit.
_INTERIOR_FASTPATH_MIN_FRACTION = 0.80

# Shape-aware NCHW -> channels_last conversion.
#
# The measured S5000 case
#   input=[64,32,210,210], weight=[64,32,5,5], stride=2, padding=1
# improved even after including the explicit conversion cost.  Keep the first
# production experiment conservative and only convert similarly large dense
# 5x5+ convolutions.  This intentionally excludes the 128x128 3x3 cases and
# all group/small convolutions until they are measured separately.
_CL_CONVERT_MIN_M = 262144
_CL_CONVERT_MIN_CI = 32
_CL_CONVERT_MIN_KERNEL_AREA = 25

# Fused NCHW -> padded-CL preprocessing tile.
# No dot-product accumulator is live here, so a 32x64 load/store tile is much
# less register-sensitive than the convolution kernels themselves.
_FUSED_PAD_CL_BLOCK_HW = 32
_FUSED_PAD_CL_BLOCK_C = 64

# Large power-of-two 3x3 fast path.  The benchmark case
#   [32,64,128,128] * [32,64,3,3], stride=1, padding=1
# historically ran faster with the original single packed kernel than with the
# later interior/boundary split.  Keep this dispatch conservative.
_POWER2_3X3_MIN_M = 262144
_3X3_FUSED_CL_MIN_M = 262144

# id(weight) -> (weakref(weight), version, shape, stride, packed_weight)
_WEIGHT_PREPACK_CACHE = {}


def _should_use_weight_prepack(batch: int, out_h: int, out_w: int) -> bool:
    return batch * out_h * out_w >= _WEIGHT_PREPACK_MIN_M


def _should_convert_input_to_channels_last(
    input: torch.Tensor,
    batch: int,
    out_h: int,
    out_w: int,
    channels_per_group: int,
    kernel_h: int,
    kernel_w: int,
    groups: int,
) -> bool:
    """Return True when paying one NCHW->CL copy is likely to amortize.

    This is intentionally shape-aware rather than global.  The first heuristic
    targets the large 5x5 dense family for which conversion+conv was measured
    faster than the NCHW kernel on S5000.

    Already-CL inputs are never copied.
    """

    if input.is_contiguous(memory_format=torch.channels_last):
        return False

    # Do not disturb non-standard strided views in this first experiment.
    if not input.is_contiguous():
        return False

    if groups != 1:
        return False

    if channels_per_group < _CL_CONVERT_MIN_CI:
        return False

    if kernel_h * kernel_w < _CL_CONVERT_MIN_KERNEL_AREA:
        return False

    if batch * out_h * out_w < _CL_CONVERT_MIN_M:
        return False

    return True


def _nchw_to_padded_channels_last(
    input: torch.Tensor,
    pad_h: int,
    pad_w: int,
) -> torch.Tensor:
    """Return channels-last [N,C,H+2P,W+2P], fusing copy + zero padding."""

    batch, channels, in_h, in_w = input.shape
    padded_h = in_h + 2 * pad_h
    padded_w = in_w + 2 * pad_w

    output = torch.empty(
        (batch, channels, padded_h, padded_w),
        device=input.device,
        dtype=input.dtype,
        memory_format=torch.channels_last,
    )

    grid = (
        triton.cdiv(
            batch * padded_h * padded_w,
            _FUSED_PAD_CL_BLOCK_HW,
        ),
        triton.cdiv(
            channels,
            _FUSED_PAD_CL_BLOCK_C,
        ),
    )

    _nchw_to_channels_last_pad_kernel[grid](
        input,
        output,
        batch,
        channels,
        in_h,
        in_w,
        padded_h,
        padded_w,
        pad_h,
        pad_w,
        *input.stride(),
        *output.stride(),
        BLOCK_HW=_FUSED_PAD_CL_BLOCK_HW,
        BLOCK_C=_FUSED_PAD_CL_BLOCK_C,
        num_warps=4,
        num_stages=1,
    )

    return output


def _compute_interior_output_bounds(
    in_h: int,
    in_w: int,
    out_h: int,
    out_w: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
):
    """Return [oh0, oh1) x [ow0, ow1) whose full receptive fields are valid."""

    # Need:
    #   oh*stride_h - pad_h >= 0
    #   oh*stride_h - pad_h + (kernel_h-1)*dilation_h <= in_h-1
    oh0 = max(0, (pad_h + stride_h - 1) // stride_h)
    ow0 = max(0, (pad_w + stride_w - 1) // stride_w)

    last_oh_num = in_h - 1 + pad_h - (kernel_h - 1) * dilation_h
    last_ow_num = in_w - 1 + pad_w - (kernel_w - 1) * dilation_w

    if last_oh_num < 0:
        oh1 = 0
    else:
        oh1 = min(out_h, last_oh_num // stride_h + 1)

    if last_ow_num < 0:
        ow1 = 0
    else:
        ow1 = min(out_w, last_ow_num // stride_w + 1)

    oh0 = min(max(oh0, 0), out_h)
    ow0 = min(max(ow0, 0), out_w)
    oh1 = min(max(oh1, oh0), out_h)
    ow1 = min(max(ow1, ow0), out_w)

    return oh0, oh1, ow0, ow1


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _should_use_tme_3x3s1(
    input: torch.Tensor,
    batch: int,
    out_channels: int,
    out_h: int,
    out_w: int,
    channels_per_group: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
) -> bool:
    """Stable TensorDescriptor/TME route for large 3x3 stride-1 FP16/BF16.

    Supported padding:
      valid/p0: OW=126 -> M64 + M32 TME + 30-column packed tail.
      p1:       OW=128 -> one full M128 TME tile, no tail.
      p2:       OW=130 -> one full M128 TME tile + 2-column packed tail.

    All other shapes stay on the proven packed implementation.
    """

    if not _HAS_STABLE_TENSOR_DESCRIPTOR:
        return False

    if input.dtype not in (torch.float16, torch.bfloat16):
        return False

    if not input.is_contiguous():
        return False

    if groups != 1:
        return False

    if channels_per_group != _TME_3X3_BLOCK_K:
        return False

    if out_channels != _TME_3X3_BLOCK_N:
        return False

    if kernel_h != 3 or kernel_w != 3:
        return False

    if stride_h != 1 or stride_w != 1:
        return False

    if pad_h not in (0, 1, 2) or pad_w not in (0, 1, 2):
        return False

    if pad_h != pad_w:
        return False

    if dilation_h != 1 or dilation_w != 1:
        return False

    if pad_h == 0:
        # VALID benchmark family: use one M64 tile + one M32 tile.  Keep this
        # mixed strategy below M128; larger widths are better served by the
        # existing M128 route or the generic packed path.
        if out_w < _TME_3X3_VALID_TME_WIDTH or out_w >= _TME_3X3_BLOCK_M:
            return False
    else:
        # p1/p2 need at least one complete M128 tile.  Any right-edge
        # remainder is handled by the packed-region fallback.
        if out_w < _TME_3X3_BLOCK_M:
            return False

    # Keep the experiment on the same large family as the power-of-two path.
    if batch * out_h * out_w < _POWER2_3X3_MIN_M:
        return False

    return True


def _should_use_power2_3x3_fastpath(
    input: torch.Tensor,
    batch: int,
    out_h: int,
    out_w: int,
    channels_per_group: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
) -> bool:
    """Use one full-output packed kernel for large power-of-two 3x3 cases."""

    if groups != 1:
        return False

    if kernel_h != 3 or kernel_w != 3:
        return False

    if stride_h != 1 or stride_w != 1:
        return False

    if dilation_h != 1 or dilation_w != 1:
        return False

    if pad_h != 1 or pad_w != 1:
        return False

    if channels_per_group < 32:
        return False

    if batch * out_h * out_w < _POWER2_3X3_MIN_M:
        return False

    if not _is_power_of_two(out_h) or not _is_power_of_two(out_w):
        return False

    # The benchmark family is standard contiguous NCHW.  Already-CL tensors
    # can still use the generic CL-oriented path, avoiding an unnecessary
    # policy interaction in this first experiment.
    if not input.is_contiguous():
        return False

    return True


def _should_use_extended_3x3_fused_cl_fastpath(
    input: torch.Tensor,
    batch: int,
    out_h: int,
    out_w: int,
    channels_per_group: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
) -> bool:
    """FP16/BF16 full-output CL path for 3x3 valid and padding=2."""

    if input.dtype not in (torch.float16, torch.bfloat16):
        return False

    if groups != 1:
        return False

    if kernel_h != 3 or kernel_w != 3:
        return False

    if stride_h != 1 or stride_w != 1:
        return False

    if dilation_h != 1 or dilation_w != 1:
        return False

    if channels_per_group < 32:
        return False

    if batch * out_h * out_w < _3X3_FUSED_CL_MIN_M:
        return False

    # This experiment intentionally targets exactly the requested benchmark
    # families.  padding=1 remains on the existing power-of-two route.
    if (pad_h, pad_w) not in ((0, 0), (2, 2)):
        return False

    # Keep non-standard views / already-CL inputs on their existing route.
    if not input.is_contiguous():
        return False

    return True


def _should_use_interior_fastpath(
    batch: int,
    out_h: int,
    out_w: int,
    bounds,
) -> bool:
    oh0, oh1, ow0, ow1 = bounds
    interior_h = oh1 - oh0
    interior_w = ow1 - ow0
    interior_area = interior_h * interior_w
    total_area = out_h * out_w

    if interior_area <= 0 or total_area <= 0:
        return False

    interior_fraction = interior_area / total_area

    # Also require enough interior M work to amortize the extra region launches.
    return (
        interior_fraction >= _INTERIOR_FASTPATH_MIN_FRACTION
        and batch * interior_area >= _WEIGHT_PREPACK_MIN_M
    )


def _get_prepacked_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return cached contiguous [KH, KW, CI, CO] weight.

    Tensor._version changes after in-place updates, which is exactly what common
    optimizers do to parameters.  A version mismatch therefore invalidates the
    cached packed copy automatically.
    """

    key = id(weight)
    version = int(weight._version)
    shape = tuple(weight.shape)
    stride = tuple(weight.stride())

    entry = _WEIGHT_PREPACK_CACHE.get(key)
    if entry is not None:
        weight_ref, cached_version, cached_shape, cached_stride, packed = entry
        if (
            weight_ref() is weight
            and cached_version == version
            and cached_shape == shape
            and cached_stride == stride
        ):
            return packed

    # OIHW -> HWIO == [KH, KW, CI, CO].
    packed = weight.permute(2, 3, 1, 0).contiguous()

    def _remove(dead_ref, cache_key=key):
        current = _WEIGHT_PREPACK_CACHE.get(cache_key)
        if current is not None and current[0] is dead_ref:
            _WEIGHT_PREPACK_CACHE.pop(cache_key, None)

    weight_ref = weakref.ref(weight, _remove)
    _WEIGHT_PREPACK_CACHE[key] = (weight_ref, version, shape, stride, packed)
    return packed


def conv2d_would_convert_input_to_channels_last(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
) -> bool:
    """Debug helper: report whether the current shape-aware policy copies input."""

    if isinstance(stride, (tuple, list)):
        stride_h, stride_w = stride
    else:
        stride_h = stride_w = stride

    if isinstance(padding, (tuple, list)):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = padding

    if isinstance(dilation, (tuple, list)):
        dilation_h, dilation_w = dilation
    else:
        dilation_h = dilation_w = dilation

    batch, _, in_h, in_w = input.shape
    _, channels_per_group, kernel_h, kernel_w = weight.shape
    out_h = conv2d_output_size(in_h, kernel_h, stride_h, pad_h, dilation_h)
    out_w = conv2d_output_size(in_w, kernel_w, stride_w, pad_w, dilation_w)

    return _should_use_weight_prepack(
        batch, out_h, out_w
    ) and _should_convert_input_to_channels_last(
        input,
        batch,
        out_h,
        out_w,
        channels_per_group,
        kernel_h,
        kernel_w,
        groups,
    )


def conv2d_would_use_fused_nchw_cl_padding(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
) -> bool:
    """Debug helper for the fused NCHW->CL+padding forward preprocess."""

    if isinstance(stride, (tuple, list)):
        stride_h, stride_w = stride
    else:
        stride_h = stride_w = stride

    if isinstance(padding, (tuple, list)):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = padding

    if isinstance(dilation, (tuple, list)):
        dilation_h, dilation_w = dilation
    else:
        dilation_h = dilation_w = dilation

    batch, _, in_h, in_w = input.shape
    _, channels_per_group, kernel_h, kernel_w = weight.shape

    out_h = conv2d_output_size(
        in_h,
        kernel_h,
        stride_h,
        pad_h,
        dilation_h,
    )
    out_w = conv2d_output_size(
        in_w,
        kernel_w,
        stride_w,
        pad_w,
        dilation_w,
    )

    return (
        (pad_h > 0 or pad_w > 0)
        and _should_use_weight_prepack(batch, out_h, out_w)
        and _should_convert_input_to_channels_last(
            input,
            batch,
            out_h,
            out_w,
            channels_per_group,
            kernel_h,
            kernel_w,
            groups,
        )
    )


def conv2d_would_use_power2_3x3_fastpath(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
) -> bool:
    """Debug helper for the full-output power-of-two 3x3 packed path."""

    if isinstance(stride, (tuple, list)):
        stride_h, stride_w = stride
    else:
        stride_h = stride_w = stride

    if isinstance(padding, (tuple, list)):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = padding

    if isinstance(dilation, (tuple, list)):
        dilation_h, dilation_w = dilation
    else:
        dilation_h = dilation_w = dilation

    batch, _, in_h, in_w = input.shape
    _, channels_per_group, kernel_h, kernel_w = weight.shape

    out_h = conv2d_output_size(in_h, kernel_h, stride_h, pad_h, dilation_h)
    out_w = conv2d_output_size(in_w, kernel_w, stride_w, pad_w, dilation_w)

    return _should_use_weight_prepack(
        batch, out_h, out_w
    ) and _should_use_power2_3x3_fastpath(
        input,
        batch,
        out_h,
        out_w,
        channels_per_group,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
    )


def conv2d_would_use_power2_3x3_fused_cl_padding(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
) -> bool:
    """Debug helper for dtype-aware power2-3x3 fused CL+padding."""

    if input.dtype not in (torch.float16, torch.bfloat16):
        return False

    if not conv2d_would_use_power2_3x3_fastpath(
        input,
        weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    ):
        return False

    if isinstance(padding, (tuple, list)):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = padding

    return pad_h > 0 or pad_w > 0


def conv2d_would_use_3x3_fused_cl_full_output(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
) -> bool:
    """Debug helper for fp16/bf16 3x3 p1/p2/valid CL full-output paths."""

    if input.dtype not in (torch.float16, torch.bfloat16):
        return False

    if isinstance(stride, (tuple, list)):
        stride_h, stride_w = stride
    else:
        stride_h = stride_w = stride

    if isinstance(padding, str):
        if padding.lower() == "valid":
            pad_h = pad_w = 0
        else:
            return False
    elif isinstance(padding, (tuple, list)):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = padding

    if isinstance(dilation, (tuple, list)):
        dilation_h, dilation_w = dilation
    else:
        dilation_h = dilation_w = dilation

    batch, _, in_h, in_w = input.shape
    _, channels_per_group, kernel_h, kernel_w = weight.shape

    out_h = conv2d_output_size(in_h, kernel_h, stride_h, pad_h, dilation_h)
    out_w = conv2d_output_size(in_w, kernel_w, stride_w, pad_w, dilation_w)

    if not _should_use_weight_prepack(batch, out_h, out_w):
        return False

    if _should_use_power2_3x3_fastpath(
        input,
        batch,
        out_h,
        out_w,
        channels_per_group,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
    ):
        return True

    return _should_use_extended_3x3_fused_cl_fastpath(
        input,
        batch,
        out_h,
        out_w,
        channels_per_group,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
    )


def conv2d_has_block_co64_candidate(
    weight: torch.Tensor,
    groups: int = 1,
) -> bool:
    """Report whether the packed forward's CO64 tile can cover a full group."""

    out_channels = weight.shape[0]
    if groups <= 0 or out_channels % groups != 0:
        return False
    return (out_channels // groups) >= 64


def conv2d_would_use_tme_3x3s1(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
) -> bool:
    """Debug helper for the stable TensorDescriptor/TME 3x3 experiment."""

    if isinstance(stride, (tuple, list)):
        stride_h, stride_w = stride
    else:
        stride_h = stride_w = stride

    if isinstance(padding, (tuple, list)):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = padding

    if isinstance(dilation, (tuple, list)):
        dilation_h, dilation_w = dilation
    else:
        dilation_h = dilation_w = dilation

    batch, _, in_h, in_w = input.shape
    out_channels, channels_per_group, kernel_h, kernel_w = weight.shape

    out_h = conv2d_output_size(in_h, kernel_h, stride_h, pad_h, dilation_h)
    out_w = conv2d_output_size(in_w, kernel_w, stride_w, pad_w, dilation_w)

    return _should_use_weight_prepack(batch, out_h, out_w) and _should_use_tme_3x3s1(
        input,
        batch,
        out_channels,
        out_h,
        out_w,
        channels_per_group,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
    )


def clear_conv2d_weight_prepack_cache():
    """Drop all cached packed weights for this module."""
    _WEIGHT_PREPACK_CACHE.clear()


class Conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        logger.debug("GEMS_MTHREADS CONV2D")

        assert input.ndim == 4, f"Input must be 4D, got {tuple(input.shape)}"
        assert weight.ndim == 4, f"Weight must be 4D, got {tuple(weight.shape)}"
        assert bias is None or bias.ndim == 1, "Bias must be 1D"

        if isinstance(stride, (tuple, list)):
            stride_h, stride_w = stride
        else:
            stride_h = stride_w = stride

        if isinstance(padding, (tuple, list)):
            pad_h, pad_w = padding
        else:
            pad_h = pad_w = padding

        if isinstance(dilation, (tuple, list)):
            dilation_h, dilation_w = dilation
        else:
            dilation_h = dilation_w = dilation

        batch, in_channels, in_h, in_w = input.shape
        out_channels, channels_per_group, kernel_h, kernel_w = weight.shape

        assert groups > 0
        assert in_channels == groups * channels_per_group
        assert out_channels % groups == 0
        assert bias is None or bias.shape[0] == out_channels

        out_h = conv2d_output_size(in_h, kernel_h, stride_h, pad_h, dilation_h)
        out_w = conv2d_output_size(in_w, kernel_w, stride_w, pad_w, dilation_w)

        # Keep the user-visible/original tensor for backward.  The forward
        # kernel may consume a temporary channels-last copy for selected large
        # shapes, but backward should continue to see the original input.
        original_input = input
        forward_input = input

        # Effective geometry seen by the forward Triton kernels.  A fused
        # NCHW->CL+padding preprocess changes these to padded H/W with pad=0,
        # while backward still records and uses the original geometry/padding.
        forward_in_h = in_h
        forward_in_w = in_w
        forward_pad_h = pad_h
        forward_pad_w = pad_w
        used_fused_pad_cl = False

        use_packed_weight = _should_use_weight_prepack(batch, out_h, out_w)

        # 3x3 full-output routing.
        #
        # p1/power2 single-packed remains available for all dtypes.
        # Fused-CL preprocessing is enabled only for fp16/bf16 because the
        # measured FP32 p1 case regressed when forced through the CL transform.
        use_tme_3x3s1 = use_packed_weight and _should_use_tme_3x3s1(
            original_input,
            batch,
            out_channels,
            out_h,
            out_w,
            channels_per_group,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            groups,
        )

        use_power2_3x3_fastpath = use_packed_weight and _should_use_power2_3x3_fastpath(
            original_input,
            batch,
            out_h,
            out_w,
            channels_per_group,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            groups,
        )

        use_extended_3x3_fused_cl_fastpath = (
            use_packed_weight
            and _should_use_extended_3x3_fused_cl_fastpath(
                original_input,
                batch,
                out_h,
                out_w,
                channels_per_group,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
                groups,
            )
        )

        use_power2_3x3_fused_cl = use_power2_3x3_fastpath and input.dtype in (
            torch.float16,
            torch.bfloat16,
        )

        use_3x3_full_output_fastpath = (
            use_power2_3x3_fastpath or use_extended_3x3_fused_cl_fastpath
        )

        convert_input_to_cl = (
            use_tme_3x3s1
            or use_power2_3x3_fused_cl
            or use_extended_3x3_fused_cl_fastpath
            or (
                use_packed_weight
                and _should_convert_input_to_channels_last(
                    input,
                    batch,
                    out_h,
                    out_w,
                    channels_per_group,
                    kernel_h,
                    kernel_w,
                    groups,
                )
            )
        )

        if convert_input_to_cl:
            if pad_h > 0 or pad_w > 0:
                # One preprocessing launch performs both operations:
                #   NCHW -> channels_last
                #   explicit symmetric zero padding
                #
                # For the large 5x5 path this makes the whole output a
                # no-spatial-mask interior region.
                #
                # For fp16/bf16 3x3 p1/p2 this materializes padding into the
                # CL tensor, so the later TME convolution runs with effective
                # padding=0 over the full output.
                forward_input = _nchw_to_padded_channels_last(
                    input,
                    pad_h,
                    pad_w,
                )
                forward_in_h = in_h + 2 * pad_h
                forward_in_w = in_w + 2 * pad_w
                forward_pad_h = 0
                forward_pad_w = 0
                used_fused_pad_cl = True
            else:
                # padding=0 (including 3x3 "valid"): only layout conversion
                # is needed; all output positions are already spatially valid.
                forward_input = input.contiguous(memory_format=torch.channels_last)

        # The forward implicit-GEMM accumulator is [M, CO].  channels_last
        # makes output C/CO stride 1, which substantially improves stores on
        # S5000.  A channels-last input likewise makes input CI stride 1.
        output = torch.empty(
            (batch, out_channels, out_h, out_w),
            device=input.device,
            dtype=input.dtype,
            memory_format=torch.channels_last,
        )

        grid = lambda meta: (
            triton.cdiv(batch * out_h * out_w, meta["BLOCK_NI_HO_WO"]),
            triton.cdiv(out_channels // groups, meta["BLOCK_CO"]),
            groups,
        )

        if use_packed_weight:
            packed_weight = _get_prepacked_weight(weight)

            if use_tme_3x3s1 and bias is None:
                # Stable 3x3 stride-1 TensorDescriptor/TME family.
                #
                # p1:    OW=128 -> M128, no tail
                # p2:    OW=130 -> M128 + 2-column packed tail
                # valid: OW=126 -> M64 + M32 + 30-column packed tail
                #
                # For p1/p2 the fused preprocess materializes explicit padding
                # into a channels-last tensor.  For valid/p0 only the layout
                # conversion is needed.  In both cases the forward convolution
                # kernel sees effective padding=0.
                assert forward_pad_h == 0 and forward_pad_w == 0
                if pad_h > 0:
                    assert used_fused_pad_cl

                x_matrix = forward_input.permute(0, 2, 3, 1).reshape(
                    -1, channels_per_group
                )
                w_matrix = packed_weight.reshape(
                    kernel_h * kernel_w * channels_per_group,
                    out_channels,
                )
                y_matrix = output.permute(0, 2, 3, 1).reshape(-1, out_channels)

                # Weight tile shape is identical for M128/M64/M32.
                w_desc = TensorDescriptor.from_tensor(
                    w_matrix,
                    [_TME_3X3_BLOCK_K, _TME_3X3_BLOCK_N],
                )

                if pad_h == 0:
                    # VALID / OW=126:
                    #   [0,64)  -> one M64 TME tile
                    #   [64,96) -> one M32 TME tile
                    #   [96,126)-> packed fallback
                    x_desc_m64 = TensorDescriptor.from_tensor(
                        x_matrix,
                        [
                            _TME_3X3_VALID_BLOCK_M64,
                            _TME_3X3_BLOCK_K,
                        ],
                    )
                    y_desc_m64 = TensorDescriptor.from_tensor(
                        y_matrix,
                        [
                            _TME_3X3_VALID_BLOCK_M64,
                            _TME_3X3_BLOCK_N,
                        ],
                    )

                    valid_grid_m64 = (batch * out_h, 1)
                    conv2d_forward_tme_3x3s1_p1_fp16_bf16_kernel[valid_grid_m64](
                        x_desc_m64,
                        w_desc,
                        y_desc_m64,
                        out_h,
                        out_w,
                        forward_in_h,
                        forward_in_w,
                        OW_START=0,
                        BLOCK_M=_TME_3X3_VALID_BLOCK_M64,
                        BLOCK_K=_TME_3X3_BLOCK_K,
                        BLOCK_N=_TME_3X3_BLOCK_N,
                        IS_BF16=input.dtype == torch.bfloat16,
                        num_warps=4,
                        num_stages=1,
                    )

                    x_desc_m32 = TensorDescriptor.from_tensor(
                        x_matrix,
                        [
                            _TME_3X3_VALID_BLOCK_M32,
                            _TME_3X3_BLOCK_K,
                        ],
                    )
                    y_desc_m32 = TensorDescriptor.from_tensor(
                        y_matrix,
                        [
                            _TME_3X3_VALID_BLOCK_M32,
                            _TME_3X3_BLOCK_N,
                        ],
                    )

                    valid_grid_m32 = (batch * out_h, 1)
                    conv2d_forward_tme_3x3s1_p1_fp16_bf16_kernel[valid_grid_m32](
                        x_desc_m32,
                        w_desc,
                        y_desc_m32,
                        out_h,
                        out_w,
                        forward_in_h,
                        forward_in_w,
                        OW_START=_TME_3X3_VALID_BLOCK_M64,
                        BLOCK_M=_TME_3X3_VALID_BLOCK_M32,
                        BLOCK_K=_TME_3X3_BLOCK_K,
                        BLOCK_N=_TME_3X3_BLOCK_N,
                        IS_BF16=input.dtype == torch.bfloat16,
                        num_warps=4,
                        num_stages=1,
                    )

                    tail_start_w = _TME_3X3_VALID_TME_WIDTH

                else:
                    # p1/p2 retain the proven M128 path.
                    x_desc = TensorDescriptor.from_tensor(
                        x_matrix,
                        [_TME_3X3_BLOCK_M, _TME_3X3_BLOCK_K],
                    )
                    y_desc = TensorDescriptor.from_tensor(
                        y_matrix,
                        [_TME_3X3_BLOCK_M, _TME_3X3_BLOCK_N],
                    )

                    tme_main_w = (out_w // _TME_3X3_BLOCK_M) * _TME_3X3_BLOCK_M

                    tme_grid = (
                        batch * out_h,
                        tme_main_w // _TME_3X3_BLOCK_M,
                    )

                    conv2d_forward_tme_3x3s1_p1_fp16_bf16_kernel[tme_grid](
                        x_desc,
                        w_desc,
                        y_desc,
                        out_h,
                        out_w,
                        forward_in_h,
                        forward_in_w,
                        OW_START=0,
                        BLOCK_M=_TME_3X3_BLOCK_M,
                        BLOCK_K=_TME_3X3_BLOCK_K,
                        BLOCK_N=_TME_3X3_BLOCK_N,
                        IS_BF16=input.dtype == torch.bfloat16,
                        num_warps=4,
                        num_stages=1,
                    )

                    tail_start_w = tme_main_w

                # Common right-edge packed fallback.
                tail_w = out_w - tail_start_w
                if tail_w > 0:
                    tail_grid = lambda meta: (
                        triton.cdiv(
                            batch * out_h * tail_w,
                            meta["BLOCK_NI_HO_WO"],
                        ),
                        triton.cdiv(
                            out_channels,
                            meta["BLOCK_CO"],
                        ),
                        1,
                    )

                    conv2d_forward_packed_boundary_kernel[tail_grid](
                        forward_input,
                        packed_weight,
                        output,
                        output,  # dummy bias ptr; HAS_BIAS=False
                        batch,
                        forward_in_h,
                        forward_in_w,
                        out_channels,
                        out_h,
                        out_w,
                        0,
                        tail_start_w,
                        out_h,
                        tail_w,
                        *forward_input.stride(),
                        *packed_weight.stride(),
                        *output.stride(),
                        channels_per_group,
                        kernel_h,
                        kernel_w,
                        stride_h,
                        stride_w,
                        forward_pad_h,
                        forward_pad_w,
                        dilation_h,
                        dilation_w,
                        groups=1,
                        HAS_BIAS=False,
                    )

            elif use_3x3_full_output_fastpath:
                # Full-output 3x3 route: bypass interior/boundary splitting
                # and reuse the original single packed kernel.
                #
                # fp16/bf16 p1/p2/valid:
                #   input is channels-last;
                #   p1/p2 are explicitly pre-padded;
                #   effective padding is zero;
                #   every sampled coordinate is spatially valid.
                #
                # fp32 p1:
                #   retain the proven single-packed power-of-two mapping but
                #   do not force the CL preprocessing that regressed FP32.
                #
                # The p1 benchmark still preserves favorable OW=128 indexing.
                if bias is None:
                    conv2d_forward_no_bias_packed_kernel[grid](
                        forward_input,
                        packed_weight,
                        output,
                        batch,
                        forward_in_h,
                        forward_in_w,
                        out_channels,
                        out_h,
                        out_w,
                        *forward_input.stride(),
                        *packed_weight.stride(),
                        *output.stride(),
                        channels_per_group,
                        kernel_h,
                        kernel_w,
                        stride_h,
                        stride_w,
                        forward_pad_h,
                        forward_pad_w,
                        dilation_h,
                        dilation_w,
                        groups=groups,
                    )
                else:
                    conv2d_forward_packed_kernel[grid](
                        forward_input,
                        packed_weight,
                        output,
                        bias,
                        batch,
                        forward_in_h,
                        forward_in_w,
                        out_channels,
                        out_h,
                        out_w,
                        *forward_input.stride(),
                        *packed_weight.stride(),
                        *output.stride(),
                        channels_per_group,
                        kernel_h,
                        kernel_w,
                        stride_h,
                        stride_w,
                        forward_pad_h,
                        forward_pad_w,
                        dilation_h,
                        dilation_w,
                        groups=groups,
                    )

            else:
                interior_bounds = _compute_interior_output_bounds(
                    forward_in_h,
                    forward_in_w,
                    out_h,
                    out_w,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    forward_pad_h,
                    forward_pad_w,
                    dilation_h,
                    dilation_w,
                )

                use_interior_fastpath = _should_use_interior_fastpath(
                    batch,
                    out_h,
                    out_w,
                    interior_bounds,
                )

                if use_interior_fastpath:
                    oh0, oh1, ow0, ow1 = interior_bounds
                    has_bias = bias is not None
                    bias_ptr = bias if has_bias else output
                    out_per_group = out_channels // groups

                    def launch_interior_rowwise(
                        region_oh_start,
                        region_ow_start,
                        region_h,
                        region_w,
                    ):
                        if region_h <= 0 or region_w <= 0:
                            return

                        # One program owns one batch/output row, one contiguous
                        # output-width tile, and one (group, CO-tile) pair.
                        interior_grid = lambda meta: (
                            batch * region_h,
                            triton.cdiv(
                                region_w,
                                meta["BLOCK_NI_HO_WO"],
                            ),
                            groups
                            * triton.cdiv(
                                out_per_group,
                                meta["BLOCK_CO"],
                            ),
                        )

                        conv2d_forward_packed_interior_kernel[interior_grid](
                            forward_input,
                            packed_weight,
                            output,
                            bias_ptr,
                            batch,
                            forward_in_h,
                            forward_in_w,
                            out_channels,
                            out_h,
                            out_w,
                            region_oh_start,
                            region_ow_start,
                            region_h,
                            region_w,
                            *forward_input.stride(),
                            *packed_weight.stride(),
                            *output.stride(),
                            channels_per_group,
                            kernel_h,
                            kernel_w,
                            stride_h,
                            stride_w,
                            forward_pad_h,
                            forward_pad_w,
                            dilation_h,
                            dilation_w,
                            groups=groups,
                            HAS_BIAS=has_bias,
                        )

                    def launch_boundary_region(
                        region_oh_start,
                        region_ow_start,
                        region_h,
                        region_w,
                    ):
                        if region_h <= 0 or region_w <= 0:
                            return

                        region_grid = lambda meta: (
                            triton.cdiv(
                                batch * region_h * region_w,
                                meta["BLOCK_NI_HO_WO"],
                            ),
                            triton.cdiv(
                                out_per_group,
                                meta["BLOCK_CO"],
                            ),
                            groups,
                        )

                        conv2d_forward_packed_boundary_kernel[region_grid](
                            forward_input,
                            packed_weight,
                            output,
                            bias_ptr,
                            batch,
                            forward_in_h,
                            forward_in_w,
                            out_channels,
                            out_h,
                            out_w,
                            region_oh_start,
                            region_ow_start,
                            region_h,
                            region_w,
                            *forward_input.stride(),
                            *packed_weight.stride(),
                            *output.stride(),
                            channels_per_group,
                            kernel_h,
                            kernel_w,
                            stride_h,
                            stride_w,
                            forward_pad_h,
                            forward_pad_w,
                            dilation_h,
                            dilation_w,
                            groups=groups,
                            HAS_BIAS=has_bias,
                        )

                    # Interior: row-wise OW tiling and no H/W bounds predicates.
                    launch_interior_rowwise(
                        oh0,
                        ow0,
                        oh1 - oh0,
                        ow1 - ow0,
                    )

                    # Boundary fallback.  These four rectangles are disjoint and,
                    # together with the interior rectangle, cover the full output.
                    #
                    # top / bottom span the full output width.
                    launch_boundary_region(
                        0,
                        0,
                        oh0,
                        out_w,
                    )
                    launch_boundary_region(
                        oh1,
                        0,
                        out_h - oh1,
                        out_w,
                    )

                    # left / right cover only the interior-height band, avoiding
                    # overlap with top and bottom.
                    launch_boundary_region(
                        oh0,
                        0,
                        oh1 - oh0,
                        ow0,
                    )
                    launch_boundary_region(
                        oh0,
                        ow1,
                        oh1 - oh0,
                        out_w - ow1,
                    )

                else:
                    # Boundary-heavy or smaller packed convolution: use the
                    # original single packed kernel to avoid extra launch overhead.
                    if bias is None:
                        conv2d_forward_no_bias_packed_kernel[grid](
                            forward_input,
                            packed_weight,
                            output,
                            batch,
                            forward_in_h,
                            forward_in_w,
                            out_channels,
                            out_h,
                            out_w,
                            *forward_input.stride(),
                            *packed_weight.stride(),
                            *output.stride(),
                            channels_per_group,
                            kernel_h,
                            kernel_w,
                            stride_h,
                            stride_w,
                            forward_pad_h,
                            forward_pad_w,
                            dilation_h,
                            dilation_w,
                            groups=groups,
                        )
                    else:
                        conv2d_forward_packed_kernel[grid](
                            forward_input,
                            packed_weight,
                            output,
                            bias,
                            batch,
                            forward_in_h,
                            forward_in_w,
                            out_channels,
                            out_h,
                            out_w,
                            *forward_input.stride(),
                            *packed_weight.stride(),
                            *output.stride(),
                            channels_per_group,
                            kernel_h,
                            kernel_w,
                            stride_h,
                            stride_w,
                            forward_pad_h,
                            forward_pad_w,
                            dilation_h,
                            dilation_w,
                            groups=groups,
                        )
        else:
            # Original OIHW path for small convolutions.  Keeping this path also
            # preserves the exact kernel used by backward when it computes dX.
            if bias is None:
                conv2d_forward_no_bias_kernel[grid](
                    forward_input,
                    weight,
                    output,
                    batch,
                    forward_in_h,
                    forward_in_w,
                    out_channels,
                    out_h,
                    out_w,
                    *forward_input.stride(),
                    *weight.stride(),
                    *output.stride(),
                    channels_per_group,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    forward_pad_h,
                    forward_pad_w,
                    dilation_h,
                    dilation_w,
                    groups=groups,
                )
            else:
                conv2d_forward_kernel[grid](
                    forward_input,
                    weight,
                    output,
                    bias,
                    batch,
                    forward_in_h,
                    forward_in_w,
                    out_channels,
                    out_h,
                    out_w,
                    *forward_input.stride(),
                    *weight.stride(),
                    *output.stride(),
                    channels_per_group,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    forward_pad_h,
                    forward_pad_w,
                    dilation_h,
                    dilation_w,
                    groups=groups,
                )

        ctx.save_for_backward(weight, original_input, bias)
        ctx.stride = (stride_h, stride_w)
        ctx.padding = (pad_h, pad_w)
        ctx.dilation = (dilation_h, dilation_w)
        ctx.weight_info = (
            out_channels // groups,
            channels_per_group,
            kernel_h,
            kernel_w,
        )
        ctx.input_info = (batch, in_h, in_w)
        ctx.output_info = (out_h, out_w)
        ctx.groups = groups
        ctx.device = input.device
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS_MTHREADS CONV2D_VJP")

        weight, input, bias = ctx.saved_tensors
        out_per_group, channels_per_group, kernel_h, kernel_w = ctx.weight_info
        batch, in_h, in_w = ctx.input_info
        out_h, out_w = ctx.output_info
        groups = ctx.groups
        stride_h, stride_w = ctx.stride
        pad_h, pad_w = ctx.padding
        dilation_h, dilation_w = ctx.dilation
        device = ctx.device

        # Build the transposed-convolution form used to obtain grad_input.
        reverse_pad_h = dilation_h * (kernel_h - 1) - pad_h
        reverse_pad_w = dilation_w * (kernel_w - 1) - pad_w

        flipped = torch.flip(weight, dims=(2, 3)).contiguous()
        if groups == 1:
            flipped = flipped.transpose(0, 1).contiguous()
        else:
            flipped = flipped.reshape(
                groups,
                out_per_group,
                channels_per_group,
                kernel_h,
                kernel_w,
            )
            flipped = (
                flipped.transpose(1, 2)
                .reshape(
                    groups * channels_per_group,
                    out_per_group,
                    kernel_h,
                    kernel_w,
                )
                .contiguous()
            )

        expanded_h = grad_output.shape[2] + (stride_h - 1) * (grad_output.shape[2] - 1)
        expanded_w = grad_output.shape[3] + (stride_w - 1) * (grad_output.shape[3] - 1)

        if stride_h == 1 and stride_w == 1:
            expanded_grad = grad_output
        else:
            expanded_grad = torch.zeros(
                grad_output.shape[0],
                grad_output.shape[1],
                expanded_h,
                expanded_w,
                device=device,
                dtype=grad_output.dtype,
            )
            expanded_grad[:, :, ::stride_h, ::stride_w] = grad_output

        grad_input = torch.empty(
            (batch, groups * channels_per_group, in_h, in_w),
            device=device,
            dtype=input.dtype,
        )

        grid_dx = lambda meta: (
            triton.cdiv(batch * in_h * in_w, meta["BLOCK_NI_HO_WO"]),
            triton.cdiv(channels_per_group, meta["BLOCK_CO"]),
            groups,
        )
        conv2d_forward_no_bias_kernel[grid_dx](
            expanded_grad,
            flipped,
            grad_input,
            batch,
            expanded_h,
            expanded_w,
            groups * channels_per_group,
            in_h,
            in_w,
            *expanded_grad.stride(),
            *flipped.stride(),
            *grad_input.stride(),
            out_per_group,
            kernel_h,
            kernel_w,
            1,
            1,
            reverse_pad_h,
            reverse_pad_w,
            dilation_h,
            dilation_w,
            groups=groups,
        )

        grad_weight = torch.empty_like(weight)
        grid_dw = lambda meta: (
            triton.cdiv(
                channels_per_group * kernel_h * kernel_w,
                meta["BLOCK_CI_HK_WK"],
            ),
            groups,
            triton.cdiv(out_per_group, meta["BLOCK_CO"]),
        )
        conv2d_backward_kernel_weight[grid_dw](
            input,
            grad_output,
            grad_weight,
            *input.stride(),
            *grad_weight.stride(),
            *grad_output.stride(),
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            channels_per_group,
            batch,
            stride_h,
            stride_w,
            out_h,
            out_w,
            out_per_group,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
        )

        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.to(torch.float64).sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None


def _normalize_2d(value, name):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly 2 values")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Drop-in 2D convolution entry used by the MThreads backend."""

    if isinstance(padding, str):
        mode = padding.lower()
        if mode == "valid":
            return Conv2d.apply(input, weight, bias, stride, 0, dilation, groups)

        if mode != "same":
            raise ValueError("padding must be an integer/tuple, 'valid', or 'same'")

        stride_h, stride_w = _normalize_2d(stride, "stride")
        if stride_h != 1 or stride_w != 1:
            raise AssertionError("padding='same' only supports stride=1")

        dilation_h, dilation_w = _normalize_2d(dilation, "dilation")
        in_h, in_w = input.shape[-2:]
        kernel_h, kernel_w = weight.shape[-2:]

        # For stride=1, total padding needed by SAME is dilation*(K-1).
        total_h = dilation_h * (kernel_h - 1)
        total_w = dilation_w * (kernel_w - 1)
        top = total_h // 2
        left = total_w // 2
        bottom = total_h - top
        right = total_w - left

        # The Triton kernel accepts symmetric per-axis padding.  If total padding
        # is odd, over-pad one side and trim after the convolution.
        sym_h = max(top, bottom)
        sym_w = max(left, right)
        out = Conv2d.apply(
            input,
            weight,
            bias,
            stride,
            (sym_h, sym_w),
            dilation,
            groups,
        )

        trim_top = sym_h - top
        trim_left = sym_w - left
        return out[..., trim_top : trim_top + in_h, trim_left : trim_left + in_w]

    return Conv2d.apply(input, weight, bias, stride, padding, dilation, groups)
