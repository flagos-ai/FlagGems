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

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.conv2d import conv2d_output_size
from flag_gems.ops.conv3d import conv3d_output_size
from flag_gems.ops.cudnn_convolution import (
    _AABS_PRESENT,
    _DOT_MIN_K,
    _aabs_disabled,
    cudnn_convolution,
)
from flag_gems.utils import libentry, pointwise_dynamic

logger = logging.getLogger(__name__)


# aten::cudnn_convolution_relu is the fused cuDNN convolution + ReLU: it
# returns relu(conv(self, weight) + bias), with the bias added before the
# activation (measured against the ATen operator on a 1x1 conv: relu(conv +
# bias) gives 0 where relu(conv) + bias gives 0.5).
#
# A fused kernel is what keeps this operator competitive with cuDNN's own
# epilogue: launching the plain conv and a separate ReLU doubles the number of
# launches (and adds a full output write+read) which costs ~0.02ms per call --
# more than the whole runtime of the small benchmark shapes. The convolution
# kernels below mirror conv2d/conv3d's implicit-GEMM structure (fp32
# accumulator, tl.dot over im2col tiles, channel-padding for small C_in) with
# the bias and max(0, .) folded into the epilogue, so the conv is rounded to
# the output dtype exactly once, as cuDNN does.
#
# The ReLU-only fallback (no bias, or shapes that escape the fused kernels)
# runs as a pointwise epilogue on the conv output.


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def relu_epi_func(x):
    return tl.where(x > 0, x, 0)


# ---------------------------------------------------------------------------
# 2D fused conv + bias + ReLU (derived from conv2d_forward_kernel)
#
# GCU400 rework of the tap loop (measured, bench kernel mode, fp16):
#   (8,3,224,224)x64k3    39.7 -> 26.6ms    (128,16,64) config class
#   (16,64,56,56)x64k3    27.3 -> 10.3ms    (128,64,64)
#   (16,512,7,7)x512k3    25.0 ->  5.6ms    (32,32,256)
#   no-bias (16,64,56,56)x128k3 51.5 -> 16.1ms (32,32,128, rerouted)
# Three structural changes carry the win, in order of impact:
#   1. tap loop static-unrolled (tl.static_range over KH*KW, both constexpr):
#      the per-iteration h/w div/mod chain and the tap row base pointer are
#      constant-folded, instead of being recomputed from the loop index every
#      iteration like the dynamic hwc loop does.
#   2. row quantities (n offset, sh*ho, sw*wo) hoisted out of the K loop; the
#      per-load mask collapses to one broadcast AND (row_ok & c_ok) instead of
#      six 2D compares per iteration; EVEN_C drops the channel mask entirely.
#   3. HAS_BIAS=False arm skips the bias load+add, so the no-bias reroute
#      needs neither a torch.zeros fill launch nor a dummy bias gather.
# The accumulation order (tap-major, c-fastest) matches the generic kernel.
# ---------------------------------------------------------------------------


def _select_conv2d_relu_config(in_n, out_h, out_w, out_c, weight_c, groups, dtype):
    """Deterministic tile choice for the fused conv+bias+relu kernel.

    Replaces the single-config autotuner: per-shape sweeps (fp16/bf16, A/B/C/N
    bench classes + test-suite edges) land each class on its own optimum, and
    two GCU constraints are encoded explicitly:
      * BLOCK_CI=64 miscompiles when the channel loop runs more than one
        iteration (fp16 (16,512,7,7)x512k3 rel err 8.3e2, bf16 NaN-adjacent
        4.7e10) but is correct and fastest at CI_COUNT==1, so it is picked
        only for exactly-64-channel inputs.
      * fp32 tl.dot miscompiles under BLOCK_NI<128 tiles here (P07 family:
        (16,64,56,56) rel 2.4e1 at (64,32,64), 8e2 at (32,32,128)); every
        BLOCK_NI=128 shape tested is exact, so fp32 pins that floor.
    weight_c is the (already padded) per-group channel count.
    """
    rows = in_n * out_h * out_w
    cols = out_c // groups
    if weight_c <= 16:
        if rows >= 131072:
            config = (128, 16, 64, 4, 2)
        else:
            config = (64, 16, 64, 4, 2)
    elif weight_c <= 32:
        config = (64, 32, 64, 4, 2)
    elif weight_c == 64:
        config = (128, 64, 64, 4, 2) if cols <= 64 else (32, 32, 128, 4, 2)
    else:
        config = (32, 32, 256, 4, 2) if cols >= 256 else (32, 32, 128, 4, 2)
    if dtype == torch.float32 and config[0] < 128:
        block_ci = 64 if weight_c == 64 else 16
        config = (128, block_ci, 64, 4, 2)
    return config


@libentry()
@triton.jit
def conv2d_relu_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    in_n,
    input_height,
    input_width,
    out_c,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_c_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    weight_c: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    EVEN_C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_ni_ho_wo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    ni_ho_wo_offset = pid_ni_ho_wo * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    ni_ho_offset = ni_ho_wo_offset // out_width
    in_n_point_value = ni_ho_offset // out_height
    output_height_point_value = ni_ho_offset % out_height
    output_width_point_value = ni_ho_wo_offset % out_width

    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    # Row quantities hoisted out of both loops: the per-row n/c-group base and
    # the strided spatial offsets, plus the tap-invariant n bound.
    row_base = input_pointer + (
        input_n_stride * in_n_point_value
        + input_c_stride * pid_group * weight_c
        + input_height_stride * stride_height * output_height_point_value
        + input_width_stride * stride_width * output_width_point_value
    )
    row_in_bounds = in_n_point_value < in_n

    weight_base = weight_pointer + (
        weight_n_stride * output_c_offset
        + weight_n_stride * pid_group * out_per_group_c
    )
    co_ok = output_c_offset < out_per_group_c

    accum = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI
    for tap in tl.static_range(weight_height * weight_width):
        h = tap // weight_width
        w = tap % weight_width
        ih = h * dilation_height - padding_height
        iw = w * dilation_width - padding_width
        tap_row_ptr = row_base + (input_height_stride * ih + input_width_stride * iw)
        row_ok = (
            row_in_bounds
            & (ih + stride_height * output_height_point_value >= 0)
            & (ih + stride_height * output_height_point_value < input_height)
            & (iw + stride_width * output_width_point_value >= 0)
            & (iw + stride_width * output_width_point_value < input_width)
        )
        tap_weight_ptr = weight_height_stride * h + weight_width_stride * w
        for cb in range(BLOCK_CI_COUNT):
            input_c_offset = cb * BLOCK_CI + tl.arange(0, BLOCK_CI)
            curr_input_pointer = (
                tap_row_ptr[:, None] + (input_c_stride * input_c_offset)[None, :]
            )
            curr_weight_pointer = (
                weight_base[None, :]
                + (weight_c_stride * input_c_offset)[:, None]
                + tap_weight_ptr
            )
            if EVEN_C:
                input_block = tl.load(curr_input_pointer, mask=row_ok[:, None])
                weight_block = tl.load(curr_weight_pointer, mask=co_ok[None, :])
            else:
                c_ok = input_c_offset < weight_c
                input_block = tl.load(
                    curr_input_pointer, mask=row_ok[:, None] & c_ok[None, :]
                )
                weight_block = tl.load(
                    curr_weight_pointer, mask=c_ok[:, None] & co_ok[None, :]
                )
            accum += tl.dot(input_block, weight_block, allow_tf32=False)

    if HAS_BIAS:
        bias_pointer += (pid_group[None] * out_per_group_c)[None, :] + output_c_offset[
            None, :
        ]
        mask_bias = (output_c_offset < out_per_group_c)[None, :]
        bias = tl.load(bias_pointer, mask_bias).to(tl.float32)
        accum = accum + bias
    # The activation rides on the fp32 accumulator so the sum is rounded to
    # the output dtype exactly once -- a bias added after storing would round
    # twice and flip near-zero sums across the ReLU boundary.
    accum = tl.maximum(accum, 0.0)

    output_pointer += (
        (output_n_stride * in_n_point_value)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + output_c_offset))[None, :]
        + (output_height_stride * output_height_point_value)[:, None]
        + (output_width_stride * output_width_point_value)[:, None]
    )
    output_mask = (
        (in_n_point_value < in_n)[:, None]
        & (output_c_offset < out_per_group_c)[None, :]
        & (output_height_point_value < out_height)[:, None]
        & (output_width_point_value < out_width)[:, None]
    )

    tl.store(output_pointer, accum, mask=output_mask)


# ---------------------------------------------------------------------------
# 3D fused conv + bias + ReLU (mirrors conv3d_forward_kernel)
# ---------------------------------------------------------------------------


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("conv3d_forward"),
    key=[
        "in_n",
        "weight_c",
        "input_depth",
        "input_height",
        "input_width",
        "out_c",
        "out_depth",
        "out_height",
        "out_width",
        "weight_depth",
        "weight_height",
        "weight_width",
        "stride_depth",
        "stride_height",
        "stride_width",
        "padding_depth",
        "padding_height",
        "padding_width",
        "groups",
    ],
)
@triton.jit
def conv3d_relu_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    in_n,
    input_depth,
    input_height,
    input_width,
    out_c,
    out_depth,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_depth_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_c_stride,
    weight_depth_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_depth_stride,
    output_height_stride,
    output_width_stride,
    weight_c: tl.constexpr,
    weight_depth: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_depth: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_depth: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_depth: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_DO_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_ni_do_ho_wo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    ni_do_ho_wo_offset = pid_ni_do_ho_wo * BLOCK_NI_DO_HO_WO + tl.arange(
        0, BLOCK_NI_DO_HO_WO
    )
    ni_do_ho_offset = ni_do_ho_wo_offset // out_width
    ni_do_offset = ni_do_ho_offset // out_height
    in_n_point_value = ni_do_offset // out_depth
    output_depth_point_value = ni_do_offset % out_depth
    output_height_point_value = ni_do_ho_offset % out_height
    output_width_point_value = ni_do_ho_wo_offset % out_width

    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    input_pointer += (
        input_n_stride * in_n_point_value + input_c_stride * pid_group * weight_c
    )[:, None]
    weight_pointer += (
        weight_n_stride * output_c_offset
        + weight_n_stride * pid_group * out_per_group_c
    )[None, :]

    accum = tl.zeros((BLOCK_NI_DO_HO_WO, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI
    for dhwc in range(weight_depth * weight_height * weight_width * BLOCK_CI_COUNT):
        c = (dhwc % BLOCK_CI_COUNT) * BLOCK_CI
        dhw = dhwc // BLOCK_CI_COUNT
        d = dhw // (weight_height * weight_width)
        hw = dhw % (weight_height * weight_width)
        h = hw // weight_width
        w = hw % weight_width

        input_c_offset = c + tl.arange(0, BLOCK_CI)
        input_depth_offset = (
            d * dilation_depth - padding_depth + stride_depth * output_depth_point_value
        )
        input_height_offset = (
            h * dilation_height
            - padding_height
            + stride_height * output_height_point_value
        )
        input_width_offset = (
            w * dilation_width - padding_width + stride_width * output_width_point_value
        )

        curr_input_pointer = (
            input_pointer
            + (input_c_stride * input_c_offset)[None, :]
            + (input_depth_stride * input_depth_offset)[:, None]
            + (input_height_stride * input_height_offset)[:, None]
            + (input_width_stride * input_width_offset)[:, None]
        )
        curr_weight_pointer = (
            weight_pointer
            + (weight_c_stride * input_c_offset)[:, None]
            + (weight_depth_stride * d)
            + (weight_height_stride * h)
            + (weight_width_stride * w)
        )

        input_mask = (
            (in_n_point_value < in_n)[:, None]
            & (input_c_offset < weight_c)[None, :]
            & (0 <= input_depth_offset)[:, None]
            & (input_depth_offset < input_depth)[:, None]
            & (0 <= input_height_offset)[:, None]
            & (input_height_offset < input_height)[:, None]
            & (0 <= input_width_offset)[:, None]
            & (input_width_offset < input_width)[:, None]
        )
        weight_mask = (input_c_offset < weight_c)[:, None] & (
            output_c_offset < out_per_group_c
        )[None, :]

        input_block = tl.load(curr_input_pointer, mask=input_mask)
        weight_block = tl.load(curr_weight_pointer, mask=weight_mask)

        accum += tl.dot(input_block, weight_block, allow_tf32=False)

    bias_pointer += (pid_group[None] * out_per_group_c)[None, :] + output_c_offset[
        None, :
    ]
    mask_bias = (output_c_offset < out_per_group_c)[None, :]
    bias = tl.load(bias_pointer, mask_bias).to(tl.float32)
    accum = accum + bias
    # See conv2d_relu_forward_kernel: single rounding before the clip.
    accum = tl.maximum(accum, 0.0)

    output_pointer += (
        (output_n_stride * in_n_point_value)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + output_c_offset))[None, :]
        + (output_depth_stride * output_depth_point_value)[:, None]
        + (output_height_stride * output_height_point_value)[:, None]
        + (output_width_stride * output_width_point_value)[:, None]
    )
    output_mask = (
        (in_n_point_value < in_n)[:, None]
        & (output_c_offset < out_per_group_c)[None, :]
        & (output_depth_point_value < out_depth)[:, None]
        & (output_height_point_value < out_height)[:, None]
        & (output_width_point_value < out_width)[:, None]
    )

    tl.store(output_pointer, accum, mask=output_mask)


def _to_list(param, ndim):
    """Broadcast a scalar spatial parameter to one entry per spatial dimension."""
    if isinstance(param, (list, tuple)):
        return list(param)
    return [param] * ndim


def _check_dtype(input):
    # cuDNN engines for the fused conv+bias+relu exist only for the three
    # float types (measured: fp64/int8 fail with "unable to find an engine"
    # on ATen), so anything else is rejected up front.
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"cudnn_convolution_relu only supports float16/bfloat16/float32, got {input.dtype}"


def _output_size(in_size, kernel, stride, padding, dilation):
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def _launch_conv2d_relu(
    input, weight, bias, stride, padding, dilation, groups, out=None
):
    in_n, _, input_height, input_width = input.shape
    out_c, weight_c, weight_height, weight_width = weight.shape
    stride_height, stride_width = stride
    padding_height, padding_width = padding
    dilation_height, dilation_width = dilation

    # tl.dot's K is BLOCK_CI-tiled over weight_c; AABS shrinks that block to
    # next_power_of_2(weight_c) when weight_c < 16, breaking the K >= 16
    # floor. Pad the channel dim the same way conv2d does (see conv2d.Conv2d
    # .forward for the grouped-conv reshaping).
    _MIN_DOT_K = 16
    if weight_c < _MIN_DOT_K:
        pad_c = _MIN_DOT_K - weight_c
        if groups == 1:
            input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, pad_c))
        else:
            N, C, H, W = input.shape
            input = input.reshape(N, groups, weight_c, H, W)
            input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, pad_c))
            input = input.reshape(N, groups * (weight_c + pad_c), H, W)
        weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, pad_c))
        weight_c = weight_c + pad_c

    out_height = conv2d_output_size(
        input_height, weight_height, stride_height, padding_height, dilation_height
    )
    out_width = conv2d_output_size(
        input_width, weight_width, stride_width, padding_width, dilation_width
    )
    output_shape = (in_n, out_c, out_height, out_width)
    if (
        out is not None
        and out.shape == output_shape
        and out.dtype == input.dtype
        and out.device == input.device
    ):
        # The .out overload's caller-supplied buffer takes the kernel's stores
        # directly (its strides are passed through like a fresh allocation's),
        # which drops the separate copy epilogue.
        output = out
    else:
        output = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    (
        block_ni_ho_wo,
        block_ci,
        block_co,
        num_warps,
        num_stages,
    ) = _select_conv2d_relu_config(
        in_n,
        out_height,
        out_width,
        out_c,
        weight_c,
        groups,
        input.dtype,
    )
    grid = (
        triton.cdiv(in_n * out_height * out_width, block_ni_ho_wo),
        triton.cdiv(int(out_c // groups), block_co),
        groups,
    )
    # bias may be None (the no-bias reroute); the pointer argument still needs
    # a tensor, so hand it the weight -- HAS_BIAS=False guards every access.
    bias_pointer = bias if bias is not None else weight
    conv2d_relu_forward_kernel[grid](
        input,
        weight,
        bias_pointer,
        output,
        in_n,
        input_height,
        input_width,
        out_c,
        out_height,
        out_width,
        *input.stride(),
        *weight.stride(),
        *output.stride(),
        weight_c,
        weight_height,
        weight_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        dilation_height,
        dilation_width,
        groups=groups,
        BLOCK_NI_HO_WO=block_ni_ho_wo,
        BLOCK_CI=block_ci,
        BLOCK_CO=block_co,
        EVEN_C=(weight_c % block_ci == 0),
        HAS_BIAS=bias is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def _launch_conv3d_relu(input, weight, bias, stride, padding, dilation, groups):
    in_n, _, input_depth, input_height, input_width = input.shape
    out_c, weight_c, weight_depth, weight_height, weight_width = weight.shape
    stride_depth, stride_height, stride_width = stride
    padding_depth, padding_height, padding_width = padding
    dilation_depth, dilation_height, dilation_width = dilation

    _MIN_DOT_K = 16
    if weight_c < _MIN_DOT_K:
        pad_c = _MIN_DOT_K - weight_c
        if groups == 1:
            input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, 0, 0, pad_c))
        else:
            N, C, D, H, W = input.shape
            input = input.reshape(N, groups, weight_c, D, H, W)
            input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, 0, 0, pad_c))
            input = input.reshape(N, groups * (weight_c + pad_c), D, H, W)
        weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, 0, 0, pad_c))
        weight_c = weight_c + pad_c

    out_depth = conv3d_output_size(
        input_depth, weight_depth, stride_depth, padding_depth, dilation_depth
    )
    out_height = conv3d_output_size(
        input_height, weight_height, stride_height, padding_height, dilation_height
    )
    out_width = conv3d_output_size(
        input_width, weight_width, stride_width, padding_width, dilation_width
    )
    output = torch.empty(
        (in_n, out_c, out_depth, out_height, out_width),
        device=input.device,
        dtype=input.dtype,
    )

    grid = lambda META: (
        triton.cdiv(
            in_n * out_depth * out_height * out_width, META["BLOCK_NI_DO_HO_WO"]
        ),
        triton.cdiv(int(out_c // groups), META["BLOCK_CO"]),
        groups,
    )
    conv3d_relu_forward_kernel[grid](
        input,
        weight,
        bias,
        output,
        in_n,
        input_depth,
        input_height,
        input_width,
        out_c,
        out_depth,
        out_height,
        out_width,
        *input.stride(),
        *weight.stride(),
        *output.stride(),
        weight_c,
        weight_depth,
        weight_height,
        weight_width,
        stride_depth,
        stride_height,
        stride_width,
        padding_depth,
        padding_height,
        padding_width,
        dilation_depth,
        dilation_height,
        dilation_width,
        groups=groups,
    )
    return output


def _conv_forward(
    input, weight, bias, stride, padding, dilation, groups, ndim, out=None
):
    """Run the convolution, keeping the bias + ReLU inside the conv kernels.

    cuDNN computes fp32(conv) + bias and rounds the sum once. Adding the bias
    to the *stored* conv output would round twice -- and a double rounding
    moves near-zero sums across the ReLU boundary, where the result is 0 for
    the reference but a full low-precision ulp here. So the bias path runs the
    fused conv2d/conv3d + bias + ReLU kernels (fp32 accumulator, one
    rounding); the no-bias path keeps cudnn_convolution's dedicated
    depthwise/pointwise/direct kernels and clips in a pointwise epilogue.
    """
    if bias is None:
        return cudnn_convolution(
            input,
            weight,
            padding,
            stride,
            dilation,
            groups,
            False,  # benchmark
            False,  # deterministic
            False,  # allow_tf32
        )
    # The cuDNN engine requires exactly one bias value per output channel,
    # delivered as (C_out,), (1, C_out, 1, ...) or a scalar; flatten to the
    # flat (C_out,) vector the conv kernels index by output channel.
    flat_bias = bias.reshape(weight.shape[0])
    if ndim == 2:
        return _launch_conv2d_relu(
            input, weight, flat_bias, stride, padding, dilation, groups, out=out
        )
    return _launch_conv3d_relu(
        input, weight, flat_bias, stride, padding, dilation, groups
    )


# Size threshold (number of MACs) for promoting 16bit → fp32 computation. Measured on this machine (bias-free fused
# path, pad=0):
#   4.1M  MACs promoted: 0.401ms vs native 0.209ms (1.92x slower; conversion overhead dominates)
#   24.9M MACs promoted: 0.730ms vs native 0.445ms (1.64x slower)
#   99.7M MACs promoted: 1.501ms vs native 1.270ms (1.18x slower)
#   398.7M MACs promoted: 5.208ms vs native 9.213ms (1.77x faster)
#   3439.9M MACs promoted: 37.223ms vs native 79.946ms (2.15x faster)
# The crossover lies between 1e8~4e8, so 2e8 is used as the switch; small convolutions still use the original dtype
# path.
_UPCAST_MIN_MACS = 2.0e8


def _conv_macs(input, weight, stride, padding, dilation):
    """Estimates the MAC count of one convolution to decide whether a wider type pays off."""
    ndim = input.ndim - 2
    stride = _to_list(stride, ndim)
    padding = _to_list(padding, ndim)
    dilation = _to_list(dilation, ndim)
    macs = input.shape[0] * weight.shape[0]
    for dim in range(ndim):
        in_size = input.shape[2 + dim]
        k = weight.shape[2 + dim]
        out_size = (in_size + 2 * padding[dim] - dilation[dim] * (k - 1) - 1) // stride[
            dim
        ] + 1
        macs *= max(out_size, 0) * k
    return macs * weight.shape[1]


def _should_compute_in_fp32(input, weight, bias, stride, padding, dilation, groups):
    """Decides whether a 16-bit input should be widened to fp32.

    Enabled only on the bias-free fused path: of three bias-carrying benchmark shapes two
    regressed and one improved after widening ((8,3,224,224) 42.7->45.6ms, (16,64,56,56)x64k3
    18.3->21.4ms slower, (16,512,7,7)x512k3 21.8->19.1ms faster), so the gain is inconsistent;
    the five bias-free shapes all follow "widen once the size exceeds 2e8 MAC".
    """
    if bias is not None:
        return False
    if input.dtype not in (torch.float16, torch.bfloat16):
        return False
    if torch.is_grad_enabled() and (input.requires_grad or weight.requires_grad):
        return False
    return _conv_macs(input, weight, stride, padding, dilation) >= _UPCAST_MIN_MACS


def cudnn_convolution_relu(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
):
    """
    Fused cuDNN-style convolution followed by a ReLU.

    Computes ``relu(conv(input, weight, stride, padding, dilation, groups) +
    bias)``. 1D/2D/3D spatial inputs, grouped convolutions and fp16/bf16/fp32
    dtypes are supported; the bias is accumulated in fp32 before the
    activation, matching the reference's single rounding.

    Args:
        input: Input tensor of shape (N, C_in, *spatial_dims)
        weight: Weight tensor of shape (C_out, C_in/groups, *kernel_dims)
        bias: Optional bias with one value per output channel
        stride: Stride for each spatial dimension
        padding: Padding for each spatial dimension
        dilation: Dilation for each spatial dimension
        groups: Number of groups for grouped convolution

    Returns:
        Output tensor of shape (N, C_out, *out_spatial_dims)
    """
    logger.debug("GEMS_ENFLAME CUDNN_CONVOLUTION_RELU")
    _check_dtype(input)

    if _should_compute_in_fp32(input, weight, bias, stride, padding, dilation, groups):
        # 16bit tensors are computed in fp32 and then downcast on this machine: GCU400's 16bit path is about 2x slower
        # than fp32
        # ((16,64,56,56)x128k3 measured fp16/bf16 79.6ms vs fp32 37.2ms); after promotion,
        # the fp32 kernel handles the compute and memory traffic, and only one downcast copy is needed at the end.
        out_fp32 = cudnn_convolution_relu(
            input.float(), weight.float(), bias, stride, padding, dilation, groups
        )
        return out_fp32.to(input.dtype)

    ndim = input.ndim - 2
    if ndim not in (1, 2, 3):
        raise ValueError(
            f"cudnn_convolution_relu only supports 1D, 2D, and 3D convolutions, "
            f"got input with {ndim} spatial dimensions"
        )
    # AABS mispairs the block sizes of every conv kernel launched below (see
    # flag_gems.ops.cudnn_convolution._aabs_disabled); the knob is read at
    # launch time, so the whole dispatch is covered. Where the fork has no
    # such knob the branch skips the manager outright.
    if _AABS_PRESENT:
        with _aabs_disabled():
            return _dispatch(
                input, weight, bias, stride, padding, dilation, groups, ndim
            )
    return _dispatch(input, weight, bias, stride, padding, dilation, groups, ndim)


def _no_bias_goes_to_fused(input, weight, stride, padding, dilation, groups, ndim):
    """Decide whether a bias-free case runs the fused kernel with HAS_BIAS=False.

    Mirrors the generic cudnn_convolution dispatch far enough to tell which
    branch a bias-free case would land on, and keeps it there whenever that
    branch is the better GCU citizen:
      * depthwise (weight_c == 1 && groups == C_in) and the pointwise 1x1 GEMM
        keep their dedicated generic kernels -- both are verified working on
        GCU and the pointwise path is already a plain GEMM;
      * 3D keeps the generic dispatch (its no-bias arm is only exercised by
        the accuracy suite, not by the core benchmark);
      * everything else would land on the FMA direct-2D kernel -- which fails
        the Triton PassManager on GCU (mlir->gcuir; compiler_problems P02) --
        or on the generic implicit-GEMM conv plus a separate pointwise ReLU
        epilogue, whose conv kernel is the whole latency of the no-bias core
        benchmark case (51.5ms vs 16.1ms rerouted, (16,64,56,56)x128k3 fp16).
    relu(conv + 0) is exactly relu(conv): with HAS_BIAS=False the zero never
    touches the fp32 accumulator, so the reroute is bit-identical to the
    bias arm's single-rounding semantics while dropping the second launch.
    """
    if ndim > 2:
        return False
    weight_c = weight.shape[1]
    if weight_c == 1 and groups == input.shape[1]:
        return False
    if (
        groups == 1
        and all(k == 1 for k in weight.shape[2:])
        and all(s == 1 for s in stride)
        and all(d == 1 for d in dilation)
        and all(p == 0 for p in padding)
        and weight_c >= _DOT_MIN_K
    ):
        return False
    return True


def _dispatch(input, weight, bias, stride, padding, dilation, groups, ndim, out=None):
    stride = _to_list(stride, ndim)
    padding = _to_list(padding, ndim)
    dilation = _to_list(dilation, ndim)
    if bias is None and _no_bias_goes_to_fused(
        input, weight, stride, padding, dilation, groups, ndim
    ):
        if ndim == 1:
            # Lift to 2D with a unit trailing dim (same trick as conv1d ->
            # conv2d). A caller's `out` rides through as an unsqueezed view,
            # which the kernel fills directly.
            lifted_out = out.unsqueeze(-1) if out is not None else None
            res = _launch_conv2d_relu(
                input.unsqueeze(-1),
                weight.unsqueeze(-1),
                None,
                [stride[0], 1],
                [padding[0], 0],
                [dilation[0], 1],
                groups,
                out=lifted_out,
            )
            return out if out is not None else res.squeeze(-1)
        return _launch_conv2d_relu(
            input, weight, None, stride, padding, dilation, groups, out=out
        )
    if ndim == 1:
        # Lift to 2D with a unit trailing dim (same trick as conv1d -> conv2d).
        out2 = _conv_forward(
            input.unsqueeze(-1),
            weight.unsqueeze(-1),
            bias,
            [stride[0], 1],
            [padding[0], 0],
            [dilation[0], 1],
            groups,
            2,
        )
        out2 = out2.squeeze(-1)
        if bias is None:
            return relu_epi_func(out2)
        return out2
    conv = _conv_forward(
        input, weight, bias, stride, padding, dilation, groups, ndim, out=out
    )
    if bias is None:
        return relu_epi_func(conv)
    return conv


def cudnn_convolution_relu_out(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, out=None
):
    """
    Variant of ``cudnn_convolution_relu`` that writes the result into ``out``.

    Args mirror ``cudnn_convolution_relu``; ``out`` must have the conv output
    shape (N, C_out, *out_spatial_dims) and is returned.
    """
    logger.debug("GEMS_ENFLAME CUDNN_CONVOLUTION_RELU_OUT")
    _check_dtype(input)

    ndim = input.ndim - 2
    if ndim not in (1, 2, 3):
        raise ValueError(
            f"cudnn_convolution_relu only supports 1D, 2D, and 3D convolutions, "
            f"got input with {ndim} spatial dimensions"
        )
    if _AABS_PRESENT:
        with _aabs_disabled():
            res = _dispatch(
                input,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
                ndim,
                out=out,
            )
            relu_into_out(res, out)
            return out
    res = _dispatch(
        input, weight, bias, stride, padding, dilation, groups, ndim, out=out
    )
    relu_into_out(res, out)
    return out


def relu_into_out(res, out):
    """Copy the fused result into the caller's out tensor.

    The fused conv kernels write their own fresh allocation, so the .out
    variant copies; the epilogue path reuses pointwise_dynamic's out0 store.
    The fused 2D arm instead takes the caller's buffer directly at launch
    (see _launch_conv2d_relu), where res IS out and the copy is a no-op.
    """
    if res.data_ptr() == out.data_ptr():
        return
    out.copy_(res)
