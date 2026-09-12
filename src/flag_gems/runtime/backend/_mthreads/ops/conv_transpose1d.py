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

from flag_gems.ops.conv_transpose1d import conv_transpose1d as default_conv_transpose1d

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def conv_transpose1d_output_size(
    in_size, kernel_size, stride, padding, output_padding, dilation
):
    return (
        (in_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


# ---------------------------------------------------------------------------
# Kernel 1: FMA outer-product path (all dtypes; required for fp32 where the
# MUSA tl.dot is numerically unsafe).  Residue-class decomposition:
# outputs are split into s classes o = r + m*s; tap activity and input shift
# are scalars per (k, r); input loads are contiguous in m.
# ---------------------------------------------------------------------------
@triton.jit
def conv_transpose1d_fma_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    input_width,
    out_width,
    in_channels_per_group,
    out_channels_per_group,
    M_BLOCKS: tl.constexpr,
    groups: tl.constexpr,
    kernel_width: tl.constexpr,
    stride_width: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_width: tl.constexpr,
    has_bias: tl.constexpr,
    tf32_mode: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # Grid: dim0 = s * M_BLOCKS, dim1 = N * ocpg_blocks, dim2 = groups
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid_g = tl.program_id(2)

    s = stride_width
    ocpg = out_channels_per_group

    r = pid0 // M_BLOCKS
    mb = pid0 % M_BLOCKS
    m = mb * BLOCK_M + tl.arange(0, BLOCK_M)

    oc_blocks = tl.cdiv(ocpg, BLOCK_OC)
    n = pid1 // oc_blocks
    ocb = pid1 % oc_blocks
    oc = ocb * BLOCK_OC + tl.arange(0, BLOCK_OC)

    M_r = (out_width - r + s - 1) // s
    m_valid = m < M_r

    in_channels = in_channels_per_group * groups
    out_channels = ocpg * groups

    input_base = (
        input_ptr + (n * in_channels + pid_g * in_channels_per_group) * input_width
    )
    weight_base = weight_ptr + (pid_g * in_channels_per_group) * (ocpg * kernel_width)

    accum = tl.zeros((BLOCK_M, BLOCK_OC), dtype=tl.float32)

    for k in range(0, kernel_width):
        num = r + padding_width - k * dilation_width
        active = (num % s) == 0
        if active:
            shift = num // s
            i = m + shift
            i_valid = m_valid & (i >= 0) & (i < input_width)
            for ic in range(0, in_channels_per_group):
                xv = tl.load(
                    input_base + ic * input_width + i,
                    mask=i_valid,
                    other=0.0,
                ).to(tl.float32)
                wv = tl.load(
                    weight_base + ic * (ocpg * kernel_width) + oc * kernel_width + k,
                    mask=oc < ocpg,
                    other=0.0,
                ).to(tl.float32)
                if tf32_mode:
                    # Match torch_musa conv_transpose1d: operands truncated to
                    # TF32 (zero the low 13 mantissa bits), fp32 accumulation.
                    xv = ((xv.to(tl.int32, bitcast=True) >> 13) << 13).to(
                        tl.float32, bitcast=True
                    )
                    wv = ((wv.to(tl.int32, bitcast=True) >> 13) << 13).to(
                        tl.float32, bitcast=True
                    )
                accum += xv[:, None] * wv[None, :]

    if has_bias:
        bv = tl.load(
            bias_ptr + pid_g * ocpg + oc,
            mask=oc < ocpg,
            other=0.0,
        ).to(tl.float32)
        accum += bv[None, :]

    o = r + m * s
    out_off = (
        output_ptr
        + (n * out_channels + pid_g * ocpg)[:, None] * out_width
        + oc[None, :] * out_width
        + o[:, None]
    )
    out_mask = (oc < ocpg)[None, :] & m_valid[:, None]
    tl.store(out_off, accum, mask=out_mask)


# ---------------------------------------------------------------------------
# Kernel 2: fp16/bf16-native tl.dot path for large in_channels_per_group
# (>= BLOCK_IC).  Same residue-class decomposition; each (k, ic-block) is one
# fp16/bf16 MMA with fp32 accumulation.
# ---------------------------------------------------------------------------
@triton.jit
def conv_transpose1d_dot_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    input_width,
    out_width,
    in_channels_per_group,
    out_channels_per_group,
    M_BLOCKS: tl.constexpr,
    groups: tl.constexpr,
    kernel_width: tl.constexpr,
    stride_width: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_width: tl.constexpr,
    has_bias: tl.constexpr,
    tf32_to_fp16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid_g = tl.program_id(2)

    s = stride_width
    ocpg = out_channels_per_group

    r = pid0 // M_BLOCKS
    mb = pid0 % M_BLOCKS
    m = mb * BLOCK_M + tl.arange(0, BLOCK_M)

    oc_blocks = tl.cdiv(ocpg, BLOCK_OC)
    n = pid1 // oc_blocks
    ocb = pid1 % oc_blocks
    oc = ocb * BLOCK_OC + tl.arange(0, BLOCK_OC)
    ic = tl.arange(0, BLOCK_IC)

    M_r = (out_width - r + s - 1) // s
    m_valid = m < M_r

    in_channels = in_channels_per_group * groups
    out_channels = ocpg * groups

    input_base = (
        input_ptr + (n * in_channels + pid_g * in_channels_per_group) * input_width
    )
    weight_base = weight_ptr + (pid_g * in_channels_per_group) * (ocpg * kernel_width)

    accum = tl.zeros((BLOCK_M, BLOCK_OC), dtype=tl.float32)

    IC_BLOCKS = (in_channels_per_group + BLOCK_IC - 1) // BLOCK_IC
    for k in range(0, kernel_width):
        num = r + padding_width - k * dilation_width
        active = (num % s) == 0
        if active:
            shift = num // s
            i = m + shift
            i_valid = m_valid & (i >= 0) & (i < input_width)
            for icb in range(0, IC_BLOCKS):
                ic_full = icb * BLOCK_IC + ic
                xv = tl.load(
                    input_base + ic_full[None, :] * input_width + i[:, None],
                    mask=i_valid[:, None] & (ic_full < in_channels_per_group)[None, :],
                    other=0.0,
                )
                wv = tl.load(
                    weight_base
                    + ic_full[:, None] * (ocpg * kernel_width)
                    + oc[None, :] * kernel_width
                    + k,
                    mask=(ic_full < in_channels_per_group)[:, None]
                    & (oc < ocpg)[None, :],
                    other=0.0,
                )
                if tf32_to_fp16:
                    # fp32 -> TF32 (zero low 13 mantissa bits) -> fp16 (exact
                    # for the truncated values within fp16 range) -> fp16 MMA
                    # with fp32 accumulation, matching torch_musa numerics.
                    xv = ((xv.to(tl.int32, bitcast=True) >> 13) << 13).to(
                        tl.float32, bitcast=True
                    )
                    wv = ((wv.to(tl.int32, bitcast=True) >> 13) << 13).to(
                        tl.float32, bitcast=True
                    )
                    xv = xv.to(tl.float16)
                    wv = wv.to(tl.float16)
                accum += tl.dot(xv, wv, allow_tf32=False)

    if has_bias:
        bv = tl.load(
            bias_ptr + pid_g * ocpg + oc,
            mask=oc < ocpg,
            other=0.0,
        ).to(tl.float32)
        accum += bv[None, :]

    o = r + m * s
    out_off = (
        output_ptr
        + (n * out_channels + pid_g * ocpg)[:, None] * out_width
        + oc[None, :] * out_width
        + o[:, None]
    )
    out_mask = (oc < ocpg)[None, :] & m_valid[:, None]
    tl.store(out_off, accum, mask=out_mask)


def conv_transpose1d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    logger.debug("GEMS_MTHREADS CONV_TRANSPOSE1D")
    if (
        not isinstance(input, torch.Tensor)
        or input.device.type != "musa"
        or input.dtype not in _SUPPORTED_DTYPES
        or (bias is not None and bias.dtype != input.dtype)
        or weight.dtype != input.dtype
    ):
        return default_conv_transpose1d(
            input, weight, bias, stride, padding, output_padding, groups, dilation
        )
    if isinstance(stride, (list, tuple)):
        stride = stride[0]
    if isinstance(padding, (list, tuple)):
        padding = padding[0]
    if isinstance(output_padding, (list, tuple)):
        output_padding = output_padding[0]
    if isinstance(dilation, (list, tuple)):
        dilation = dilation[0]

    batch_size, in_channels, input_width = input.shape
    in_channels_w, out_channels_per_group, kernel_width = weight.shape

    out_channels = out_channels_per_group * groups
    out_width = (
        (input_width - 1) * stride
        - 2 * padding
        + dilation * (kernel_width - 1)
        + output_padding
        + 1
    )

    input = input.contiguous()
    weight = weight.contiguous()

    out = torch.empty(
        (batch_size, out_channels, out_width),
        device=input.device,
        dtype=input.dtype,
    )

    has_bias = bias is not None
    if has_bias:
        bias_ptr = bias
    else:
        bias_ptr = out  # unused when has_bias is False

    in_channels_per_group = in_channels // groups

    use_dot = (in_channels_per_group * kernel_width >= 32) and (
        out_channels_per_group >= 32
    )

    if use_dot:
        tf32_to_fp16 = input.dtype == torch.float32
        if tf32_to_fp16:
            # fp32: TF32-truncated operands are exactly representable in fp16,
            # so fp16 MMA reproduces the torch_musa reference bitwise.
            # BLOCK_M=32 is faster for narrower ocpg (probe: fp32 w0
            # 0.054 -> 0.039ms at ocpg=64), BLOCK_M=64 for wider ocpg.
            BLOCK_M = 32 if out_channels_per_group <= 64 else 64
            BLOCK_IC = 16
            BLOCK_OC = 32
            num_warps = 2
            num_stages = 3
        elif (in_channels_per_group >= 32) and (out_channels_per_group >= 96):
            # channel-heavy wide-oc workloads (e.g. 48x128 K=5 s=2);
            # num_stages=2 pipelines the gather loads much better (probe:
            # 0.091 vs 0.131ms at the default 3 stages).
            BLOCK_M = 128
            BLOCK_IC = 16
            BLOCK_OC = 64
            num_warps = 8
            num_stages = 2
        elif in_channels_per_group >= 32:
            # channel-heavy narrow-oc workloads (e.g. 64x64 K=3)
            BLOCK_M = 64
            BLOCK_IC = 16
            BLOCK_OC = 32
            num_warps = 4
            num_stages = 3
        else:
            # small-ic workloads (e.g. 24x96 K=7): IC=8 dots with 8 warps
            # are fastest (probe: 0.059 vs 0.083ms at 4 warps)
            BLOCK_M = 128
            BLOCK_IC = 8
            BLOCK_OC = 32
            num_warps = 8
            num_stages = 3
        M_BLOCKS = triton.cdiv(triton.cdiv(out_width, stride), BLOCK_M)
        grid = (
            stride * M_BLOCKS,
            batch_size * triton.cdiv(out_channels_per_group, BLOCK_OC),
            groups,
        )
        conv_transpose1d_dot_kernel[grid](
            input,
            weight,
            bias_ptr,
            out,
            batch_size,
            input_width,
            out_width,
            in_channels_per_group,
            out_channels_per_group,
            M_BLOCKS=M_BLOCKS,
            groups=groups,
            kernel_width=kernel_width,
            stride_width=stride,
            padding_width=padding,
            dilation_width=dilation,
            has_bias=has_bias,
            tf32_to_fp16=tf32_to_fp16,
            BLOCK_M=BLOCK_M,
            BLOCK_IC=BLOCK_IC,
            BLOCK_OC=BLOCK_OC,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        BLOCK_M = 128
        # Small ocpg workloads waste half the OC lanes at BLOCK_OC=32;
        # BLOCK_OC=16 measured 20-40% faster on w3/w4.
        BLOCK_OC = 16 if out_channels_per_group <= 16 else 32
        M_BLOCKS = triton.cdiv(triton.cdiv(out_width, stride), BLOCK_M)
        grid = (
            stride * M_BLOCKS,
            batch_size * triton.cdiv(out_channels_per_group, BLOCK_OC),
            groups,
        )
        conv_transpose1d_fma_kernel[grid](
            input,
            weight,
            bias_ptr,
            out,
            batch_size,
            input_width,
            out_width,
            in_channels_per_group,
            out_channels_per_group,
            M_BLOCKS=M_BLOCKS,
            groups=groups,
            kernel_width=kernel_width,
            stride_width=stride,
            padding_width=padding,
            dilation_width=dilation,
            has_bias=has_bias,
            tf32_mode=(input.dtype == torch.float32),
            BLOCK_M=BLOCK_M,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
        )
    return out
