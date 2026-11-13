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

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# The generic gather-style kernel (per-element div/mod input indices plus a
# divisibility mask feeding tl.dot) fails to compile on this backend with
# "Pipeline run failed: PassManager execution failed" (P02 family).  This
# variant mirrors the proven _conv_transpose2d_residue_kernel instead: output
# positions are decomposed by stride residue so the input index is affine and
# load masks are pure range checks.  fp16/bf16 inputs are upcast to fp32 in
# kernel before the ieee dot and rounded once at the store, which is
# deterministic on this backend (raw low-precision dots intermittently return
# garbage).
_DOT_PRECISION = tl.constexpr("ieee")


def conv_transpose1d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
) -> int:
    """
    Determines the output size of a 1D transposed convolution operation.

    Args:
        in_size: Input size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.
        output_padding: Output padding.
        dilation: Dilation.

    Returns:
        Output size of 1D transposed convolution.
    """
    return (
        (in_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


@libentry()
@triton.jit
def conv_transpose1d_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    batch_size: tl.constexpr,
    input_channels: tl.constexpr,
    input_width: tl.constexpr,
    output_channels: tl.constexpr,
    output_width: tl.constexpr,
    weight_width: tl.constexpr,
    output_channels_per_group: tl.constexpr,
    input_channels_per_group: tl.constexpr,
    stride_width: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_width: tl.constexpr,
    has_bias: tl.constexpr,
    n_subgrids: tl.constexpr,
    BLOCK_NC: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """
    Triton kernel for 1D transposed convolution forward pass (residue form).

    For transposed convolution:
    - input has shape (N, in_channels, in_width)
    - weight has shape (in_channels, out_channels/groups, kernel_width)
    - output has shape (N, out_channels, out_width)

    Output position ow receives contributions from input position
    iw = (ow + padding - kw * dilation) / stride, which requires
    (ow + padding - kw * dilation) to be divisible by stride.  Instead of
    testing divisibility per element (the pattern that breaks the GCU
    compiler pipeline), the grid is split by the stride residue
    r = ow % stride; within a subgrid only taps kw with
    (kw * dilation) % stride == (r + padding) % stride contribute, and the
    input index becomes affine in the compact output index.
    """
    pid_nc = tl.program_id(0)
    pid_co_in_group = tl.program_id(1)
    pid_phase_group = tl.program_id(2)

    pid_subgrid = pid_phase_group % n_subgrids
    group = pid_phase_group // n_subgrids
    compact_width: tl.constexpr = (output_width + stride_width - 1) // stride_width

    compact_offsets = pid_nc * BLOCK_NC + tl.arange(0, BLOCK_NC)
    n = compact_offsets // compact_width
    compact_w = compact_offsets % compact_width
    ow = compact_w * stride_width + pid_subgrid

    co_in_offsets = pid_co_in_group * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_offsets = group * output_channels_per_group + co_in_offsets

    accum = tl.zeros((BLOCK_NC, BLOCK_CO), dtype=tl.float32)
    if has_bias:
        bias_values = tl.load(
            bias_pointer + co_offsets,
            mask=co_in_offsets < output_channels_per_group,
            other=0.0,
        ).to(tl.float32)
        accum += bias_values[None, :]

    ci_blocks: tl.constexpr = tl.cdiv(input_channels_per_group, BLOCK_CI)
    width_residue = (pid_subgrid + padding_width) % stride_width
    for kw in tl.static_range(0, weight_width):
        kw_residue = (kw * dilation_width) % stride_width
        if kw_residue == width_residue:
            iw_unstrided = ow + padding_width - kw * dilation_width
            iw = iw_unstrided // stride_width
            valid_w = (
                (n < batch_size)
                & (iw_unstrided >= 0)
                & (iw < input_width)
                & (ow < output_width)
            )
            for ci_base in range(ci_blocks):
                ci_in_offsets = ci_base * BLOCK_CI + tl.arange(0, BLOCK_CI)
                ci_offsets = group * input_channels_per_group + ci_in_offsets
                input_offsets = (
                    n[:, None] * input_channels + ci_offsets[None, :]
                ) * input_width + iw[:, None]
                weight_offsets = (
                    ci_offsets[:, None] * output_channels_per_group
                    + co_in_offsets[None, :]
                ) * weight_width + kw
                input_mask = valid_w[:, None] & (
                    ci_in_offsets[None, :] < input_channels_per_group
                )
                weight_mask = (ci_in_offsets[:, None] < input_channels_per_group) & (
                    co_in_offsets[None, :] < output_channels_per_group
                )
                input_block = tl.load(
                    input_pointer + input_offsets, mask=input_mask, other=0.0
                ).to(tl.float32)
                weight_block = tl.load(
                    weight_pointer + weight_offsets, mask=weight_mask, other=0.0
                ).to(tl.float32)
                accum += tl.dot(
                    input_block, weight_block, input_precision=_DOT_PRECISION
                )

    output_offsets = (
        n[:, None] * output_channels + co_offsets[None, :]
    ) * output_width + ow[:, None]
    output_mask = (
        (n[:, None] < batch_size)
        & (ow[:, None] < output_width)
        & (co_in_offsets[None, :] < output_channels_per_group)
        & (co_offsets[None, :] < output_channels)
    )
    tl.store(
        output_pointer + output_offsets,
        accum.to(output_pointer.dtype.element_ty),
        mask=output_mask,
    )


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
    """
    Applies a 1D transposed convolution operator over an input signal.

    Args:
        input: Input tensor of shape (N, in_channels, L_in)
        weight: Filters of shape (in_channels, out_channels/groups, kernel_width)
        bias: Optional bias of shape (out_channels). Default: None
        stride: Stride of the convolution. Default: 1
        padding: Zero-padding added to both sides. Default: 0
        output_padding: Additional size added to output shape. Default: 0
        groups: Number of blocked connections. Default: 1
        dilation: Spacing between kernel elements. Default: 1

    Returns:
        Output tensor of shape (N, out_channels, L_out)
    """
    logger.debug("GEMS CONV_TRANSPOSE1D")

    assert input.ndim == 3, f"Input must be 3D, received shape {input.shape}"
    assert weight.ndim == 3, f"Weights must be 3D, received shape {weight.shape}"
    assert (
        bias is None or bias.ndim == 1
    ), f"Bias must be 1D, received shape {bias.shape}"

    # Parse stride, padding, output_padding, dilation
    if isinstance(stride, (list, tuple)):
        stride_width = stride[0]
    else:
        stride_width = stride

    if isinstance(padding, (list, tuple)):
        padding_width = padding[0]
    else:
        padding_width = padding

    if isinstance(output_padding, (list, tuple)):
        output_padding_width = output_padding[0]
    else:
        output_padding_width = output_padding

    if isinstance(dilation, (list, tuple)):
        dilation_width = dilation[0]
    else:
        dilation_width = dilation

    batch_size, in_channels, input_width = input.shape
    in_channels_weight, out_channels_per_group, kernel_width = weight.shape

    assert (
        in_channels == in_channels_weight
    ), f"Input channels ({in_channels}) must match weight in_channels ({in_channels_weight})"
    assert (
        in_channels % groups == 0
    ), f"in_channels ({in_channels}) must be divisible by groups ({groups})"

    out_channels = out_channels_per_group * groups

    assert (
        bias is None or bias.shape[0] == out_channels
    ), f"Bias shape ({bias.shape}) doesn't match out_channels ({out_channels})"

    # Calculate output size
    out_width = conv_transpose1d_output_size(
        input_width,
        kernel_width,
        stride_width,
        padding_width,
        output_padding_width,
        dilation_width,
    )

    # Allocate output
    output_dtype = input.dtype
    output = torch.empty(
        (batch_size, out_channels, out_width),
        device=input.device,
        dtype=output_dtype,
    )
    if output.numel() == 0:
        return output

    # Ensure contiguous tensors
    input_contig = input.contiguous()
    weight_contig = weight.contiguous()

    in_channels_per_group = in_channels // groups

    # Tile heuristic mirroring the conv_transpose2d residue path on this
    # backend: fp32 prefers narrower channel tiles; small channel counts
    # clamp BLOCK_CI/BLOCK_CO to 16 (tl.dot minimum).
    block_nc = 64
    block_ci = 32
    block_co = 32
    if input.dtype is torch.float32:
        block_ci = 16
        block_co = 16
    if in_channels_per_group <= 16:
        block_ci = 16
    if out_channels_per_group <= 16:
        block_co = 16

    # Grid: (batch * compact-width tiles, output-channel tiles, groups * stride)
    compact_width = triton.cdiv(out_width, stride_width)
    grid = (
        triton.cdiv(batch_size * compact_width, block_nc),
        triton.cdiv(out_channels_per_group, block_co),
        groups * stride_width,
    )
    bias_pointer = bias if bias is not None else input_contig

    conv_transpose1d_forward_kernel[grid](
        input_contig,
        weight_contig,
        bias_pointer,
        output,
        batch_size,
        in_channels,
        input_width,
        out_channels,
        out_width,
        kernel_width,
        out_channels_per_group,
        in_channels_per_group,
        stride_width,
        padding_width,
        dilation_width,
        bias is not None,
        stride_width,
        BLOCK_NC=block_nc,
        BLOCK_CI=block_ci,
        BLOCK_CO=block_co,
        num_warps=4,
        num_stages=2,
    )

    return output
