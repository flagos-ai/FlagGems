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
# variant mirrors the proven conv_transpose1d/2d residue kernels instead:
# the grid is split by the (stride_d, stride_h, stride_w) residue phase of
# the output position, so within a subgrid the input index is affine in the
# compact output index and load masks are pure range checks.  Kernel taps
# are unrolled with tl.static_range (the documented P06 bypass for loop
# carried fp32 dot accumulators) and filtered per subgrid by a runtime
# residue comparison.  fp16/bf16 inputs are upcast to fp32 in kernel before
# the ieee dot and rounded once at the store, which is deterministic on
# this backend (raw low-precision dots intermittently return garbage).
_DOT_PRECISION = tl.constexpr("ieee")


def conv_transpose3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
) -> int:
    """
    Determines the output size along one spatial dimension of a 3D transposed
    convolution operation.
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
def conv_transpose3d_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    input_depth: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    out_channels: tl.constexpr,
    out_depth: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    kernel_depth: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    out_channels_per_group: tl.constexpr,
    in_channels_per_group: tl.constexpr,
    stride_depth: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_depth: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_depth: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    has_bias: tl.constexpr,
    n_subgrids: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """
    Triton kernel for 3D transposed convolution forward pass (residue form).

    - input has shape (N, in_channels, iD, iH, iW)
    - weight has shape (in_channels, out_channels/groups, kD, kH, kW)
    - output has shape (N, out_channels, oD, oH, oW)

    Output position o along one axis receives contributions from input
    position i = (o + padding - k * dilation) / stride, which requires
    (o + padding - k * dilation) to be divisible by stride.  Instead of
    testing divisibility per element (the pattern that breaks the GCU
    compiler pipeline), the grid is split by the stride residue phase
    r = o % stride on each axis; within a subgrid only taps k with
    (k * dilation) % stride == (r + padding) % stride contribute, and the
    input index becomes affine in the compact output index.
    """
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_phase_group = tl.program_id(2)

    pid_subgrid = pid_phase_group % n_subgrids
    group = pid_phase_group // n_subgrids

    compact_depth: tl.constexpr = (out_depth + stride_depth - 1) // stride_depth
    compact_height: tl.constexpr = (out_height + stride_height - 1) // stride_height
    compact_width: tl.constexpr = (out_width + stride_width - 1) // stride_width

    residue_width = pid_subgrid % stride_width
    residue_height = (pid_subgrid // stride_width) % stride_height
    residue_depth = pid_subgrid // (stride_width * stride_height)

    # Decompose the flattened (batch, compact_d, compact_h, compact_w) index.
    compact_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    compact_n_dh = compact_offsets // compact_width
    compact_w = compact_offsets % compact_width
    compact_n_d = compact_n_dh // compact_height
    compact_h = compact_n_dh % compact_height
    n = compact_n_d // compact_depth
    compact_d = compact_n_d % compact_depth

    od = compact_d * stride_depth + residue_depth
    oh = compact_h * stride_height + residue_height
    ow = compact_w * stride_width + residue_width

    co_in_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_offsets = group * out_channels_per_group + co_in_offsets

    accum = tl.zeros((BLOCK_N, BLOCK_CO), dtype=tl.float32)
    if has_bias:
        bias_values = tl.load(
            bias_pointer + co_offsets,
            mask=co_in_offsets < out_channels_per_group,
            other=0.0,
        ).to(tl.float32)
        accum += bias_values[None, :]

    depth_residue = (residue_depth + padding_depth) % stride_depth
    height_residue = (residue_height + padding_height) % stride_height
    width_residue = (residue_width + padding_width) % stride_width

    ci_blocks: tl.constexpr = (in_channels_per_group + BLOCK_CI - 1) // BLOCK_CI
    for kd in tl.static_range(0, kernel_depth):
        if (kd * dilation_depth) % stride_depth == depth_residue:
            id_unstrided = od + padding_depth - kd * dilation_depth
            in_d = id_unstrided // stride_depth
            valid_d = (id_unstrided >= 0) & (in_d < input_depth)
            for kh in tl.static_range(0, kernel_height):
                if (kh * dilation_height) % stride_height == height_residue:
                    ih_unstrided = oh + padding_height - kh * dilation_height
                    in_h = ih_unstrided // stride_height
                    valid_dh = valid_d & (ih_unstrided >= 0) & (in_h < input_height)
                    for kw in tl.static_range(0, kernel_width):
                        if (kw * dilation_width) % stride_width == width_residue:
                            iw_unstrided = ow + padding_width - kw * dilation_width
                            in_w = iw_unstrided // stride_width
                            valid_dhw = (
                                (n < batch_size)
                                & valid_dh
                                & (iw_unstrided >= 0)
                                & (in_w < input_width)
                                & (od < out_depth)
                                & (oh < out_height)
                                & (ow < out_width)
                            )
                            for ci_base in tl.static_range(0, ci_blocks):
                                ci_in_offsets = ci_base * BLOCK_CI + tl.arange(
                                    0, BLOCK_CI
                                )
                                ci_offsets = (
                                    group * in_channels_per_group + ci_in_offsets
                                )
                                input_offsets = (
                                    (n[:, None] * in_channels + ci_offsets[None, :])
                                    * (input_depth * input_height * input_width)
                                    + in_d[:, None] * (input_height * input_width)
                                    + in_h[:, None] * input_width
                                    + in_w[:, None]
                                )
                                weight_offsets = (
                                    (
                                        ci_offsets[:, None] * out_channels_per_group
                                        + co_in_offsets[None, :]
                                    )
                                    * (kernel_depth * kernel_height * kernel_width)
                                    + kd * (kernel_height * kernel_width)
                                    + kh * kernel_width
                                    + kw
                                )
                                input_mask = valid_dhw[:, None] & (
                                    ci_in_offsets[None, :] < in_channels_per_group
                                )
                                weight_mask = (
                                    ci_in_offsets[:, None] < in_channels_per_group
                                ) & (co_in_offsets[None, :] < out_channels_per_group)
                                input_block = tl.load(
                                    input_pointer + input_offsets,
                                    mask=input_mask,
                                    other=0.0,
                                ).to(tl.float32)
                                weight_block = tl.load(
                                    weight_pointer + weight_offsets,
                                    mask=weight_mask,
                                    other=0.0,
                                ).to(tl.float32)
                                accum += tl.dot(
                                    input_block,
                                    weight_block,
                                    input_precision=_DOT_PRECISION,
                                )

    output_offsets = (
        (n[:, None] * out_channels + co_offsets[None, :])
        * (out_depth * out_height * out_width)
        + od[:, None] * (out_height * out_width)
        + oh[:, None] * out_width
        + ow[:, None]
    )
    output_mask = (
        (n[:, None] < batch_size)
        & (od[:, None] < out_depth)
        & (oh[:, None] < out_height)
        & (ow[:, None] < out_width)
        & (co_in_offsets[None, :] < out_channels_per_group)
    )
    tl.store(
        output_pointer + output_offsets,
        accum.to(output_pointer.dtype.element_ty),
        mask=output_mask,
    )


def _triple(value):
    if isinstance(value, (list, tuple)):
        assert len(value) == 3, "expected a single int or a tuple of three ints"
        return int(value[0]), int(value[1]), int(value[2])
    return int(value), int(value), int(value)


def conv_transpose3d(
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
    Applies a 3D transposed convolution operator over an input volume.

    Args:
        input: Input tensor of shape (N, in_channels, iD, iH, iW)
        weight: Filters of shape (in_channels, out_channels/groups, kD, kH, kW)
        bias: Optional bias of shape (out_channels). Default: None
        stride: Stride of the convolution. Default: 1
        padding: Zero-padding added to both sides. Default: 0
        output_padding: Additional size added to output shape. Default: 0
        groups: Number of blocked connections. Default: 1
        dilation: Spacing between kernel elements. Default: 1

    Returns:
        Output tensor of shape (N, out_channels, oD, oH, oW)
    """
    logger.debug("GEMS CONV_TRANSPOSE3D")

    assert input.ndim == 5, f"Input must be 5D, received shape {input.shape}"
    assert weight.ndim == 5, f"Weights must be 5D, received shape {weight.shape}"
    assert (
        bias is None or bias.ndim == 1
    ), f"Bias must be 1D, received shape {bias.shape}"

    stride_depth, stride_height, stride_width = _triple(stride)
    padding_depth, padding_height, padding_width = _triple(padding)
    (
        output_padding_depth,
        output_padding_height,
        output_padding_width,
    ) = _triple(output_padding)
    dilation_depth, dilation_height, dilation_width = _triple(dilation)

    batch_size, in_channels, input_depth, input_height, input_width = input.shape
    (
        in_channels_weight,
        out_channels_per_group,
        kernel_depth,
        kernel_height,
        kernel_width,
    ) = weight.shape

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

    out_depth = conv_transpose3d_output_size(
        input_depth,
        kernel_depth,
        stride_depth,
        padding_depth,
        output_padding_depth,
        dilation_depth,
    )
    out_height = conv_transpose3d_output_size(
        input_height,
        kernel_height,
        stride_height,
        padding_height,
        output_padding_height,
        dilation_height,
    )
    out_width = conv_transpose3d_output_size(
        input_width,
        kernel_width,
        stride_width,
        padding_width,
        output_padding_width,
        dilation_width,
    )

    output_dtype = input.dtype
    output = torch.empty(
        (batch_size, out_channels, out_depth, out_height, out_width),
        device=input.device,
        dtype=output_dtype,
    )
    if output.numel() == 0:
        return output

    input_contig = input.contiguous()
    weight_contig = weight.contiguous()

    in_channels_per_group = in_channels // groups

    # Tile heuristic mirroring the conv_transpose1d/2d residue paths on this
    # backend: fp32 prefers narrower channel tiles; small channel counts
    # clamp BLOCK_CI/BLOCK_CO to 16 (tl.dot minimum).
    block_n = 64
    block_ci = 32
    block_co = 32
    if input.dtype is torch.float32:
        block_ci = 16
        block_co = 16
    if in_channels_per_group <= 16:
        block_ci = 16
    if out_channels_per_group <= 16:
        block_co = 16

    compact_depth = triton.cdiv(out_depth, stride_depth)
    compact_height = triton.cdiv(out_height, stride_height)
    compact_width = triton.cdiv(out_width, stride_width)
    compact_total = compact_depth * compact_height * compact_width
    n_subgrids = stride_depth * stride_height * stride_width

    # Grid: (batch * compact-volume tiles, output-channel tiles, groups *
    # stride-residue subgrids).
    grid = (
        triton.cdiv(batch_size * compact_total, block_n),
        triton.cdiv(out_channels_per_group, block_co),
        groups * n_subgrids,
    )
    bias_pointer = bias if bias is not None else input_contig

    conv_transpose3d_forward_kernel[grid](
        input_contig,
        weight_contig,
        bias_pointer,
        output,
        batch_size,
        in_channels,
        input_depth,
        input_height,
        input_width,
        out_channels,
        out_depth,
        out_height,
        out_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        out_channels_per_group,
        in_channels_per_group,
        stride_depth,
        stride_height,
        stride_width,
        padding_depth,
        padding_height,
        padding_width,
        dilation_depth,
        dilation_height,
        dilation_width,
        bias is not None,
        n_subgrids,
        BLOCK_N=block_n,
        BLOCK_CI=block_ci,
        BLOCK_CO=block_co,
        num_warps=4,
        num_stages=2,
    )

    return output
