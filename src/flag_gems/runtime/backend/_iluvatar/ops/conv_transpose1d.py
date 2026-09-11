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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


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
@triton.autotune(
    configs=runtime.get_tuned_config("conv_transpose1d"),
    key=[
        "batch_size",
        "in_channels",
        "input_width",
        "out_channels",
        "out_width",
        "kernel_width",
        "stride_width",
        "padding_width",
        "groups",
    ],
)
@triton.jit
def conv_transpose1d_gemm_kernel(
    input_pointer,
    weight_pointer,
    cols_pointer,
    batch_size,
    input_width,
    out_channels,
    out_width,
    input_n_stride,
    input_c_stride,
    input_w_stride,
    weight_ic_stride,
    weight_oc_stride,
    weight_w_stride,
    cols_b_stride,
    cols_g_stride,
    cols_m_stride,
    cols_w_stride,
    in_channels: tl.constexpr,
    out_channels_per_group: tl.constexpr,
    kernel_width: tl.constexpr,
    stride_width: tl.constexpr,
    padding_width: tl.constexpr,
    groups: tl.constexpr,
    M: tl.constexpr,
    BLOCK_N_OW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    """
    Dense GEMM stage of the col2im transposed convolution (PyTorch-style).

    Computes the "columns" tensor
        cols[b, g, m, iw] = sum_ic weight[g, ic, oc, k] * input[b, g, ic, iw]
    where m = oc * kernel_width + k, i.e. the kernel taps are folded into the
    GEMM's M dimension (OC_pg * KW) rather than the reduction axis. The result is
    a clean dense matmul  A^T[M, IC] . B[IC, IW]  per (batch, group) with no
    gather and no divisibility masking -- all stride/padding bookkeeping is
    deferred to the col2im stage.

    Tile mapping (uses the conv_transpose1d autotune space in tune_configs.yaml):
      BLOCK_OC   -> block over M = OC_pg * KW
      BLOCK_N_OW -> block over N = input_width
      BLOCK_IC   -> block over K = in_channels_per_group
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bg = tl.program_id(2)

    batch_idx = pid_bg // groups
    pid_group = pid_bg % groups

    m_offset = pid_m * BLOCK_OC + tl.arange(0, BLOCK_OC)
    n_offset = pid_n * BLOCK_N_OW + tl.arange(0, BLOCK_N_OW)

    # Decode M index into (out channel, kernel tap).
    oc_idx = m_offset // kernel_width
    k_idx = m_offset % kernel_width

    in_channels_per_group = in_channels
    accum = tl.zeros((BLOCK_OC, BLOCK_N_OW), dtype=tl.float32)

    input_base = (
        input_pointer
        + input_n_stride * batch_idx
        + input_c_stride * pid_group * in_channels_per_group
    )
    weight_base = weight_pointer + weight_ic_stride * pid_group * in_channels_per_group

    m_mask = m_offset < M
    n_mask = n_offset < input_width

    BLOCK_IC_COUNT = (in_channels_per_group + BLOCK_IC - 1) // BLOCK_IC
    for ic_block in range(BLOCK_IC_COUNT):
        ic_offset = ic_block * BLOCK_IC + tl.arange(0, BLOCK_IC)
        ic_mask = ic_offset < in_channels_per_group

        # A^T tile [BLOCK_OC(M), BLOCK_IC]: weight[ic, oc, k] indexed by (m, ic).
        weight_ptr = (
            weight_base
            + (weight_oc_stride * oc_idx + weight_w_stride * k_idx)[:, None]
            + (weight_ic_stride * ic_offset)[None, :]
        )
        weight_tile = tl.load(
            weight_ptr, mask=m_mask[:, None] & ic_mask[None, :], other=0.0
        )

        # B tile [BLOCK_IC, BLOCK_N_OW]: input[b, ic, iw] indexed by (ic, n).
        input_ptr = (
            input_base
            + (input_c_stride * ic_offset)[:, None]
            + (input_w_stride * n_offset)[None, :]
        )
        input_tile = tl.load(
            input_ptr, mask=ic_mask[:, None] & n_mask[None, :], other=0.0
        )

        # allow_tf32=True lets fp32 inputs use the TCU (the CoreX backend keeps
        # enough precision to pass the fp32 tolerance); fp16/bf16 ignore this flag.
        accum += tl.dot(weight_tile, input_tile, allow_tf32=True)

    cols_ptr = (
        cols_pointer
        + cols_b_stride * batch_idx
        + cols_g_stride * pid_group
        + (cols_m_stride * m_offset)[:, None]
        + (cols_w_stride * n_offset)[None, :]
    )
    tl.store(cols_ptr, accum, mask=m_mask[:, None] & n_mask[None, :])


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 256}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_W": 2048}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_W": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_W": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 1024}, num_warps=4, num_stages=2),
    ],
    key=["out_width", "kernel_width", "stride_width", "dilation_width"],
)
@triton.jit
def conv_transpose1d_col2im_kernel(
    cols_pointer,
    output_pointer,
    bias_pointer,
    out_channels,
    out_width,
    input_width,
    cols_b_stride,
    cols_g_stride,
    cols_m_stride,
    cols_w_stride,
    output_n_stride,
    output_c_stride,
    output_w_stride,
    out_channels_per_group: tl.constexpr,
    kernel_width: tl.constexpr,
    stride_width: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_width: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    col2im stage: scatter-free reduction of the GEMM columns into the output.

    For each output position ow, out[b, oc, ow] sums the kernel taps that land on
    it: out_w = iw * stride - padding + k * dilation, so the contributing column
    is cols[b, g, oc*KW + k, iw] with iw = (ow + padding - k*dilation) / stride
    (only when divisible and in range). Because each (b, oc, ow) is owned by a
    single program the reduction is a plain accumulate -- no atomics needed.
    """
    pid_boc = tl.program_id(0)
    pid_w = tl.program_id(1)

    batch_idx = pid_boc // out_channels
    oc_global = pid_boc % out_channels
    pid_group = oc_global // out_channels_per_group
    oc_idx = oc_global % out_channels_per_group

    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    ow_mask = ow < out_width

    accum = tl.zeros((BLOCK_W,), dtype=tl.float32)
    cols_base = cols_pointer + cols_b_stride * batch_idx + cols_g_stride * pid_group

    for k in range(kernel_width):
        numerator = ow + padding_width - k * dilation_width
        iw = numerator // stride_width
        valid = (
            ow_mask & (numerator % stride_width == 0) & (iw >= 0) & (iw < input_width)
        )
        m = oc_idx * kernel_width + k
        cols_ptr = cols_base + cols_m_stride * m + cols_w_stride * iw
        accum += tl.load(cols_ptr, mask=valid, other=0.0)

    bias = tl.load(bias_pointer + oc_global).to(tl.float32)
    accum += bias

    out_ptr = (
        output_pointer
        + output_n_stride * batch_idx
        + output_c_stride * oc_global
        + output_w_stride * ow
    )
    tl.store(out_ptr, accum, mask=ow_mask)


def _conv_transpose1d_gemm_col2im(
    input_contig,
    weight_contig,
    output,
    bias_pointer,
    batch_size,
    in_channels_per_group,
    input_width,
    out_channels,
    out_channels_per_group,
    out_width,
    kernel_width,
    stride_width,
    padding_width,
    dilation_width,
    groups,
):
    """PyTorch-style dense GEMM + col2im path (handles general stride/dilation)."""
    m_size = out_channels_per_group * kernel_width

    # The op is memory bound here, so for half-precision inputs keep the intermediate
    # columns in the input dtype to halve the GEMM-write / col2im-read bandwidth (the
    # GEMM still accumulates in fp32). Grouped shapes have a tiny per-group reduction
    # and thus a very tight tolerance, where the extra fp16/bf16 rounding of cols fails
    # accuracy, so they fall back to fp32 columns.
    use_half_cols = groups == 1 and input_contig.dtype in (
        torch.float16,
        torch.bfloat16,
    )
    cols_dtype = input_contig.dtype if use_half_cols else torch.float32
    cols = torch.empty(
        (batch_size, groups, m_size, input_width),
        device=input_contig.device,
        dtype=cols_dtype,
    )

    gemm_grid = lambda META: (
        triton.cdiv(m_size, META["BLOCK_OC"]),
        triton.cdiv(input_width, META["BLOCK_N_OW"]),
        batch_size * groups,
    )
    conv_transpose1d_gemm_kernel[gemm_grid](
        input_contig,
        weight_contig,
        cols,
        batch_size,
        input_width,
        out_channels,
        out_width,
        *input_contig.stride(),
        *weight_contig.stride(),
        *cols.stride(),
        in_channels_per_group,
        out_channels_per_group,
        kernel_width,
        stride_width,
        padding_width,
        M=m_size,
        groups=groups,
    )

    col2im_grid = lambda META: (
        batch_size * out_channels,
        triton.cdiv(out_width, META["BLOCK_W"]),
    )
    conv_transpose1d_col2im_kernel[col2im_grid](
        cols,
        output,
        bias_pointer,
        out_channels,
        out_width,
        input_width,
        *cols.stride(),
        *output.stride(),
        out_channels_per_group,
        kernel_width,
        stride_width,
        padding_width,
        dilation_width,
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
    logger.debug("GEMS_ILUVATAR CONV_TRANSPOSE1D")

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

    # Create bias pointer (zeros if no bias)
    if bias is None:
        bias_pointer = torch.zeros(
            out_channels, device=input.device, dtype=output_dtype
        )
    else:
        bias_pointer = bias

    # Ensure contiguous tensors
    input_contig = input.contiguous()
    weight_contig = weight.contiguous()

    in_channels_per_group = in_channels // groups

    # PyTorch-style dense GEMM + col2im (the gemm -> columns -> col2im pattern used
    # by the slow/naive transposed-convolution path), used for every
    # stride/dilation/dtype. The kernel taps are folded into the GEMM's M dimension
    # (OC_pg * KW) rather than the reduction axis, giving a clean dense matmul with no
    # gather and no divisibility waste; col2im then reduces the columns into the
    # strided output (output-owned, so scatter-free).
    _conv_transpose1d_gemm_col2im(
        input_contig,
        weight_contig,
        output,
        bias_pointer,
        batch_size,
        in_channels_per_group,
        input_width,
        out_channels,
        out_channels_per_group,
        out_width,
        kernel_width,
        stride_width,
        padding_width,
        dilation_width,
        groups,
    )

    return output
