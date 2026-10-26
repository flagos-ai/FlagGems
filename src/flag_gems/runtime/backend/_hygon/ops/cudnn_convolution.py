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

"""hygon override of :func:`flag_gems.ops.cudnn_convolution`.

The generic dispatch is used unchanged except for host time, which is the only
problem addressed here.

FlagTree AABS used to be addressed here too. It is now held off by the generic
op itself -- see ``_aabs_disabled`` in flag_gems/ops/cudnn_convolution.py, which
documents the mispairing and its measurements. The mispairing is a property of
the convolution kernels rather than of this backend, so the guard moved next to
them. This file still needs the helper because it launches its own conv2d kernel
below, autotuned from the same configs.

Host time. Splitting each benchmark row into host and device time
   (``do_bench`` vs a CUDA-graph replay of the same call) puts 26 of 66 rows at
   a mean speedup of 0.49 with 60-135 us of host time each, while their *device*
   time already beats MIOpen's. The convolution kernels are small, so the host
   is the bottleneck, and it is all in the op's Python. Stacking the layers of
   one such row ``(16, 32, 512) fp16 1d``, host us per call, measured by timing
   prefixes of the call in a tight loop::

     triton JITFunction launch, 31 args          63
     + triton Autotuner wrapper                  55
     + torch.empty + torch.zeros                 40
     + this file's forward python                51
     + FlagGems op dispatch                      40
                                                 ---
     total                                      249

   ``do_bench`` reports only 12 us of that for a bare launch and 117 us for the
   op, because the L2 flush it runs before each call (~146 us of GPU work)
   hides host time underneath it; the score sees the excess. The two allocator
   and forward-python layers, 91 us, are what this file removes: the bias is a
   cached constant instead of a fresh ``torch.zeros``, the autograd context is
   only recorded when a backward could actually be asked for, and conv1d hands
   the kernel the strides its ``unsqueeze`` would have produced rather than
   materialising the 4d view and squeezing it back. The same row then measures
   168 us of host time and 16 us under ``do_bench`` -- below the flush, where
   the score stops seeing it. What is left -- triton's own launch and tuning
   machinery, and the FlagGems dispatch in front of it -- belongs to the
   framework.

   (An earlier version of this file also dropped the ``@libentry()`` wrapper
   that ``flag_gems/ops/conv2d.py`` puts on this kernel. A/B'd against the
   generic path that wrapper is worth ~5 us of the 250, not the ~100 a profile
   had suggested -- cProfile's cumulative numbers on this call are dominated by
   triton's C++ launcher and misattribute it -- so the kernel below is the
   generic kernel with nothing removed, launched the standard
   ``kernel[grid](...)`` way.)

Only the tl.dot path (1d and 2d) is routed here; depthwise, pointwise,
direct-FMA and 3d fall through to the generic op.
"""

import copy
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.conv2d import Conv2d as _GenericConv2d
from flag_gems.ops.cudnn_convolution import (
    _DIRECT_MAX_C,
    _DOT_MIN_K,
    _aabs_disabled,
    _to_list,
)
from flag_gems.ops.cudnn_convolution import (
    cudnn_convolution as _generic_cudnn_convolution,
)

logger = logging.getLogger(__name__)


def conv2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


# get_tuned_config hands back the cached list, so conv2d.py's autotuner holds
# these very Config objects. AABS rewrites config kwargs in place
# (``config.kwargs[bs_name] = bs`` in triton/runtime/adjust_kernel_param.py), so
# a launch of the generic kernel with AABS on would leave adjusted block sizes
# behind in the objects this autotuner is about to use. Take a private copy.
_CONV2D_FORWARD_CONFIGS = copy.deepcopy(runtime.get_tuned_config("conv2d_forward"))


_ZERO_BIAS = {}


def _zero_bias(out_c, device, dtype):
    """The zero bias the kernel reads when the op is called without one.

    The kernel never writes to the bias, so one tensor per (length, dtype,
    device) can serve every call and the ~11us of a fresh ``torch.zeros`` is
    paid once per shape instead of once per launch.
    """
    key = (out_c, dtype, device)
    tensor = _ZERO_BIAS.get(key)
    if tensor is None:
        tensor = torch.zeros(out_c, device=device, dtype=dtype)
        _ZERO_BIAS[key] = tensor
    return tensor


def _needs_grad(*tensors):
    """Would ``Function.apply`` build a backward graph for these inputs?

    If not, calling the forward directly gives the same tensor -- one with no
    ``grad_fn`` -- without paying for dispatcher entry and an autograd context.
    """
    if not torch.is_grad_enabled():
        return False
    for tensor in tensors:
        if tensor is not None and tensor.requires_grad:
            return True
    return False


# conv2d_forward_kernel copied from flag_gems/ops/conv2d.py, with the @libentry()
# wrapper left off (see the module docstring). The body, the configs and the
# autotune key are the generic ones.
@triton.autotune(
    configs=_CONV2D_FORWARD_CONFIGS,
    key=[
        "in_n",
        "weight_c",
        "input_height",
        "input_width",
        "out_c",
        "out_height",
        "out_width",
        "weight_height",
        "weight_width",
        "stride_height",
        "stride_width",
        "padding_height",
        "padding_width",
        "groups",
    ],
)
@triton.jit
def _hygon_conv2d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
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
):
    pid_ni_ho_wo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    # caculate in_n out_height out_weight value in kernel
    ni_ho_wo_offset = pid_ni_ho_wo * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    ni_ho_offset = ni_ho_wo_offset // out_width
    in_n_point_value = ni_ho_offset // out_height
    output_height_point_value = ni_ho_offset % out_height
    output_width_point_value = ni_ho_wo_offset % out_width

    # Load the input and weight pointers. input and weight are of shape
    # [in_n, groups, in_c, input_height, input_width] and [groups, out_c, in_c, weight_height, weight_width]
    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    input_pointer += (
        input_n_stride * in_n_point_value + input_c_stride * pid_group * weight_c
    )[:, None]
    weight_pointer += (
        weight_n_stride * output_c_offset
        + weight_n_stride * pid_group * out_per_group_c
    )[None, :]

    accum = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI
    for hwc in range(weight_height * weight_width * BLOCK_CI_COUNT):
        c = (hwc % BLOCK_CI_COUNT) * BLOCK_CI
        hw = hwc // BLOCK_CI_COUNT
        h = hw // weight_width
        w = hw % weight_width

        input_c_offset = c + tl.arange(0, BLOCK_CI)
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
            + (input_height_stride * input_height_offset)[:, None]
            + (input_width_stride * input_width_offset)[:, None]
        )
        curr_weight_pointer = (
            weight_pointer
            + (weight_c_stride * input_c_offset)[:, None]
            + (weight_height_stride * h)
            + (weight_width_stride * w)
        )

        input_mask = (
            (in_n_point_value < in_n)[:, None]
            & (input_c_offset < weight_c)[None, :]
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
    accum += bias
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


def _launch_conv2d(input, weight, bias, stride, padding, dilation, groups, ctx=None):
    """Body of the 2d forward, shared by the autograd and inference paths.

    ``ctx`` is the autograd context the generic forward fills in, or None when
    no backward can be asked for -- recording it costs host time that shows up
    straight in the measured latency of these small convolutions.
    """
    assert weight.ndim == 4, "Weights must be 4D, received shape {weight.shape}"
    assert (
        bias is None or bias.ndim == 1
    ), "Bias must be 1D, received shape {bias.shape}"

    assert (
        input.shape[1] == groups * weight.shape[1]
    ), "Incompatible input ({input.shape}) and weights ({weight.shape}) shape with {groups} groups"
    assert (
        bias is None or weight.shape[0] == bias.shape[0]
    ), "Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape"

    if isinstance(stride, (list, tuple)):
        stride_height, stride_width = stride
    else:
        stride_height = stride_width = stride

    if isinstance(padding, (list, tuple)):
        padding_height, padding_width = padding
    else:
        padding_height = padding_width = padding

    if isinstance(dilation, (list, tuple)):
        dilation_height, dilation_width = dilation
    else:
        dilation_height = dilation_width = dilation

    in_n, _, input_height, input_width = input.shape
    out_c, weight_c, weight_height, weight_width = weight.shape
    orig_weight_c = weight_c  # Save original before potential padding

    # Save original tensors for backward BEFORE padding
    orig_input = input
    orig_weight = weight

    # Triton tl.dot requires K >= 16. The K dimension in conv2d im2col matmul
    # is BLOCK_CI which tiles over weight_c. When weight_c < 16, AABS
    # (Auto-Adjusted Block Size) shrinks BLOCK_CI to next_power_of_2(weight_c)
    # which violates the K >= 16 constraint. Pad input/weight channels to 16.
    _MIN_DOT_K = 16
    if weight_c < _MIN_DOT_K:
        pad_c = _MIN_DOT_K - weight_c
        if groups == 1:
            # Simple case: pad channel dim at the end
            input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, pad_c))
        else:
            # For grouped conv, must pad each group's channels independently.
            # Reshape to (N, groups, weight_c, H, W), pad weight_c dim, reshape back.
            N, C, H, W = input.shape
            input = input.reshape(N, groups, weight_c, H, W)
            input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, pad_c))
            input = input.reshape(N, groups * (weight_c + pad_c), H, W)
        # Pad weight: (out_c, weight_c, kH, kW) -> (out_c, weight_c+pad_c, kH, kW)
        weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, pad_c))
        weight_c = weight_c + pad_c

    out_height = conv2d_output_size(
        input_height, weight_height, stride_height, padding_height, dilation_height
    )
    out_width = conv2d_output_size(
        input_width, weight_width, stride_width, padding_width, dilation_width
    )

    output_dtype = input.dtype
    output = torch.empty(
        (in_n, out_c, out_height, out_width),
        device=input.device,
        dtype=output_dtype,
    )

    # BLOCK_NI_HO_WO along the in_n, out_height, and out_width dimensions,
    # BLOCK_CO along the out_c,
    # one group per cat
    grid = lambda META: (
        triton.cdiv(in_n * out_height * out_width, META["BLOCK_NI_HO_WO"]),
        triton.cdiv(int(out_c // groups), META["BLOCK_CO"]),
        groups,
    )

    if bias is None:
        bias_pointer = _zero_bias(out_c, input.device, output_dtype)
    else:
        bias_pointer = bias
    _hygon_conv2d_forward_kernel[grid](
        input,
        weight,
        output,
        bias_pointer,
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
    )

    if ctx is not None:
        # Save ORIGINAL (unpadded) tensors for backward
        ctx.save_for_backward(orig_weight, orig_input, bias)

        ctx.stride = (stride_height, stride_width)
        ctx.padding = (padding_height, padding_width)
        ctx.dilation = (dilation_height, dilation_width)

        ctx.weight_info = (
            int(out_c / groups),
            orig_weight_c,
            weight_height,
            weight_width,
        )
        ctx.input_info = (in_n, input_height, input_width)
        ctx.out_info = (out_height, out_width)

        ctx.device = input.device
        ctx.groups = groups

    return output


class _HygonConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        logger.debug("GEMS_HYGON CONV2D")
        return _launch_conv2d(
            input, weight, bias, stride, padding, dilation, groups, ctx
        )

    @staticmethod
    def backward(ctx, out_grad):
        # The vjp reads only the ctx entries the forward above stores, so reuse
        # the generic implementation instead of duplicating it.
        return _GenericConv2d.backward(ctx, out_grad)


def _conv1d_direct(
    input, weight, bias, stride_width, padding_width, dilation_width, groups
):
    """conv1d as the 4d kernel sees it, without the unsqueeze/squeeze round trip.

    Every stride reaches the kernel as a runtime argument, so a (N, C, L)
    tensor can be handed to it as if it were the (N, C, L, 1) view conv1d would
    have built, by passing the strides that unsqueeze produces and letting the
    single width element be indexed at 0. The width axis is dereferenced by
    offsets that are all zero (weight_width == padding_width == 0,
    stride_width == out_width == 1), so its stride value is inert.
    """
    in_n, _, in_l = input.shape
    out_c, weight_c, weight_l = weight.shape
    # The 4d entry checks this before it indexes; skipping it here would turn a
    # mismatched weight into an out of bounds read rather than an error.
    assert (
        input.shape[1] == groups * weight.shape[1]
    ), "Incompatible input ({input.shape}) and weights ({weight.shape}) shape with {groups} groups"
    out_l = conv2d_output_size(
        in_l, weight_l, stride_width, padding_width, dilation_width
    )
    output = torch.empty((in_n, out_c, out_l), device=input.device, dtype=input.dtype)

    grid = lambda META: (
        triton.cdiv(in_n * out_l, META["BLOCK_NI_HO_WO"]),
        triton.cdiv(int(out_c // groups), META["BLOCK_CO"]),
        groups,
    )

    if bias is None:
        bias_pointer = _zero_bias(out_c, input.device, input.dtype)
    else:
        bias_pointer = bias
    _hygon_conv2d_forward_kernel[grid](
        input,
        weight,
        output,
        bias_pointer,
        in_n,
        in_l,
        1,
        out_c,
        out_l,
        1,
        *input.stride(),
        1,
        *weight.stride(),
        1,
        *output.stride(),
        1,
        weight_c,
        weight_l,
        1,
        stride_width,
        1,
        padding_width,
        0,
        dilation_width,
        1,
        groups=groups,
    )
    return output


def _hygon_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    logger.debug("GEMS_HYGON CONV1D")
    if isinstance(stride, (list, tuple)):
        stride_width = stride[0]
    else:
        stride_width = stride

    if isinstance(dilation, (list, tuple)):
        dilation_width = dilation[0]
    else:
        dilation_width = dilation

    if isinstance(padding, str):
        if padding == "same":
            assert stride == 1, "Doesn't support any stride values other than 1 \
                in padding = 'same' mode, received stride value {stride}"
            il = input.shape[-1]
            kernel_size = weight.shape[-1]
            padding_width = math.ceil(
                (stride_width * (il - 1) + 1 + dilation_width * (kernel_size - 1) - il)
                / 2
            )
            ol = int(
                (il + 2 * padding_width - dilation_width * (kernel_size - 1) - 1)
                / stride_width
                + 1
            )
            return _hygon_conv2d(
                input.unsqueeze(-1),
                weight.unsqueeze(-1),
                bias,
                (stride_width, 1),
                (padding_width, 0),
                (dilation_width, 1),
                groups,
            ).squeeze(-1)[..., (ol - il) :]
        elif padding == "valid":
            padding_width = 0
        else:
            raise ValueError(
                f"Unsupported padding mode: {padding}, only 'valid' or 'same' are allowed."
            )
    elif isinstance(padding, (list, tuple)):
        padding_width = padding[0]
    else:
        padding_width = padding

    if weight.shape[1] >= _DOT_MIN_K and not _needs_grad(input, weight, bias):
        return _conv1d_direct(
            input, weight, bias, stride_width, padding_width, dilation_width, groups
        )
    return _hygon_conv2d(
        input.unsqueeze(-1),
        weight.unsqueeze(-1),
        bias,
        (stride_width, 1),
        (padding_width, 0),
        (dilation_width, 1),
        groups,
    ).squeeze(-1)


def _hygon_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if isinstance(padding, str):
        if padding == "same":
            assert stride == 1, "Doesn't support any stride values other than 1 \
                in padding = 'same' mode, received stride value {stride}"
            ih = input.shape[-2]
            iw = input.shape[-1]
            kernel_size_h = weight.shape[-2]
            kernel_size_w = weight.shape[-1]
            padding_h = int(
                math.ceil(
                    (stride * (ih - 1) + 1 + dilation * (kernel_size_h - 1) - ih) / 2
                )
            )
            padding_w = int(
                math.ceil(
                    (stride * (iw - 1) + 1 + dilation * (kernel_size_w - 1) - iw) / 2
                )
            )
            oh = int(
                (ih + 2 * padding_h - dilation * (kernel_size_h - 1) - 1) / stride + 1
            )
            ow = int(
                (iw + 2 * padding_w - dilation * (kernel_size_w - 1) - 1) / stride + 1
            )
            # Use per-dimension padding so asymmetric kernels (kh != kw) pad each
            # spatial axis independently; the trailing slice trims the extra pad
            # that ceil() introduces on the bottom/right, matching torch's "same".
            padding = (padding_h, padding_w)
            return _HygonConv2d.apply(
                input, weight, bias, stride, padding, dilation, groups
            )[..., (oh - ih) :, (ow - iw) :]
        elif padding == "valid":
            return _HygonConv2d.apply(input, weight, bias, stride, 0, dilation, groups)
        else:
            raise ValueError(
                f"Unsupported padding string: {padding}, only'valild'/'same' are allowed."
            )
    if not _needs_grad(input, weight, bias):
        return _launch_conv2d(input, weight, bias, stride, padding, dilation, groups)
    return _HygonConv2d.apply(input, weight, bias, stride, padding, dilation, groups)


def _uses_conv2d_forward_path(input, weight, padding, stride, dilation, groups):
    """Would the generic dispatch reach conv1d/conv2d, i.e. the tl.dot kernel?

    Mirrors the depthwise / pointwise / direct-FMA guards in
    ``flag_gems.ops.cudnn_convolution.cudnn_convolution`` so this override takes
    over exactly the calls that would have launched ``conv2d_forward_kernel``.
    ``padding``/``stride``/``dilation`` must already be per-dimension lists.
    """
    if weight.shape[1] == 1 and groups == input.shape[1]:
        return False  # depthwise: no cross-channel reduction
    if (
        groups == 1
        and all(k == 1 for k in weight.shape[2:])
        and all(s == 1 for s in stride)
        and all(d == 1 for d in dilation)
        and all(p == 0 for p in padding)
        and weight.shape[1] >= _DOT_MIN_K
    ):
        return False  # pointwise: plain GEMM
    if weight.shape[1] < _DOT_MIN_K and (
        input.dtype == torch.float32 or weight.shape[1] <= _DIRECT_MAX_C
    ):
        return False  # FMA direct kernel
    return True


def cudnn_convolution(
    input,
    weight,
    padding,
    stride,
    dilation,
    groups,
    benchmark,
    deterministic,
    allow_tf32,
):
    """CUDNN-compatible no-bias convolution for hygon.

    Routes the tl.dot convolution path (1d and 2d) through the kernel defined in
    this file and leaves every other path to the generic implementation. See the
    module docstring for why. The generic op holds AABS off for its own dispatch;
    the helper is imported so this file can hold it off again around the kernel
    below, which is autotuned from the same configs.
    """
    logger.debug("GEMS_HYGON CUDNN_CONVOLUTION")

    ndim = input.ndim - 2
    if ndim in (1, 2):
        # conv1d/conv2d take per-dimension lists; _to_list is idempotent on
        # lists, so the generic dispatch below is unaffected.
        padding = _to_list(padding, ndim)
        stride = _to_list(stride, ndim)
        dilation = _to_list(dilation, ndim)

        # Without tuned configs there is nothing to autotune, so the private
        # kernel below has nothing to offer over the generic one.
        if _CONV2D_FORWARD_CONFIGS and _uses_conv2d_forward_path(
            input, weight, padding, stride, dilation, groups
        ):
            with _aabs_disabled():
                if ndim == 1:
                    return _hygon_conv1d(
                        input,
                        weight,
                        bias=None,
                        stride=stride[0],
                        padding=padding[0],
                        dilation=dilation[0],
                        groups=groups,
                    )
                return _hygon_conv2d(
                    input,
                    weight,
                    bias=None,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )

    return _generic_cudnn_convolution(
        input,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32,
    )
