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
import threading

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.utils import libentry

from .conv2d import conv2d_output_size

logger = logging.getLogger(__name__)


def conv3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """
    Determines the output size of a 3D convolution operation.

    Args:
        in_size: Input size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.
        dilation: Dilation.

    Returns:
        Output size of 3D convolution.
    """
    return conv2d_output_size(in_size, kernel_size, stride, padding, dilation)


# Scalar arguments kept as runtime values so a value of 1 is not folded out.
_CONV3D_RUNTIME_ARGS = [
    "in_n",
    "input_depth",
    "input_height",
    "input_width",
    "out_c",
    "out_depth",
    "out_height",
    "out_width",
    "input_n_stride",
    "input_c_stride",
    "input_depth_stride",
    "input_height_stride",
    "input_width_stride",
    "weight_n_stride",
    "weight_c_stride",
    "weight_depth_stride",
    "weight_height_stride",
    "weight_width_stride",
    "output_n_stride",
    "output_c_stride",
    "output_depth_stride",
    "output_height_stride",
    "output_width_stride",
    "weight_c",
    "weight_depth",
    "weight_height",
    "weight_width",
    "stride_depth",
    "stride_height",
    "stride_width",
    "padding_depth",
    "padding_height",
    "padding_width",
    "dilation_depth",
    "dilation_height",
    "dilation_width",
    "groups",
]


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("conv3d_forward"),
#     key=[
#         "in_n",
#         "weight_c",
#         "input_depth",
#         "input_height",
#         "input_width",
#         "out_c",
#         "out_depth",
#         "out_height",
#         "out_width",
#         "weight_depth",
#         "weight_height",
#         "weight_width",
#         "stride_depth",
#         "stride_height",
#         "stride_width",
#         "padding_depth",
#         "padding_height",
#         "padding_width",
#         "groups",
#     ],
# )
@triton.jit(do_not_specialize=_CONV3D_RUNTIME_ARGS)
def conv3d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
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
    groups,
    # Constexpr duplicates kept only for the static loop bounds below.
    weight_c_c: tl.constexpr,
    weight_depth_c: tl.constexpr,
    weight_height_c: tl.constexpr,
    weight_width_c: tl.constexpr,
    BLOCK_NI_DO_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_ni_do_ho_wo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    # caculate in_n out_depth out_height out_weight value in kernel
    ni_do_ho_wo_offset = pid_ni_do_ho_wo * BLOCK_NI_DO_HO_WO + tl.arange(
        0, BLOCK_NI_DO_HO_WO
    )
    ni_do_ho_offset = ni_do_ho_wo_offset // out_width
    ni_do_offset = ni_do_ho_offset // out_height
    in_n_point_value = ni_do_offset // out_depth
    output_depth_point_value = ni_do_offset % out_depth
    output_height_point_value = ni_do_ho_offset % out_height
    output_width_point_value = ni_do_ho_wo_offset % out_width

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

    accum = tl.zeros((BLOCK_NI_DO_HO_WO, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT = (weight_c_c + BLOCK_CI - 1) // BLOCK_CI
    for dhwc in range(
        weight_depth_c * weight_height_c * weight_width_c * BLOCK_CI_COUNT
    ):
        c = (dhwc % BLOCK_CI_COUNT) * BLOCK_CI
        dhw = dhwc // BLOCK_CI_COUNT
        dh = dhw // weight_width_c
        d = dh // weight_height_c
        h = dh % weight_height_c
        w = dhw % weight_width_c

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

        input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0)
        weight_block = tl.load(curr_weight_pointer, mask=weight_mask, other=0.0)

        accum += tl.sum(input_block[:, :, None] * weight_block[None, :, :], axis=1)
    bias_pointer += (pid_group[None] * out_per_group_c)[None, :] + output_c_offset[
        None, :
    ]
    mask_bias = (output_c_offset < out_per_group_c)[None, :]
    bias = tl.load(bias_pointer, mask=mask_bias, other=0.0).to(tl.float32)
    accum += bias
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


# A 1x1x1 convolution with padding p is exactly the same convolution with
# padding 0 applied to the spatially zero-padded input: the output extent
# `(n + 2p - 1) // s + 1` is identical and the kernel reads the same shifted
# samples. `conv3d` folds the padding into the input for that shape.
_ZERO_PAD3D_BLOCK = 1024


@triton.jit
def _zero_pad3d_spatial_kernel(
    src_ptr,
    dst_ptr,
    total,
    in_c,
    src_d,
    src_h,
    src_w,
    src_sn,
    src_sc,
    src_sd,
    src_sh,
    src_sw,
    dst_d,
    dst_h,
    dst_w,
    pad_d,
    pad_h,
    pad_w,
    BLOCK: tl.constexpr,
):
    # `dst` is a freshly allocated contiguous (n, c, dst_d, dst_h, dst_w) tensor,
    # so the flat program offset already is its element index. The source is
    # addressed through its own strides, which keeps non-contiguous inputs exact.
    # Every computed address is kept inside the source by clamping, and the zero
    # fill is done with `tl.where` rather than by relying on `other=` on a
    # masked load.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < total
    safe = tl.where(in_range, offs, 0)
    w = safe % dst_w
    t = safe // dst_w
    h = t % dst_h
    t = t // dst_h
    d = t % dst_d
    t = t // dst_d  # t == n * in_c + c
    n = t // in_c
    c = t % in_c
    sd = d - pad_d
    sh = h - pad_h
    sw = w - pad_w
    inside = (
        (sd >= 0) & (sd < src_d) & (sh >= 0) & (sh < src_h) & (sw >= 0) & (sw < src_w)
    )
    sd = tl.where(inside, sd, 0)
    sh = tl.where(inside, sh, 0)
    sw = tl.where(inside, sw, 0)
    src_off = n * src_sn + c * src_sc + sd * src_sd + sh * src_sh + sw * src_sw
    val = tl.load(src_ptr + src_off)
    val = tl.where(in_range & inside, val, 0.0)
    tl.store(dst_ptr + offs, val, mask=in_range)


def _zero_pad3d_spatial(x, pad_d, pad_h, pad_w):
    """Zero-pad the three spatial axes of a 5D (n, c, d, h, w) tensor."""
    n, c, d, h, w = x.shape
    dst_d = d + 2 * pad_d
    dst_h = h + 2 * pad_h
    dst_w = w + 2 * pad_w
    out = torch.empty((n, c, dst_d, dst_h, dst_w), device=x.device, dtype=x.dtype)
    total = n * c * dst_d * dst_h * dst_w
    if total == 0:
        return out
    grid = (triton.cdiv(total, _ZERO_PAD3D_BLOCK),)
    _zero_pad3d_spatial_kernel[grid](
        x,
        out,
        total,
        c,
        d,
        h,
        w,
        *x.stride(),
        dst_d,
        dst_h,
        dst_w,
        pad_d,
        pad_h,
        pad_w,
        BLOCK=_ZERO_PAD3D_BLOCK,
    )
    return out


# class Conv3d(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
#         pass


_VENDOR3_CACHE = {}
_VENDOR3_BLOCKED = set()
_VENDOR3_LOCK = threading.Lock()
_VENDOR3_MAX_KEYS = 64
_VENDOR3_BIAS = {}


def _zero_bias3(out_c, device):
    key = (out_c, str(device))
    t = _VENDOR3_BIAS.get(key)
    if t is None:
        t = torch.zeros(out_c, device=device, dtype=torch.float)
        _VENDOR3_BIAS[key] = t
    return t


def _raw_stream3(dev_index):
    try:
        return torch._C._cuda_getCurrentRawStream(dev_index)
    except Exception:
        return torch.cuda.current_stream().cuda_stream


def _record3(out, cap, input, out_c, out_dtype, compute_dtype, promote):
    """Build a replay record from a captured launcher call.

    Only the launcher and its scalar arguments are retained; every tensor slot
    is nulled here and re-spliced on each replay, so no tensor content is keyed
    by address.
    """
    a2 = cap.get("a2")
    launcher = cap.get("launcher")
    if a2 is None or launcher is None:
        return None
    tpos = [i for i, v in enumerate(a2) if isinstance(v, torch.Tensor)]
    if len(tpos) < 3:
        return None
    j_in, j_w, j_out = tpos[0], tpos[1], tpos[2]
    jb = tpos[3] if len(tpos) > 3 else None
    a2 = list(a2)
    a2[j_in] = None
    a2[j_w] = None
    a2[j_out] = None
    if jb is not None:
        a2[jb] = None
    return {
        "launcher": launcher,
        "a2": tuple(a2),
        "j_in": j_in,
        "j_w": j_w,
        "j_out": j_out,
        "jb": jb,
        "out_shape": tuple(out.shape),
        "out_dtype": out_dtype,
        "compute_dtype": compute_dtype,
        "promote": promote,
        "out_c": out_c,
        "dev": input.device.index,
        "stream": _raw_stream3(input.device.index),
        "ref": out.detach().clone(),
        "verified": False,
    }


def _replay3(rec, input, weight, bias):
    if _raw_stream3(rec["dev"]) != rec["stream"]:
        raise RuntimeError("stream changed")
    a = list(rec["a2"])
    if rec["promote"]:
        xi = input.to(torch.float32)
        wi = weight.to(torch.float32)
        bi = (
            bias.to(torch.float32)
            if bias is not None
            else _zero_bias3(rec["out_c"], input.device)
        )
    else:
        xi, wi = input, weight
        bi = bias if bias is not None else _zero_bias3(rec["out_c"], input.device)
    out = torch.empty(rec["out_shape"], device=input.device, dtype=rec["compute_dtype"])
    a[rec["j_in"]] = xi
    a[rec["j_w"]] = wi
    a[rec["j_out"]] = out
    if rec["jb"] is not None:
        a[rec["jb"]] = bi
    rec["launcher"].launch(*a)
    return out


def _verify3(rec, out):
    ref = rec["ref"]
    if ref is None:
        return True
    tol = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 2e-2}[
        rec["compute_dtype"]
    ]
    try:
        d = (out.detach().float() - ref.float()).abs().max().item()
    except Exception:
        return False
    if d <= tol:
        rec["verified"] = True
        rec["ref"] = None
        return True
    return False


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    logger.debug("GEMS_KUNLUNXIN CONV3D")

    assert weight.ndim == 5, "Weights must be 5D, received shape {weight.shape}"
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
        stride_depth, stride_height, stride_width = stride
    else:
        stride_depth = stride_height = stride_width = stride

    if isinstance(dilation, (list, tuple)):
        dilation_depth, dilation_height, dilation_width = dilation
    else:
        dilation_depth = dilation_height = dilation_width = dilation

    # `padding` may be the strings "valid"/"same". "valid" is exactly
    # symmetric zero padding; "same" pads dilation*(k-1) in total, split as
    # floor(total/2) before and the remainder after, so an odd total needs an
    # asymmetric pad, which this kernel cannot express (it takes one symmetric
    # value per axis). For an odd total we launch with ceil(total/2) symmetric
    # (the output is one element larger per such axis) and drop the leading
    # element, which is the identical convolution shifted by one.
    crop = (0, 0, 0)
    if isinstance(padding, str):
        if padding == "valid":
            padding_depth = padding_height = padding_width = 0
        elif padding == "same":
            if stride_depth != 1 or stride_height != 1 or stride_width != 1:
                raise RuntimeError("conv3d only supports padding='same' with stride 1")
            pads = []
            crops = []
            for dil, ksize in (
                (dilation_depth, weight.shape[2]),
                (dilation_height, weight.shape[3]),
                (dilation_width, weight.shape[4]),
            ):
                total = dil * (ksize - 1)
                pads.append(total // 2 + (total % 2))
                crops.append(total % 2)
            padding_depth, padding_height, padding_width = pads
            crop = tuple(crops)
        else:
            raise ValueError(
                f"Unsupported padding string: {padding}, only 'valid'/'same' are allowed."
            )
    elif isinstance(padding, (list, tuple)):
        padding_depth, padding_height, padding_width = padding
    else:
        padding_depth = padding_height = padding_width = padding

    in_n, _, input_depth, input_height, input_width = input.shape
    out_c, weight_c, weight_depth, weight_height, weight_width = weight.shape

    # See `_zero_pad3d_spatial_kernel`: fold the padding of a 1x1x1 kernel into
    # the input (identical output extent, identical sampled positions).
    if (
        weight_depth == 1
        and weight_height == 1
        and weight_width == 1
        and (padding_depth or padding_height or padding_width)
    ):
        input = _zero_pad3d_spatial(input, padding_depth, padding_height, padding_width)
        in_n, _, input_depth, input_height, input_width = input.shape
        padding_depth = padding_height = padding_width = 0

    out_depth = conv3d_output_size(
        input_depth, weight_depth, stride_depth, padding_depth, dilation_depth
    )

    out_height = conv3d_output_size(
        input_height, weight_height, stride_height, padding_height, dilation_height
    )

    out_width = conv3d_output_size(
        input_width, weight_width, stride_width, padding_width, dilation_width
    )

    output_dtype = input.dtype

    # For float16 inputs, promote to float32 for computation to prevent overflow
    # Same issue as conv2d: float16 max value ~65504, 3D convolution with large
    # channels/kernels easily overflows causing NaN propagation
    use_fp32_compute = input.dtype == torch.float16
    compute_dtype = torch.float32 if use_fp32_compute else output_dtype

    def _launch(xi, wi, bi):
        out = torch.empty(
            (in_n, out_c, out_depth, out_height, out_width),
            device=xi.device,
            dtype=compute_dtype,
        )
        grid = (
            triton.cdiv(in_n * out_depth * out_height * out_width, 32),
            triton.cdiv(out_c // groups, 32),
            groups,
        )
        conv3d_forward_kernel[grid](
            xi,
            wi,
            out,
            bi,
            in_n,
            input_depth,
            input_height,
            input_width,
            out_c,
            out_depth,
            out_height,
            out_width,
            *xi.stride(),
            *wi.stride(),
            *out.stride(),
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
            weight_c_c=weight_c,
            weight_depth_c=weight_depth,
            weight_height_c=weight_height,
            weight_width_c=weight_width,
            BLOCK_NI_DO_HO_WO=32,
            BLOCK_CI=32,
            BLOCK_CO=32,
        )
        return out

    key = (
        tuple(input.shape),
        tuple(input.stride()),
        input.dtype,
        tuple(weight.shape),
        tuple(weight.stride()),
        input.device.index,
        stride_depth,
        stride_height,
        stride_width,
        padding_depth,
        padding_height,
        padding_width,
        dilation_depth,
        dilation_height,
        dilation_width,
        groups,
        crop,
    )

    rec = _VENDOR3_CACHE.get(key)
    if rec is not None:
        try:
            raw = _replay3(rec, input, weight, bias)
            if rec["verified"] or _verify3(rec, raw):
                out = raw
            else:
                raise RuntimeError("replay mismatch")
        except Exception:
            with _VENDOR3_LOCK:
                _VENDOR3_CACHE.pop(key, None)
                if len(_VENDOR3_BLOCKED) < 256:
                    _VENDOR3_BLOCKED.add(key)
            out = None
    elif key in _VENDOR3_BLOCKED:
        out = None
    else:
        # imported lazily: conv_depthwise2d imports conv2d at module scope
        from .conv_depthwise2d import _capture_and_delegate

        with _VENDOR3_LOCK:
            if len(_VENDOR3_CACHE) >= _VENDOR3_MAX_KEYS:
                out = None
            else:
                if use_fp32_compute:
                    xi = input.to(torch.float32)
                    wi = weight.to(torch.float32)
                else:
                    xi, wi = input, weight
                bi = (
                    bias.to(torch.float32)
                    if bias is not None
                    else _zero_bias3(out_c, input.device)
                )
                raw, cap = _capture_and_delegate(lambda: _launch(xi, wi, bi))
                rec2 = (
                    _record3(
                        raw,
                        cap,
                        input,
                        out_c,
                        output_dtype,
                        compute_dtype,
                        use_fp32_compute,
                    )
                    if cap
                    else None
                )
                if rec2 is None:
                    if len(_VENDOR3_BLOCKED) < 256:
                        _VENDOR3_BLOCKED.add(key)
                    out = raw
                else:
                    _VENDOR3_CACHE[key] = rec2
                    out = raw

    if out is None:
        if use_fp32_compute:
            xi = input.to(torch.float32)
            wi = weight.to(torch.float32)
        else:
            xi, wi = input, weight
        bi = (
            bias.to(torch.float32)
            if bias is not None
            else _zero_bias3(out_c, input.device)
        )
        out = _launch(xi, wi, bi)

    if crop[0] or crop[1] or crop[2]:
        out = out[
            ...,
            crop[0] : crop[0] + out_depth - crop[0],
            crop[1] : crop[1] + out_height - crop[1],
            crop[2] : crop[2] + out_width - crop[2],
        ]

    if use_fp32_compute:
        out = out.to(output_dtype)

    return out
