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
#
# kunlunxin (XPU) conv_transpose2d.
#
# The generic triton implementation hits the SDNN pipeline on XPU and produces
# wrong values / aborts (tickets/t4).  This overlay binds the operator to the
# vendor implementation instead: the kernel below is a launch-table binding
# shell ("conv_transpose2d_forward" pattern) whose real computation is
# performed by xpudnn::conv2d_transpose_fusion_v2 inside liblaunch_shared.so
# (see third_party/xpu/device/xpu3/launch_extra.cpp).  bf16 has no vendor
# transpose instantiation; bf16 inputs are converted in the C handler
# (cast -> fp32 v2 -> cast back, all inside one launch).
#
# Performance note: the flag_gems `to`/`_to_copy` overlay (a pointwise_dynamic
# kernel) costs ~0.1-0.2 ms of HOST time per call on this stack, and python
# level pad/assign ops used to materialise the dilation are in the same class.
# On these small shapes that host cost dominates the measured latency, so the
# casts and the dilation fold are done with two bare triton kernels (~25 us of
# host time each) instead of the overlay ops.
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# NOTE: the binding shell below deliberately has no @libentry wrapper: the
# wrapper costs ~0.02 ms of host time per call on this stack and the shell body
# is never executed anyway (the launch-table handler serves it).

_ZERO_BIAS_CACHE = {}


_SCRATCH_CACHE = {}


def _scratch_buf(shape, device):
    # internal fp32 staging buffers for the C-side bf16 conversion; reused
    # across calls (they are fully overwritten before use).
    key = (tuple(shape), str(device))
    t = _SCRATCH_CACHE.get(key)
    if t is None:
        t = torch.empty(shape, dtype=torch.float32, device=device)
        _SCRATCH_CACHE[key] = t
    return t


def _zero_bias(out_c, device):
    key = (out_c, str(device))
    t = _ZERO_BIAS_CACHE.get(key)
    if t is None:
        t = torch.zeros(out_c, device=device, dtype=torch.float)
        _ZERO_BIAS_CACHE[key] = t
    return t


@triton.jit
def _cast_kernel(src, dst, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(src + offs, mask=mask, other=0.0)
    tl.store(dst + offs, v.to(dst.dtype.element_ty), mask=mask)


def _fast_cast(t, dtype):
    out = torch.empty(t.shape, dtype=dtype, device=t.device)
    n = t.numel()
    # a large block keeps the CTA count (and its dispatch cost) low; verified
    # ~2x cheaper than BLOCK=1024 on the official event-timing protocol.
    _cast_kernel[(triton.cdiv(n, 16384),)](t, out, n, BLOCK=16384)
    return out


@triton.jit
def _fold_kernel(src, dst, kh, kw, kh2, kw2, dh, dw, BLOCK: tl.constexpr):
    r = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < kh2 * kw2
    col = offs % kw2
    row = offs // kw2
    keep = (row % dh == 0) & (col % dw == 0)
    v = tl.load(
        src + r * kh * kw + (row // dh) * kw + (col // dw), mask=mask, other=0.0
    )
    # NB: folding the keep predicate into the load mask (mask & keep) is ignored
    # by the current XPU triton build (observed on 2026-09-18); an explicit
    # select is used instead.
    v = tl.where(keep, v, 0.0)
    tl.store(dst + r * kh2 * kw2 + offs, v, mask=mask)


def _fold_dilation(weight, dh, dw):
    kh, kw = weight.shape[2], weight.shape[3]
    kh2 = (kh - 1) * dh + 1
    kw2 = (kw - 1) * dw + 1
    rows = weight.numel() // (kh * kw)
    out = torch.empty((rows, kh2 * kw2), device=weight.device, dtype=weight.dtype)
    _fold_kernel[(triton.cdiv(kh2 * kw2, 256), rows)](
        weight, out, kh, kw, kh2, kw2, dh, dw, BLOCK=256
    )
    return out.view(weight.shape[0], weight.shape[1], kh2, kw2)


_DIM_ARGS = [
    "in_n",
    "input_height",
    "input_width",
    "out_c",
    "out_height",
    "out_width",
    "input_n_stride",
    "input_c_stride",
    "input_height_stride",
    "input_width_stride",
    "weight_n_stride",
    "weight_c_stride",
    "weight_height_stride",
    "weight_width_stride",
    "output_n_stride",
    "output_c_stride",
    "output_height_stride",
    "output_width_stride",
    "weight_c",
    "weight_height",
    "weight_width",
    "stride_height",
    "stride_width",
    "padding_height",
    "padding_width",
    "dilation_height",
    "dilation_width",
    "groups",
    "output_padding_height",
    "output_padding_width",
    "has_bias",
]


@triton.jit(do_not_specialize=_DIM_ARGS)
def conv_transpose2d_forward_kernel(
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
    weight_c,
    weight_height,
    weight_width,
    stride_height,
    stride_width,
    padding_height,
    padding_width,
    dilation_height,
    dilation_width,
    groups,
    output_padding_height,
    output_padding_width,
    has_bias,
    scratch_x_pointer,
    scratch_w_pointer,
    out_bf16_pointer,
    BLOCK: tl.constexpr,
):
    # Binding shell: the launch-table handler serves this kernel through
    # xpudnn::conv2d_transpose_fusion_v2 and strips this body on the SDNN
    # pipeline.  The dead tl.dot below is what makes the launcher classify
    # this kernel as an SDNN kernel, which pins the kernel-parameter table
    # layout the C++ handler indexes into (same convention as the conv2d
    # binding kernels).  The branch is unreachable (program ids are >= 0).
    pid = tl.program_id(0)
    offs = tl.arange(0, 1)
    keep = (pid < 0) & (offs < 0)
    v = tl.load(input_pointer + offs, mask=keep, other=0.0)
    tl.store(output_pointer + offs, v.to(output_pointer.dtype.element_ty), mask=keep)
    if pid == -1:
        _z = tl.zeros((16, 16), dtype=tl.float32)
        _d = tl.dot(_z, _z)
        tl.store(
            output_pointer + tl.arange(0, 16),
            tl.sum(_d, axis=0),
            mask=tl.arange(0, 16) < 0,
        )


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
    logger.debug("GEMS_KUNLUNXIN CONV_TRANSPOSE2D")

    from flag_gems.ops.conv_transpose2d import (
        _unsupported_conv_transpose2d,
        _validate_conv_transpose2d_args,
    )

    def _pair2(v):
        # the aten schema may pass int[N] args as lists; accept ints, length-1
        # and length-2 sequences, reject longer ones like the generic _pair
        if isinstance(v, (list, tuple)):
            if len(v) == 1:
                return int(v[0]), int(v[0])
            if len(v) != 2:
                raise RuntimeError("expected a single int or a pair of ints")
            return int(v[0]), int(v[1])
        return v, v

    stride_h, stride_w = _pair2(stride)
    padding_h, padding_w = _pair2(padding)
    output_padding_h, output_padding_w = _pair2(output_padding)
    dilation_h, dilation_w = _pair2(dilation)

    input_was_unbatched = input.dim() == 3
    if input_was_unbatched:
        input = input.unsqueeze(0)

    if not input.is_contiguous():
        input = input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    if not _validate_conv_transpose2d_args(
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
        _unsupported_conv_transpose2d(
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

    orig_dtype = input.dtype
    # bf16 has no vendor transpose instantiation at all; for fp16 the fusion
    # entry's grouped+strided combination is far slower than the fp32 v2 one.
    # Both ride the C handler's cast pipeline (cast -> fp32 v2 -> cast back).
    use_cast_ride = orig_dtype == torch.bfloat16 or (
        orig_dtype == torch.float16 and groups > 1 and (stride_h > 1 or stride_w > 1)
    )
    scratch_x = scratch_w = None
    if use_cast_ride:
        scratch_x = _scratch_buf(input.shape, input.device)
        scratch_w = _scratch_buf(weight.shape, weight.device)

    if bias is None:
        # a None argument would drop its slot from the launcher parameter
        # table and shift every later index; always pass a real fp32 tensor
        # (mirrors the conv2d binding).  The bias value itself is applied in
        # python below; the vendor bias path is unreliable on degenerate or
        # heavily padded shapes.
        bias_f32 = _zero_bias(weight.shape[1] * groups, input.device)
    elif bias.dtype != torch.float32:
        bias_f32 = _fast_cast(bias, torch.float32)
    else:
        bias_f32 = bias
    has_bias = 0 if bias is None else 1

    if dilation_h != 1 or dilation_w != 1:
        # the vendor's combined asymmetric stride+dilation handling is broken
        # (t4 family); materialising the dilation into a zero-stuffed weight
        # is mathematically exact and lets us pass dilation=(1, 1).
        weight = _fold_dilation(weight, dilation_h, dilation_w)
        dilation_h, dilation_w = 1, 1

    # output_padding identity: conv_transpose(s, p, op) equals
    #   conv_transpose(s, p - op, 0)[op : op + out]
    # which keeps the vendor call free of output_padding (its op handling is
    # unreliable, t4 family).  Valid whenever p >= op (holds for the official
    # matrix); otherwise fall back to passing op through.
    borrow_op = (
        (output_padding_h or output_padding_w)
        and padding_h >= output_padding_h
        and padding_w >= output_padding_w
    )
    pad_h = padding_h - output_padding_h if borrow_op else padding_h
    pad_w = padding_w - output_padding_w if borrow_op else padding_w
    op_h = 0 if borrow_op else output_padding_h
    op_w = 0 if borrow_op else output_padding_w

    n, c, h, w = input.shape
    kh, kw = weight.shape[2], weight.shape[3]
    out_c = weight.shape[1] * groups
    out_h = (
        (h - 1) * stride_h
        - 2 * padding_h
        + dilation_h * (kh - 1)
        + output_padding_h
        + 1
    )
    out_w = (
        (w - 1) * stride_w
        - 2 * padding_w
        + dilation_w * (kw - 1)
        + output_padding_w
        + 1
    )

    # the borrowed call produces out0 + 2*op rows/cols; target already holds
    # out0 + op, so the buffer needs exactly target + op
    alloc_h = out_h + (output_padding_h if borrow_op else 0)
    alloc_w = out_w + (output_padding_w if borrow_op else 0)
    if use_cast_ride:
        out = torch.empty(
            (n, out_c, alloc_h, alloc_w), device=input.device, dtype=torch.float32
        )
        out_final = torch.empty(
            (n, out_c, alloc_h, alloc_w), device=input.device, dtype=orig_dtype
        )
        sx, sw = scratch_x, scratch_w
    else:
        out = torch.empty(
            (n, out_c, alloc_h, alloc_w), device=input.device, dtype=input.dtype
        )
        out_final = out
        sx = sw = out
    conv_transpose2d_forward_kernel[(1,)](
        input,
        weight,
        out,
        bias_f32,
        n,
        h,
        w,
        out_c,
        alloc_h,
        alloc_w,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        c,
        kh,
        kw,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        op_h,
        op_w,
        has_bias,
        sx,
        sw,
        out_final,
        BLOCK=64,
    )
    if use_cast_ride:
        out = out_final
    if borrow_op:
        out = out[
            ...,
            output_padding_h : output_padding_h + out_h,
            output_padding_w : output_padding_w + out_w,
        ]
        out = out.contiguous()
    if bias is not None:
        # the vendor bias path is unreliable on degenerate or heavily padded
        # shapes; apply the bias here instead.  For fp32 outputs the add is
        # exact; for narrow dtypes it is done in fp32 with a single rounding
        # (matching the generic op's accumulation contract).
        b = bias_f32.view(1, -1, 1, 1)
        if out.dtype == torch.float32:
            out = out + b
        else:
            out = _fast_cast(_fast_cast(out, torch.float32) + b, out.dtype)
    if input_was_unbatched:
        out = out.squeeze(0)
    return out
