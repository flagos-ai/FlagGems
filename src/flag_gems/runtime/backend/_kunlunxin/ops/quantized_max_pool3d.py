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
from flag_gems.ops.quantized_max_pool3d import (
    _aten_dtype_name,
    _check_quantized_input,
    _int_view,
    _parse_pool3d_params,
    _quantized_max_pool3d_output_shape,
    _resize_quantized_out,
)
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)

# Quantized integer dtypes natively accepted by ``aten::quantized_max_pool3d``,
# mapped to the dtype of their ``int_repr()``.
_QINT_TO_INT = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("quantized_max_pool3d"),
    key=[
        "out_d",
        "out_h",
        "out_w",
        "kernel_d",
        "kernel_h",
        "kernel_w",
        "stride_d",
        "stride_h",
        "stride_w",
    ],
)
@triton.jit
def quantized_max_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    # Output tensor strides
    out_stride_n,
    out_stride_c,
    out_stride_d,
    out_stride_h,
    out_stride_w,
    # Input/Output shapes
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    # Pooling parameters
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Meta-parameters for tiling
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Forward kernel for quantized 3-D max pooling (XPU-safe variant).

    Same tiling as the general kernel, but no ``tl.load(..., other=)``:
    out-of-bounds indices are clamped to a valid address and the value mask is
    applied with ``tl.where`` after the (unconditional) load. Masked stores
    for the tile tail are kept.
    """
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    num_h_blocks = tl.cdiv(out_h, BLOCK_H)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)

    d_block_idx = pid_dhw // (num_h_blocks * num_w_blocks)
    hw_remainder = pid_dhw % (num_h_blocks * num_w_blocks)
    h_block_idx = hw_remainder // num_w_blocks
    w_block_idx = hw_remainder % num_w_blocks

    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    d_out = d_block_idx

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_val, dtype=dtype)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kd in tl.static_range(0, kernel_d):
        d_in = d_out * stride_d - padding_d + kd * dilation_d
        d_ok = (d_in >= 0) & (d_in < in_d)
        d_safe = tl.where(d_ok, d_in, 0)
        for kh in tl.static_range(0, kernel_h):
            h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
            h_ok = (h_in >= 0) & (h_in < in_h)
            h_safe = tl.where(h_ok, h_in, 0)
            for kw in tl.static_range(0, kernel_w):
                w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w
                w_ok = (w_in >= 0) & (w_in < in_w)
                w_safe = tl.where(w_ok, w_in, 0)
                valid = d_ok & h_ok & w_ok
                input_offset = (
                    d_safe * in_stride_d + h_safe * in_stride_h + w_safe * in_stride_w
                )
                current_val = tl.load(input_base_ptr + input_offset)
                max_val_acc = tl.maximum(
                    max_val_acc, tl.where(valid, current_val, min_val)
                )

    out_base_ptr = (
        output_ptr + n_idx * out_stride_n + c_idx * out_stride_c + d_out * out_stride_d
    )
    output_block_ptr = (
        out_base_ptr
        + h_out_offsets[:, None] * out_stride_h
        + w_out_offsets[None, :] * out_stride_w
    )

    out_mask = (h_out_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)
    tl.store(output_block_ptr, max_val_acc, mask=out_mask)


def _launch_quantized_max_pool3d(input, params, out_shape, out_int):
    """Run the pooling kernel over ``input``'s integer representation."""
    kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw = params
    in_n, in_c, in_d, in_h, in_w = input.shape
    _, _, out_d, out_h, out_w = out_shape

    # Alias the storage rather than calling ``int_repr()`` (``_int_view``):
    # the vendor ``int_repr`` registration exists, but a direct view also
    # avoids the dispatch entirely and works for strided inputs.
    int_repr = _int_view(input)

    grid = lambda meta: (
        in_n * in_c,
        out_d
        * triton.cdiv(out_h, meta["BLOCK_H"])
        * triton.cdiv(out_w, meta["BLOCK_W"]),
    )

    quantized_max_pool3d_forward_kernel[grid](
        int_repr,
        out_int,
        *int_repr.stride(),
        *out_int.stride(),
        in_c,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        kd,
        kh,
        kw,
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        dd,
        dh,
        dw,
    )
    return out_int


def _alloc_quantized(out_shape, input, memory_format):
    """Allocate a per-tensor quantized output with the input's qparams."""
    return torch._empty_affine_quantized(
        tuple(out_shape),
        scale=float(input.q_scale()),
        zero_point=int(input.q_zero_point()),
        dtype=input.dtype,
        device=input.device,
        memory_format=memory_format,
    )


def _output_memory_format(input):
    """Match ATen: channels-last in, channels-last out; else contiguous."""
    if input.is_contiguous(memory_format=torch.channels_last_3d) and not (
        input.is_contiguous()
    ):
        return torch.channels_last_3d
    return torch.contiguous_format


def quantized_max_pool3d(
    input: torch.Tensor,
    kernel_size,
    stride=(),
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """Apply 3-D max pooling to a per-tensor quantized tensor (XPU-safe)."""
    logger.debug("GEMS_KUNLUNXIN QUANTIZED_MAX_POOL3D")

    params = _parse_pool3d_params(kernel_size, stride, padding, dilation)
    _check_quantized_input(input)
    out_shape = _quantized_max_pool3d_output_shape(input, params, ceil_mode)

    out = _alloc_quantized(out_shape, input, _output_memory_format(input))

    if out.numel() != 0:
        _launch_quantized_max_pool3d(input, params, out_shape, _int_view(out))

    return out


def quantized_max_pool3d_out(
    input: torch.Tensor,
    kernel_size,
    stride=(),
    padding=0,
    dilation=1,
    ceil_mode=False,
    *,
    out: torch.Tensor,
):
    """``out`` variant of :func:`quantized_max_pool3d` (XPU-safe).

    Follows the same validation order as the autogenerated
    ``quantized_max_pool3d.out`` kernel. XDNN's quantized ``copy_`` neither
    adopts the source's scale/zero_point (ATen does) nor supports as-strided
    destinations, so:

    * when ``out`` already carries the input's quantization parameters, the
      pooled values are written straight through ``out``'s integer view --
      this keeps the alias contract (``res is out``) and preserves a strided
      destination's strides;
    * otherwise a freshly allocated result with the input's parameters is
      returned, mirroring what ATen leaves behind (``res is out`` does not
      hold in that case, but ``out`` is never partially overwritten).
    """
    logger.debug("GEMS_KUNLUNXIN QUANTIZED_MAX_POOL3D_OUT")

    params = _parse_pool3d_params(kernel_size, stride, padding, dilation)
    _check_quantized_input(input)
    out_shape = _quantized_max_pool3d_output_shape(input, params, ceil_mode)

    # ATen's generated ``resize_out`` checks dtype before device.
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {_aten_dtype_name(input.dtype)}, "
            f"but got {_aten_dtype_name(out.dtype)} instead"
        )
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor to have device {input.device}, "
            f"but got {out.device} instead"
        )

    if tuple(out.shape) != tuple(out_shape):
        _resize_quantized_out(out, out_shape)

    match_qparams = abs(float(out.q_scale()) - float(input.q_scale())) < 1e-9 and int(
        out.q_zero_point()
    ) == int(input.q_zero_point())

    if match_qparams:
        if out.numel() != 0:
            _launch_quantized_max_pool3d(input, params, out_shape, _int_view(out))
        return out

    return quantized_max_pool3d(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


def _int_repr_impl(t):
    """Replacement for XDNN's broken ``aten::int_repr`` on ``QuantizedCUDA``.

    Returns an *aliasing* view of the quantized tensor's storage with the
    corresponding plain integer dtype and the tensor's own sizes/strides --
    the semantics of ATen's ``int_repr`` (the integer representation shares
    the quantized storage). XDNN's kernel raises
    ``cudaErrorInvalidDeviceFunction`` for every quantized tensor, which also
    poisons the device context, so ``tensor.int_repr()`` is unusable as a
    reference on this stack without the override.
    """
    if not t.is_quantized:
        raise RuntimeError("int_repr expects a quantized tensor")
    int_dtype = _QINT_TO_INT.get(t.dtype)
    if int_dtype is None:
        raise RuntimeError(
            f"int_repr not implemented for '{t.dtype}' on kunlunxin; "
            "expected one of torch.quint8, torch.qint8, torch.qint32"
        )
    return torch.empty(0, dtype=int_dtype, device=t.device).set_(
        t.untyped_storage(), t.storage_offset(), tuple(t.shape), t.stride()
    )


# Quantized tensors carry ``DispatchKeySet(QuantizedCUDA, ...)``, so a ``CUDA``
# registration is silently ignored; ``QuantizedCUDA`` is the key that is hit.
_int_repr_lib = torch.library.Library("aten", "IMPL")
_int_repr_lib.impl("int_repr", _int_repr_impl, "QuantizedCUDA", allow_override=True)
