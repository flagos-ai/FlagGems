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
"""Kunlunxin override for the quantized max-pool family.

``aten::quantized_max_pool{1d,2d}[.out]`` pool the *integer* representation of
a per-tensor affine-quantized tensor.  Two platform defects block the generic
implementation on XPU3:

1. It materialises the input with ``Tensor.int_repr()``.  ``aten::int_repr``
   has only an NVIDIA QuantizedCUDA kernel on this build, so the call dies with
   ``CUDA error: invalid device function`` before any pooling runs.  This
   override reads the storage through a zero-copy integer alias instead
   (``_int_view``), so no native quantized kernel is needed.
2. Its pooling kernel expresses padding with ``tl.load(..., mask=..., other=min)``.
   XPU3 ignores ``other=`` on an in-bounds masked load, so a padding lane reads
   whatever neighbour shares the cache line and can win the max.  Every real
   padded window is therefore potentially wrong.

The kernels below are adapted from the proven ``max_pool1d``/``max_pool3d``
override recipe: one lane per output, every tap statically unrolled, indices
clamped to the window's own lattice points (no per-tap validity predicate, no
``other=``).  Clamping a tap onto the first/last lattice point inside the input
is numerically harmless -- both ends are taps of the same window and their
value is already part of the max -- and it keeps every load in bounds.

The pooled integers are already the exact dequantized argmax (a positive scale
makes the quantization map monotone), so no requantization happens.
"""

import importlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.quantized_max_pool1d import (
    _check_out,
    _int_dtype_for,
    _int_view_of as _int_view_1d,
    _may_overlap,
    _plan,
    _resize_out,
)
from flag_gems.ops.quantized_max_pool2d import (
    _adopt_quantizer,
    _aliases,
    _as_4d,
    _aten_dtype_name,
    _check_quantized_input,
    _copy_int,
    _int_view,
    _parse_pool_params,
    _quantized_max_pool2d_output_shape,
    _resize_quantized_out,
    _same_quantizer,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

__all__ = [
    "quantized_max_pool1d",
    "quantized_max_pool1d_out",
    "quantized_max_pool2d",
    "quantized_max_pool2d_out",
]

# Quantized dtype -> dtype of its integer representation.
_Q_TO_INT = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}


# ---------------------------------------------------------------------------
# Native ``aten::int_repr`` enabler
# ---------------------------------------------------------------------------
# The tests / benchmarks of every quantized operator read results with
# ``Tensor.int_repr()``.  ``aten::int_repr`` only has an NVIDIA QuantizedCUDA
# kernel on this build, and flag_gems' generic registration targets the plain
# ``CUDA`` key, so the dispatcher never reaches the (working) Triton override in
# ``_kunlunxin/ops/int_repr.py``.  Registering that same pure-Triton copy on the
# ``QuantizedCUDA`` key is what makes the quantized family's observable contract
# (``int_repr``) usable here; it is a real device kernel, not a fallback.  The
# registration is best-effort so a missing/renamed kernel can never break the
# import of this module.
_INT_REPR_LIB = None


def _enable_quantized_int_repr():
    global _INT_REPR_LIB
    if _INT_REPR_LIB is not None:
        return
    try:
        module = importlib.import_module(
            "flag_gems.runtime.backend._kunlunxin.ops.int_repr"
        )
        fn = module.int_repr
    except Exception:  # pragma: no cover - defensive
        logger.debug("int_repr override unavailable; QuantizedCUDA key not enabled")
        return
    lib = torch.library.Library("aten", "IMPL")
    try:
        lib.impl("int_repr", fn, "QuantizedCUDA")
        lib.impl("int_repr.out", fn, "QuantizedCUDA")
    except Exception:  # pragma: no cover - defensive
        try:
            lib._destroy()
        except Exception:
            pass
        return
    _INT_REPR_LIB = lib


_enable_quantized_int_repr()


def _block_for(total, elem_size):
    """Tile ladder measured on XPU3 for integer element copies/pools."""
    if total <= 2048:
        block = 512
    elif total <= 16384:
        block = 1024
    elif elem_size <= 1:
        block = 4096 if total > 262144 else 2048
    else:
        block = 2048 if total > 163840 else 1024
    return block


@triton.jit
def _lattice(start, in_size, dilation):
    """First/last lattice point of a window that lies inside ``[0, in_size)``.

    Returns ``(lo, hi, has_tap)``.  ``lo`` is the first tap the window would
    read that is inside the input and ``hi`` the last, both snapped to the
    window's lattice ``start + m * dilation``.  ``has_tap`` is false when the
    window falls entirely in the padding (legal with ``dilation > 1``); the
    returned ``lo``/``hi`` are still clamped into range so the (discarded) load
    stays inside the allocation.
    """
    if dilation == 1:
        lo = tl.maximum(start, 0)
    else:
        lo = tl.where(
            start >= 0,
            start,
            start + ((-start + dilation - 1) // dilation) * dilation,
        )
    has_tap = lo <= in_size - 1
    lo = tl.minimum(lo, in_size - 1)
    hi = lo + ((in_size - 1 - lo) // dilation) * dilation
    return lo, hi, has_tap


@libentry()
@triton.jit
def _qmaxpool1d_kernel(
    in_ptr,
    out_ptr,
    n,
    c,
    in_l,
    out_l,
    in_stride_n,
    in_stride_c,
    in_stride_l,
    out_stride_n,
    out_stride_c,
    out_stride_l,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    neutral: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_l_blocks = tl.cdiv(out_l, BLOCK)
    nc = pid // num_l_blocks
    l_block = pid % num_l_blocks
    n_idx = nc // c
    c_idx = nc % c

    q = (l_block * BLOCK + tl.arange(0, BLOCK)).to(tl.int32)
    out_mask = q < out_l
    start = q * stride - padding
    lo, hi, has_tap = _lattice(start, in_l, dilation)

    acc_ty: tl.constexpr = in_ptr.dtype.element_ty
    mx = tl.full((BLOCK,), neutral, dtype=acc_ty)

    in_base = n_idx * in_stride_n + c_idx * in_stride_c
    for k in tl.static_range(0, kernel_size):
        il = tl.minimum(tl.maximum(start + k * dilation, lo), hi)
        value = tl.load(in_ptr + in_base + il * in_stride_l)
        mx = tl.maximum(mx, tl.where(has_tap, value, neutral))

    out_base = n_idx * out_stride_n + c_idx * out_stride_c
    tl.store(out_ptr + out_base + q * out_stride_l, mx, mask=out_mask)


@libentry()
@triton.jit
def _qmaxpool2d_kernel(
    in_ptr,
    out_ptr,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    neutral: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block = pid_hw // num_w_blocks
    w_block = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    h_out = (h_block * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int32)
    w_out = (w_block * BLOCK_W + tl.arange(0, BLOCK_W)).to(tl.int32)
    start_h = h_out[:, None] * stride_h - padding_h
    start_w = w_out[None, :] * stride_w - padding_w
    lo_h, hi_h, has_h = _lattice(start_h, in_h, dilation_h)
    lo_w, hi_w, has_w = _lattice(start_w, in_w, dilation_w)
    has_tap = has_h & has_w

    acc_ty: tl.constexpr = in_ptr.dtype.element_ty
    mx = tl.full((BLOCK_H, BLOCK_W), neutral, dtype=acc_ty)

    in_base = n_idx * in_stride_n + c_idx * in_stride_c
    for kh in tl.static_range(0, kernel_h):
        ih = tl.minimum(tl.maximum(start_h + kh * dilation_h, lo_h), hi_h)
        for kw in tl.static_range(0, kernel_w):
            iw = tl.minimum(tl.maximum(start_w + kw * dilation_w, lo_w), hi_w)
            value = tl.load(
                in_ptr + in_base + ih * in_stride_h + iw * in_stride_w
            )
            mx = tl.maximum(mx, tl.where(has_tap, value, neutral))

    out_base = n_idx * out_stride_n + c_idx * out_stride_c
    out_mask = (h_out[:, None] < out_h) & (w_out[None, :] < out_w)
    tl.store(
        out_ptr
        + out_base
        + h_out[:, None] * out_stride_h
        + w_out[None, :] * out_stride_w,
        mx,
        mask=out_mask,
    )


def _run_1d(in_int, out_int, kernel_size, stride, padding, dilation):
    in_l = in_int.shape[-1]
    out_l = out_int.shape[-1]
    n, c, in_sn, in_sc = _nc_strides(in_int)
    _, _, out_sn, out_sc = _nc_strides(out_int)
    total = out_int.numel()
    block = _block_for(total, in_int.element_size())
    grid = (n * c * triton.cdiv(out_l, block),)
    with torch_device_fn.device(in_int.device):
        _qmaxpool1d_kernel[grid](
            in_int,
            out_int,
            n,
            c,
            in_l,
            out_l,
            in_sn,
            in_sc,
            in_int.stride(-1),
            out_sn,
            out_sc,
            out_int.stride(-1),
            kernel_size,
            stride,
            padding,
            dilation,
            torch.iinfo(in_int.dtype).min,
            BLOCK=block,
            num_warps=1,
            buffer_size_limit=2048,
            isCloseVectorization=False,
        )


def _nc_strides(tensor):
    if tensor.dim() == 2:
        return tensor.shape[0], 1, tensor.stride(0), 0
    return tensor.shape[0], tensor.shape[1], tensor.stride(0), tensor.stride(1)


def _run_2d(in_int, out_int, params):
    (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    ) = params
    in_int = _as_4d(in_int)
    out_int = _as_4d(out_int)
    _, in_c, in_h, in_w = in_int.shape
    _, _, out_h, out_w = out_int.shape

    bh = 8
    bw = 128
    grid = (
        in_int.shape[0] * in_c,
        triton.cdiv(out_h, bh) * triton.cdiv(out_w, bw),
    )
    with torch_device_fn.device(in_int.device):
        _qmaxpool2d_kernel[grid](
            in_int,
            out_int,
            in_c,
            in_h,
            in_w,
            out_h,
            out_w,
            *in_int.stride(),
            *out_int.stride(),
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            torch.iinfo(in_int.dtype).min,
            BLOCK_H=bh,
            BLOCK_W=bw,
            num_warps=4,
            buffer_size_limit=2048,
            isCloseVectorization=False,
        )


# ---------------------------------------------------------------------------
# 1-D entry points
# ---------------------------------------------------------------------------
def quantized_max_pool1d(
    input,
    kernel_size,
    stride=[],
    padding=[0],
    dilation=[1],
    ceil_mode=False,
):
    logger.debug("GEMS_KUNLUNXIN QUANTIZED_MAX_POOL1D")

    out_shape, kernel_size, stride, padding, dilation = _plan(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )
    int_dtype = _int_dtype_for(input.dtype)
    int_repr = _int_view_1d(input)
    out_int = torch.empty(out_shape, dtype=int_dtype, device=input.device)

    if out_int.numel() > 0:
        _run_1d(int_repr, out_int, kernel_size, stride, padding, dilation)

    return torch._make_per_tensor_quantized_tensor(
        out_int,
        scale=float(input.q_scale()),
        zero_point=int(input.q_zero_point()),
    )


def _adopt_quantizer_1d(out, scale, zero_point):
    """Install ``(scale, zero_point)`` on ``out`` without a value copy."""
    if out.q_scale() == scale and out.q_zero_point() == zero_point:
        return out
    alias = torch._empty_affine_quantized(
        0,
        scale=scale,
        zero_point=zero_point,
        dtype=out.dtype,
        device=out.device,
    )
    alias.set_(
        out.untyped_storage(),
        out.storage_offset(),
        tuple(out.shape),
        out.stride(),
    )
    out.data = alias
    return out


def quantized_max_pool1d_out(
    input,
    kernel_size,
    stride=[],
    padding=[0],
    dilation=[1],
    ceil_mode=False,
    *,
    out=None,
):
    logger.debug("GEMS_KUNLUNXIN QUANTIZED_MAX_POOL1D_OUT")

    if out is None:
        return quantized_max_pool1d(
            input, kernel_size, stride, padding, dilation, ceil_mode
        )

    out_shape, kernel_size, stride, padding, dilation = _plan(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )
    _check_out(input, out)

    scale = float(input.q_scale())
    zero_point = int(input.q_zero_point())

    # Snapshot the input's integer representation *before* ``out`` may be
    # resized, so ``out is input`` pools the original data.
    int_repr = _int_view_1d(input).contiguous()

    _resize_out(out, out_shape)

    if out.numel() > 0:
        out_int = _int_view_1d(out)
        if _may_overlap(input, out):
            staged = torch.empty(
                out_shape, dtype=out_int.dtype, device=out_int.device
            )
            _run_1d(int_repr, staged, kernel_size, stride, padding, dilation)
            out_int.copy_(staged)
        else:
            _run_1d(int_repr, out_int, kernel_size, stride, padding, dilation)

    return _adopt_quantizer_1d(out, scale, zero_point)


# ---------------------------------------------------------------------------
# 2-D entry points
# ---------------------------------------------------------------------------
def quantized_max_pool2d(
    input: torch.Tensor,
    kernel_size,
    stride=(),
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    logger.debug("GEMS_KUNLUNXIN QUANTIZED_MAX_POOL2D")

    params = _parse_pool_params(kernel_size, stride, padding, dilation)
    out_shape = _quantized_max_pool2d_output_shape(input, params, ceil_mode)
    _check_quantized_input(input)

    scale = float(input.q_scale())
    zero_point = int(input.q_zero_point())

    # NOTE(layout): the generic implementation gives every 4-D accelerator
    # input a channels-last result (the layout ``quantized_max_pool2d_cudnn``
    # allocates).  There is no native QuantizedCUDA pooling kernel on this
    # build, so that layout contract has no counterpart here, and a
    # channels-last quantized tensor cannot be moved to the host on XPU3
    # (``Tensor.cpu()`` on a non-contiguous quantized tensor launches an
    # unimplemented native kernel -> ``invalid device function``).  A contiguous
    # result keeps the pooled integers fully usable (host comparison / device
    # dequantize both work), so this backend allocates contiguous for 4-D too.
    # The ``.out`` variant still writes through exactly the caller's strides.
    out = torch._empty_affine_quantized(
        out_shape,
        scale=scale,
        zero_point=zero_point,
        dtype=input.dtype,
        device=input.device,
        memory_format=torch.contiguous_format,
    )
    if out.numel() != 0:
        _run_2d(_int_view(input), _int_view(out), params)
    return out


def quantized_max_pool2d_out(
    input: torch.Tensor,
    kernel_size,
    stride=(),
    padding=0,
    dilation=1,
    ceil_mode=False,
    *,
    out: torch.Tensor,
):
    logger.debug("GEMS_KUNLUNXIN QUANTIZED_MAX_POOL2D_OUT")

    params = _parse_pool_params(kernel_size, stride, padding, dilation)
    out_shape = _quantized_max_pool2d_output_shape(input, params, ceil_mode)
    _check_quantized_input(input)

    result = None
    if _aliases(input, out):
        result = quantized_max_pool2d(
            input, kernel_size, stride, padding, dilation, ceil_mode
        )

    if tuple(out.shape) != tuple(out_shape):
        _resize_quantized_out(out, out_shape)

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
    if out.qscheme() != torch.per_tensor_affine:
        raise RuntimeError("Quantized Copy only works with same qscheme")

    if result is None:
        if out.numel() != 0:
            _run_2d(_int_view(input), _int_view(out), params)
        if _same_quantizer(input, out):
            return out
        return _adopt_quantizer(out, input)

    if out.numel() != 0:
        _copy_int(_int_view(out), _int_view(result))
    return _adopt_quantizer(out, result)
