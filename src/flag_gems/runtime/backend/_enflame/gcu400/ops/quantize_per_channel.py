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

"""Per-channel affine quantization, enflame GCU adaptation.

Three GCU facts reshape the generic implementation:

1. **No fp64 vector arithmetic on the device.**  The generic kernel's fp64
   division chain trips the P04 primitive family: ``tl_extra_shim.nearbyint``
   rejects an fp64 operand with ``KeyError: (triton.language.float64)`` and an
   ``arith.extf`` rewrite dies in GCUIR legalization
   (compiler_problems/P04, sources/flaggems-enflame_quantize_per_channel.md).
   The device chain here is all-fp32.

2. **The fp32 reciprocal chain *is* the reference semantics.**  A 2e8-sample
   near-tie experiment (same P04 archive) showed ATen's CPU kernel computes
   ``clamp(nearbyint(x * rne(1/s)) + zp)`` — a multiply by an IEEE-rounded
   reciprocal, not a wider division — so the fp32 chain matches the CPU
   reference where the generic fp64 chain actually diverges from it.  GCU's
   ``/`` and ``div_rn`` both lower to the approximate path (24.5% of results
   1 ulp off IEEE), so the RNE reciprocal is rebuilt as ``rcp_rn`` plus one
   fma Newton refinement, and the float-qparams scheme fuses the zero-point
   add with ``fma`` like the native CUDA kernel.

3. **The quantized wrapper stays on the device.**  torch_gcu serves the
   per-channel quantized factories through its QuantizedPrivateUse1 backend
   fallbacks (RegisterQuantizedGCU.cpp), so ``_empty_per_channel_affine_quantized``
   with ``device=gcu`` allocates real GCU storage behind a native quantized
   ``Tensor``.  The wrapper can therefore be addressed by the Triton kernel
   directly and no host round trip is needed: the kernel writes the integer
   payload *in place* through a storage alias (``_payload_view``), both for
   the allocating path and for the ``out=`` path.  What that machinery cannot
   fix is the public ``torch.quantize_per_channel`` entry, which torch_gcu's
   transfer layer routes to a CPU fallback ahead of the dispatcher; the entry
   correction installed at the bottom of this module routes GCU calls back to
   this implementation while the FlagGems registration is active.

   In-place qparam adoption: PyTorch keeps a per-channel quantizer's
   ``scales``/``zero_points`` as ordinary tensors, and the accessors return
   those tensors themselves rather than copies, so writing the new qparam
   values into them reproduces the native ``out.copy_(result)`` semantics
   (which adopts the source quantizer) without a metadata setter that ATen does
   not expose.  That is what makes the ``out=`` overload overwrite a stale
   destination quantizer instead of silently keeping it.

int64 zero-points never reach the kernel: the host range check confines them
to the target dtype's ``[q_min, q_max]``, which is int32-representable for
quint8 / qint8 / exactly qint32's range, so the kernel loads them as int32.
"""

import logging

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

from flag_gems.runtime import torch_device_fn

# The quantized-platform plumbing (storage-only ``set_`` redispatch on the
# QuantizedCPU keyset) is shared with the per-tensor sibling, which solved the
# same "adopt the source quantizer without copying" problem first.
from flag_gems.runtime.backend._enflame.gcu400.ops.quantize_per_tensor import (
    _QCPU_KEYSET,
    _SET_STORAGE_OP,
    _gems_registration_active,
)
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# Quantized integer ranges per torch quantized dtype:
#   quint8  -> uint8  [0, 255]
#   qint8   -> int8   [-128, 127]
#   qint32  -> int32  [-2147483648, 2147483647]
# ``zero_points`` is the integer zero-point dtype each scheme stores; the float
# scheme keeps its zero-points in float32.
_QRANGE = {
    torch.quint8: (torch.uint8, 0, 255),
    torch.qint8: (torch.int8, -128, 127),
    torch.qint32: (torch.int32, -2147483648, 2147483647),
}


def _scalar_type_name(dtype):
    """Return the C++ type name PyTorch uses in its error messages."""
    return {
        torch.float16: "Half",
        torch.bfloat16: "BFloat16",
        torch.float64: "Double",
        torch.float32: "Float",
        torch.int8: "Char",
        torch.uint8: "Byte",
        torch.int16: "Short",
        torch.int32: "Int",
        torch.int64: "Long",
        torch.bool: "Bool",
        torch.quint8: "QUInt8",
        torch.qint8: "QInt8",
        torch.qint32: "QInt32",
        torch.quint4x2: "QUInt4x2",
    }.get(dtype, str(dtype))


# Elements per program for the flat (1-D) kernel: it is the register-resident
# working set, so it trades CTA count against per-CTA occupancy.
_BLOCK_SIZE = 4096
_BLOCK_NUM_WARPS = 4


# ``float2int_rn`` is a rounding conversion instruction on GCU, while ``nearbyint`` goes through libdevice's
# slow path (measured 0.47ms vs 0.26ms with the same flat grid, see 16.7M fp32 -> int8 on 4.0).
# Both are round-half-to-even, so the former can be used for 8-bit targets. To avoid triggering undefined
# behavior on int32 overflow, apply a ±2^30 pre-clamp first: the 8-bit quantization ranges ([0,255] / [-128,127])
# plus the zero-point offset are far smaller than 2^30, so any result that would land in range is not changed by this
# clamp,
# while out-of-range values would be clamped to q_min / q_max anyway, so it is bit-for-bit equivalent.
# Use ``float2int_rn`` when q_max does not exceed this value; qint32 keeps ``nearbyint`` (the range hugs
# the int32 boundary, where pre-clamping cannot guarantee equivalence).
_F2I_MAX_QMAX = 65535


@triton.jit
def _round_half_even(scaled, RNE_F2I: tl.constexpr):
    """Rounds ``scaled`` half-to-even in place, keeping fp32."""
    if RNE_F2I:
        safe = tl.minimum(tl.maximum(scaled, -1073741824.0), 1073741824.0)
        return libdevice.float2int_rn(safe).to(tl.float32)
    return libdevice.nearbyint(scaled)


@libentry()
@triton.jit
def quantize_per_channel_kernel(
    x_ptr,
    scales_ptr,
    zero_points_ptr,
    out_ptr,
    n_elements,
    stride_axis,
    shape_axis,
    axis_mask,
    q_min: tl.constexpr,
    q_max: tl.constexpr,
    USE_FLOAT_ZERO_POINT: tl.constexpr,
    SCALAR_CHANNEL: tl.constexpr,
    ALIGNED_CHANNEL: tl.constexpr,
    POW2_CHANNEL: tl.constexpr,
    EVEN: tl.constexpr,
    RNE_F2I: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Tail masking is only needed when a block can run past the last element;
    # dropping it removes a predicate from every load/store in the hot loop.
    if EVEN:
        x_val = tl.load(x_ptr + offsets).to(tl.float32)
    else:
        mask = offsets < n_elements
        x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Channel index along `axis` for a contiguous (or axis-aligned) tensor:
    #   axis_coord = (flat_index // stride_axis) % shape_axis
    # The trailing-dim case (``stride_axis == 1``) drops the divide-by-one and
    # has two cheap forms — both replace the per-element integer modulo, which
    # dominates this kernel's issue cost:
    #   * ALIGNED_CHANNEL: the block never spans a channel boundary, so the
    #     modulo collapses into a per-program scalar plus ``tl.arange``.
    #   * POW2_CHANNEL: the channel count is a power of two, so the modulo is
    #     an AND with a runtime mask (no integer division).
    if ALIGNED_CHANNEL:
        axis_coord = (block_start % shape_axis) + tl.arange(0, BLOCK_SIZE)
    elif POW2_CHANNEL:
        axis_coord = offsets & axis_mask
    elif stride_axis == 1:
        axis_coord = offsets % shape_axis
    else:
        axis_coord = (offsets // stride_axis) % shape_axis

    if SCALAR_CHANNEL:
        # One block covers exactly one channel row, so the qparams collapse to
        # two scalar loads instead of a BLOCK-wide gather — this removes the
        # per-element qparam traffic that otherwise doubles the read volume.
        channel = block_start % shape_axis
        scale = tl.load(scales_ptr + channel).to(tl.float32)
        inv_scale = libdevice.rcp_rn(scale)
        residual = libdevice.fma(-inv_scale, scale, 1.0)
        inv_scale = libdevice.fma(residual, inv_scale, inv_scale)
        if USE_FLOAT_ZERO_POINT:
            zero_point_f = tl.load(zero_points_ptr + channel).to(tl.float32)
            q = libdevice.nearbyint(libdevice.fma(x_val, inv_scale, zero_point_f))
        else:
            zero_point_i = tl.load(zero_points_ptr + channel).to(tl.int32)
            q = _round_half_even(x_val * inv_scale, RNE_F2I) + zero_point_i.to(
                tl.float32
            )
    else:
        if EVEN:
            scale = tl.load(scales_ptr + axis_coord).to(tl.float32)
        else:
            scale = tl.load(scales_ptr + axis_coord, mask=mask, other=1.0).to(
                tl.float32
            )

        # RNE reciprocal: ``rcp_rn`` plus one fma Newton refinement.  GCU's ``/``
        # and ``div_rn`` both lower to the approximate path, which is off IEEE
        # round-to-nearest by 1 ulp on ~24.5% of inputs; this chain reproduces
        # the CPU reference's ``1.0f / scale`` (see the module docstring).
        inv_scale = libdevice.rcp_rn(scale)
        residual = libdevice.fma(-inv_scale, scale, 1.0)
        inv_scale = libdevice.fma(residual, inv_scale, inv_scale)

        if USE_FLOAT_ZERO_POINT:
            # per_channel_affine_float_qparams: the native kernel computes
            #   q = lrintf(raw * (1.f / scale) + zero_point)
            # entirely in float32, fusing the zero-point add into one fma so the
            # product is rounded only once.
            if EVEN:
                zero_point_f = tl.load(zero_points_ptr + axis_coord).to(tl.float32)
            else:
                zero_point_f = tl.load(
                    zero_points_ptr + axis_coord, mask=mask, other=0.0
                ).to(tl.float32)
            q = libdevice.nearbyint(libdevice.fma(x_val, inv_scale, zero_point_f))
        else:
            # per_channel_affine: the zero-point is a genuine integer, validated
            # on the host against the target dtype's range (which is
            # int32-representable), so it loads as int32.  Both the CPU and the
            # CUDA kernel round the scaled value *first* and then add the
            # zero-point, so the addition sits outside ``nearbyint``.
            if EVEN:
                zero_point_i = tl.load(zero_points_ptr + axis_coord).to(tl.int32)
            else:
                zero_point_i = tl.load(
                    zero_points_ptr + axis_coord, mask=mask, other=0
                ).to(tl.int32)
            q = _round_half_even(x_val * inv_scale, RNE_F2I) + zero_point_i.to(
                tl.float32
            )

    # _round_half_even implements round-half-to-even (PyTorch's default mode).
    q = tl.minimum(tl.maximum(q, q_min), q_max)
    q = q.to(out_ptr.dtype.element_ty)

    if EVEN:
        tl.store(out_ptr + offsets, q)
    else:
        tl.store(out_ptr + offsets, q, mask=mask)


# Elements per program for the row-major (2-D) formulation: ``ROWS * CHANNEL``
# is held in registers, so this is the same budget the 1-D kernel spends.
_ROWS_TARGET_BLOCK = 32768
# Upper bound on the rows per program, so a small channel count cannot inflate
# the grid granularity into a single-program launch.
_ROWS_MAX = 64
_ROWS_NUM_WARPS = 4
# Above this element count the reciprocal precompute pays for its extra launch.
_INV_SCALE_MIN_ELEMENTS = 1 << 22
_RECIPROCAL_BLOCK = 1024


@libentry()
@triton.jit
def rne_reciprocal_kernel(scales_ptr, inv_scales_ptr, n_channels, BLOCK: tl.constexpr):
    """Materialize the fp32 RNE reciprocal of one qparam vector.

    ``rcp_rn`` plus the Newton refinement is three ALU ops; the row-major kernel
    would replay them once per program for the whole ``CHANNEL`` vector, so on
    large tensors it is cheaper to pay them once over ``n_channels`` elements and
    let the main kernel read the finished reciprocal.
    """
    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_channels
    scale = tl.load(scales_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    inv_scale = libdevice.rcp_rn(scale)
    residual = libdevice.fma(-inv_scale, scale, 1.0)
    inv_scale = libdevice.fma(residual, inv_scale, inv_scale)
    tl.store(inv_scales_ptr + offsets, inv_scale, mask=mask)


@libentry()
@triton.jit
def quantize_per_channel_rows_kernel(
    x_ptr,
    scales_ptr,
    zero_points_ptr,
    inv_scales_ptr,
    out_ptr,
    n_rows,
    q_min: tl.constexpr,
    q_max: tl.constexpr,
    USE_FLOAT_ZERO_POINT: tl.constexpr,
    INV_PRECOMPUTED: tl.constexpr,
    CHANNEL: tl.constexpr,
    ROWS: tl.constexpr,
    ROW_TAIL: tl.constexpr,
    RNE_F2I: tl.constexpr,
):
    """Quantize a contiguous tensor whose channel axis is the trailing dim.

    The flat space is viewed as ``(n_rows, CHANNEL)``.  On this layout the
    channel coordinate of an element is simply its column, so the qparams are
    loaded once as one coalesced ``CHANNEL``-wide vector and broadcast over the
    rows -- replacing the per-element ``scales_ptr + axis_coord`` gather, which
    re-read the whole qparam vector for every block and doubled the read volume.
    """
    pid = tl.program_id(axis=0)
    row = pid * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, CHANNEL)
    offsets = row[:, None] * CHANNEL + col[None, :]
    if ROW_TAIL:
        mask = row[:, None] < n_rows
        x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    else:
        x_val = tl.load(x_ptr + offsets).to(tl.float32)

    # Same fp32 RNE reciprocal chain as the 1-D kernel (see module docstring),
    # unless the launcher already materialized it for this qparam vector.
    if INV_PRECOMPUTED:
        inv_scale = tl.load(inv_scales_ptr + col).to(tl.float32)[None, :]
    else:
        scale = tl.load(scales_ptr + col).to(tl.float32)
        inv_scale = libdevice.rcp_rn(scale)
        residual = libdevice.fma(-inv_scale, scale, 1.0)
        inv_scale = libdevice.fma(residual, inv_scale, inv_scale)[None, :]

    if USE_FLOAT_ZERO_POINT:
        zero_point_f = tl.load(zero_points_ptr + col).to(tl.float32)[None, :]
        q = libdevice.nearbyint(libdevice.fma(x_val, inv_scale, zero_point_f))
    else:
        zero_point_i = tl.load(zero_points_ptr + col).to(tl.int32)[None, :]
        q = _round_half_even(x_val * inv_scale, RNE_F2I) + zero_point_i.to(tl.float32)

    q = tl.minimum(tl.maximum(q, q_min), q_max)
    q = q.to(out_ptr.dtype.element_ty)
    if ROW_TAIL:
        tl.store(out_ptr + offsets, q, mask=mask)
    else:
        tl.store(out_ptr + offsets, q)


def _is_float_qparams(scales, zero_points):
    """Whether this qparam pair selects ``per_channel_affine_float_qparams``.

    Mirrors ``make_per_channel_affine_quantizer``, which picks the float scheme
    when (and only when) the zero-points are floating point.
    """
    return zero_points.dtype.is_floating_point


def _validate_qparams(input, scales, zero_points, axis, dtype):
    """Validate the input contract the way native ``make_per_channel_affine_quantizer`` does.

    Every check mirrors an ATen assertion, and it runs before the kernel is
    launched so a malformed qparam vector can never be read out of bounds by
    the kernel's per-element ``tl.load``.
    """
    if dtype not in _QRANGE:
        raise NotImplementedError(
            f'"quantize_tensor_per_channel_affine" not implemented for '
            f"'{_scalar_type_name(dtype)}'"
        )

    # Device consistency first -- it is what ATen's argument checker reports
    # before any semantic validation, for both qparam vectors. Native raises
    # instead of silently transferring, so a CPU qparam tensor never reaches
    # the kernel through an implicit H2D copy.
    for name, param in (("scales", scales), ("zero_points", zero_points)):
        if param.device != input.device:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got {name} is on "
                f"{param.device}, different from other tensors on {input.device} "
                "(when checking argument in method wrapper_CUDA__quantize_per_channel)"
            )

    # checkFloatTensor: only float32 inputs reach the native per-channel kernels.
    # The two quantizers phrase this differently -- the float-qparams quantizer
    # checks inside its own ``quantize`` ("Quantize only works on Float Tensor"),
    # the affine one inside the native kernel ("... expects a Float Tensor").
    if input.dtype != torch.float32:
        got = _scalar_type_name(input.dtype)
        if _is_float_qparams(scales, zero_points):
            raise RuntimeError(f"Quantize only works on Float Tensor, got {got}")
        raise RuntimeError(
            f"quantize_tensor_per_channel_affine expects a Float Tensor, got {got}"
        )

    # checkPerChannelParamDims: both vectors must be 1-D with equal length.
    if scales.dim() != 1:
        raise RuntimeError("scale tensor must have dimension 1")
    if zero_points.dim() != 1:
        raise RuntimeError("zero_points tensor must have dimension 1")
    if scales.numel() != zero_points.numel():
        raise RuntimeError("number of elements in scales and zero_points must match")

    # make_per_channel_affine_quantizer: scales are always floating point, and
    # a floating-point zero_points selects the *_float_qparams scheme.
    if not scales.dtype.is_floating_point:
        raise RuntimeError("scale tensor must be floating point")

    # ``axis`` is a Python int in the aten schema, so ``input.size(axis)`` below
    # is safe once the range check has run. Negative axes are rejected rather
    # than accepted, matching ATen's ``0 <= axis && axis < dim``.
    if not (0 <= axis < input.dim()):
        scheme = "float qparams" if _is_float_qparams(scales, zero_points) else "affine"
        raise RuntimeError(
            f"Channel axis out of range in per channel {scheme} quantization. "
            f"Got: {axis}Expected: [0, {input.dim()})"
        )

    n_channels = input.size(axis)
    if scales.numel() != n_channels:
        raise RuntimeError(
            f"length of scales must equal to channel, expected {n_channels} got, "
            f"{scales.numel()}"
        )
    if zero_points.numel() != n_channels:
        raise RuntimeError(
            f"length of zero_points must equal to channel expected {n_channels} got, "
            f"{zero_points.numel()}"
        )

    return _is_float_qparams(scales, zero_points)


def _check_zero_points_on_host(zero_points, q_min, q_max, use_float_zero_point):
    """Validate the zero-point range on the host, mirroring ``check_zero_points_cuda``.

    ATen performs this check inside the CUDA kernel after an explicit device
    synchronization, so the failure is a catchable synchronous RuntimeError
    and the CUDA context survives. The previous device-side PTX ``trap`` turned
    the same input error into an asynchronous, context-killing fault instead.

    ATen checks the lower bound first: a vector that violates both bounds
    reports "below lower bound".
    """
    scheme = "float_qparams" if use_float_zero_point else "affine"
    prefix = f"quantize_tensor_per_channel_{scheme}_cuda"
    if zero_points.numel() == 0:
        return
    zp_min = zero_points.min().item()
    if zp_min < q_min:
        raise RuntimeError(f"{prefix}zero_point is below lower bound.")
    zp_max = zero_points.max().item()
    if zp_max > q_max:
        raise RuntimeError(f"{prefix}zero_point is above upper bound.")


def _check_zero_points_once(zero_points, q_min, q_max, use_float_zero_point):
    """Run ``_check_zero_points_on_host`` at most once per zero-point vector state.

    The check needs the values on the host, i.e. a D2H copy plus a stream
    synchronisation, which costs more than the kernel itself on small tensors.
    The range only depends on the vector's contents, so the verdict is memoized
    on the tensor keyed by ``_version``: any in-place update invalidates it and
    the copy is retaken.
    """
    signature = (zero_points._version, q_min, q_max, use_float_zero_point)
    if getattr(zero_points, _ZP_RANGE_SIG_ATTR, None) == signature:
        return
    _check_zero_points_on_host(zero_points.cpu(), q_min, q_max, use_float_zero_point)
    try:
        setattr(zero_points, _ZP_RANGE_SIG_ATTR, signature)
    except (AttributeError, TypeError):
        pass


def _as_kernel_dtype(param, dtype):
    """Convert one qparam vector to the kernel-side dtype without spurious copies.

    ``.to(dtype).contiguous()`` issues two dispatcher calls even when both are
    no-ops, and each call is ~15us of host time on this backend; the attribute
    checks below keep the common already-normalized case at zero calls.
    """
    if param.dtype != dtype:
        param = param.to(dtype)
    if not param.is_contiguous():
        param = param.contiguous()
    return param


def _qparam_tensors(scales, zero_points, device, use_float_zero_point):
    """Build the qparam tensors a native per-channel quantizer would store.

    The affine scheme keeps scales in float64 and zero-points in int64, the
    float-qparams scheme keeps both in float32; feeding the factory these exact
    dtypes is what makes ``q_per_channel_scales``/``_zero_points`` observable
    with native dtypes.
    """
    if use_float_zero_point:
        return (
            scales.to(device=device, dtype=torch.float32),
            zero_points.to(device=device, dtype=torch.float32),
        )
    return (
        scales.to(device=device, dtype=torch.float64),
        zero_points.to(device=device, dtype=torch.int64),
    )


def _empty_per_channel_quantized(shape, q_scales, q_zero_points, axis, dtype, device):
    """Allocate a per-channel quantized tensor whose payload lives on ``device``.

    The QuantizedPrivateUse1 bridge redispatches this device-generic factory to
    the native allocator, so the result is a real quantized tensor over GCU
    storage — no host shell, and therefore no downward copy of the payload.
    """
    return torch.ops.aten._empty_per_channel_affine_quantized(
        list(shape),
        scales=q_scales,
        zero_points=q_zero_points,
        axis=axis,
        dtype=dtype,
        layout=None,
        device=device,
        pin_memory=None,
        memory_format=torch.contiguous_format,
    )


def _payload_view(quantized, int_dtype):
    """Alias a quantized tensor's storage as a plain integer tensor.

    The kernel addresses the payload as contiguous integers, while the wrapper
    only exposes an opaque quantized dtype; ``set_`` on a zero-sized tensor
    hands back a view of the very same storage, so kernel writes land inside
    the destination (or freshly allocated) quantized tensor with zero copies.
    """
    return torch.empty(0, dtype=int_dtype, device=quantized.device).set_(
        quantized.untyped_storage(),
        quantized.storage_offset(),
        tuple(quantized.shape),
        quantized.stride(),
    )


# Per-tensor memo caches live on the tensors themselves, so the invalidation is
# exact: a cached entry can only match the very object it was computed for, and
# any in-place update bumps ``_version`` and therefore drops the entry.  This is
# what keeps the ``out=`` fast path free of the host round trips (a payload
# alias construction, the zero-point range check's D2H copy and a quantizer
# rebuild) that dominate the small-shape latency of this elementwise kernel.
_PAYLOAD_VIEW_ATTR = "_gems_gcu400_payload_view"
_QUANTIZER_SIG_ATTR = "_gems_gcu400_quantizer_signature"
_ZP_RANGE_SIG_ATTR = "_gems_gcu400_zp_range_signature"


def _cached_payload_view(quantized, int_dtype):
    """``_payload_view`` memoized per (storage, dtype, shape, stride, offset)."""
    key = (
        quantized.untyped_storage().data_ptr(),
        int_dtype,
        tuple(quantized.shape),
        tuple(quantized.stride()),
        quantized.storage_offset(),
    )
    cached = getattr(quantized, _PAYLOAD_VIEW_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    view = _payload_view(quantized, int_dtype)
    try:
        setattr(quantized, _PAYLOAD_VIEW_ATTR, (key, view))
    except (AttributeError, TypeError):
        pass
    return view


def _launch_quantize_kernel(
    input,
    scales_k,
    zero_points_k,
    int_out,
    axis,
    q_min,
    q_max,
    use_float_zero_point,
):
    """Launch the element-wise quantize kernel over the flat element space."""
    n_elements = input.numel()
    if n_elements == 0:
        return
    BLOCK_SIZE = _BLOCK_SIZE
    shape_axis = input.shape[axis]
    stride_axis = input.stride(axis)
    # Row-major path: a trailing channel axis turns the flat space into
    # ``(n_rows, shape_axis)``, so the qparams collapse into one broadcast vector
    # instead of a per-element gather.  ``tl.arange`` needs a power-of-two width,
    # which is the only requirement the 1-D kernel below does not have.
    if (
        stride_axis == 1
        and shape_axis >= 16
        and shape_axis <= _ROWS_TARGET_BLOCK
        and (shape_axis & (shape_axis - 1)) == 0
    ):
        rows_per_program = max(1, min(_ROWS_MAX, _ROWS_TARGET_BLOCK // shape_axis))
        n_rows = n_elements // shape_axis
        row_grid = (triton.cdiv(n_rows, rows_per_program),)
        # Only worth a second launch once the reciprocal would be replayed often
        # enough to pay for it (see ``rne_reciprocal_kernel``).
        inv_scales = scales_k
        inv_precomputed = n_elements >= _INV_SCALE_MIN_ELEMENTS
        if inv_precomputed:
            inv_scales = torch.empty_like(scales_k)
        with torch_device_fn.device(input.device):
            if inv_precomputed:
                rne_reciprocal_kernel[(triton.cdiv(shape_axis, _RECIPROCAL_BLOCK),)](
                    scales_k,
                    inv_scales,
                    shape_axis,
                    BLOCK=_RECIPROCAL_BLOCK,
                )
            quantize_per_channel_rows_kernel[row_grid](
                input,
                scales_k,
                zero_points_k,
                inv_scales,
                int_out,
                n_rows,
                q_min,
                q_max,
                USE_FLOAT_ZERO_POINT=use_float_zero_point,
                INV_PRECOMPUTED=inv_precomputed,
                CHANNEL=shape_axis,
                ROWS=rows_per_program,
                ROW_TAIL=n_rows % rows_per_program != 0,
                RNE_F2I=q_max <= _F2I_MAX_QMAX,
                num_warps=_ROWS_NUM_WARPS,
            )
        return
    # A block that starts channel-aligned and stays inside one channel row can
    # derive the channel index from a scalar instead of a per-element modulo;
    # when the block covers a whole channel row the qparams become true scalars.
    aligned_channel = stride_axis == 1 and shape_axis % BLOCK_SIZE == 0
    scalar_channel = stride_axis == 1 and shape_axis == BLOCK_SIZE
    pow2_channel = stride_axis == 1 and (shape_axis & (shape_axis - 1)) == 0
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(input.device):
        quantize_per_channel_kernel[grid](
            input,
            scales_k,
            zero_points_k,
            int_out,
            n_elements,
            stride_axis,
            shape_axis,
            shape_axis - 1,
            q_min,
            q_max,
            USE_FLOAT_ZERO_POINT=use_float_zero_point,
            SCALAR_CHANNEL=scalar_channel,
            ALIGNED_CHANNEL=aligned_channel,
            POW2_CHANNEL=pow2_channel,
            EVEN=n_elements % BLOCK_SIZE == 0,
            RNE_F2I=q_max <= _F2I_MAX_QMAX,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=_BLOCK_NUM_WARPS,
        )


def _prepare_quantize_operands(input, scales, zero_points, axis, dtype):
    """Validate the call and normalize the kernel operands.

    Returns ``(input, scales_k, zero_points_k, int_dtype, q_min, q_max,
    use_float_zero_point)`` with ``input`` made contiguous and the qparam
    vectors converted to the fp32 / int32 domains the kernel consumes.
    """
    use_float_zero_point = _validate_qparams(input, scales, zero_points, axis, dtype)
    int_dtype, q_min, q_max = _QRANGE[dtype]

    input = input.contiguous()
    # The zero-point range check syncs on the original dtype values, before
    # any conversion, like ATen's ``check_zero_points_cuda``.  The vector is
    # read on its host copy: under ``use_gems`` a device-side ``min`` over an
    # int32 zero-point vector would dispatch into FlagGems' float-only
    # ``amin`` and die.  The verdict is memoized (see ``_check_zero_points_once``).
    _check_zero_points_once(zero_points, q_min, q_max, use_float_zero_point)

    # Kernel operands stay in the fp32 / int32 safe domains on the device:
    # scales as fp32 (the fp32 reciprocal chain is the ATen CPU semantics),
    # zero-points as int32 (the host range check above confines them to
    # [q_min, q_max], int32-representable for all three quantized dtypes).
    scales_k = _as_kernel_dtype(scales, torch.float32)
    if use_float_zero_point:
        zero_points_k = _as_kernel_dtype(zero_points, torch.float32)
    else:
        zero_points_k = _as_kernel_dtype(zero_points, torch.int32)
    return (
        input,
        scales_k,
        zero_points_k,
        int_dtype,
        q_min,
        q_max,
        use_float_zero_point,
    )


def _quantize_per_channel_impl(input, scales, zero_points, axis, dtype):
    (
        input,
        scales_k,
        zero_points_k,
        int_dtype,
        q_min,
        q_max,
        use_float_zero_point,
    ) = _prepare_quantize_operands(input, scales, zero_points, axis, dtype)

    # Allocate the wrapper first and let the kernel write into its payload, so
    # the quantization result never leaves the device.
    q_scales, q_zero_points = _qparam_tensors(
        scales, zero_points, input.device, use_float_zero_point
    )
    result = _empty_per_channel_quantized(
        input.shape, q_scales, q_zero_points, axis, dtype, input.device
    )
    _launch_quantize_kernel(
        input,
        scales_k,
        zero_points_k,
        _payload_view(result, int_dtype),
        axis,
        q_min,
        q_max,
        use_float_zero_point,
    )
    return result


def quantize_per_channel(input, scales, zero_points, axis, dtype):
    logger.debug("GEMS_ENFLAME QUANTIZE_PER_CHANNEL")
    return _quantize_per_channel_impl(input, scales, zero_points, axis, dtype)


def quantize_per_channel_out(input, scales, zero_points, axis, dtype, *, out=None):
    logger.debug("GEMS_ENFLAME QUANTIZE_PER_CHANNEL_OUT")
    if out is None:
        return _quantize_per_channel_impl(input, scales, zero_points, axis, dtype)
    return _quantize_per_channel_out_impl(input, scales, zero_points, axis, dtype, out)


def _quantize_per_channel_out_impl(input, scales, zero_points, axis, dtype, out):
    # ``out`` must be a quantized tensor of the requested dtype: the
    # autogenerated redispatch wrapper (``check_out_type``) rejects anything
    # else before the operator body runs.
    if not out.is_quantized or out.dtype != dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {dtype}, but got {out.dtype} instead"
        )
    # The native out= path copies the freshly-quantized tensor into ``out`` with
    # ``copy_``, which for a pair of quantized tensors requires the same qscheme
    # and then adopts the source quantizer. A float-qparams result therefore
    # cannot land in an integer-qparams ``out`` (and vice versa).
    float_result = _is_float_qparams(scales, zero_points)
    float_out = out.qscheme() == torch.per_channel_affine_float_qparams
    if float_result != float_out:
        raise RuntimeError("Quantized Copy only works with same qscheme")

    if tuple(out.shape) != tuple(input.shape):
        # ATen warns and resizes a non-empty mismatched out. Resizing a
        # per-channel quantized tensor is not implemented for this backend, so
        # only the exact-shape contract can be served; report the mismatch
        # rather than silently reinterpreting the destination storage.
        raise RuntimeError(
            f"The size of tensor a {tuple(out.shape)} must match the size of "
            f"tensor b {tuple(input.shape)} at non-singleton dimension 0"
        )

    (
        input,
        scales_k,
        zero_points_k,
        int_dtype,
        q_min,
        q_max,
        use_float_zero_point,
    ) = _prepare_quantize_operands(input, scales, zero_points, axis, dtype)

    payload = _cached_payload_view(out, int_dtype)
    if payload.is_contiguous() and payload.device == input.device:
        # Straight into the destination storage: no intermediate buffer.
        int_out = payload
    else:
        int_out = torch.empty(input.shape, dtype=int_dtype, device=input.device)
    _launch_quantize_kernel(
        input,
        scales_k,
        zero_points_k,
        int_out,
        axis,
        q_min,
        q_max,
        use_float_zero_point,
    )
    if int_out is not payload:
        payload.copy_(int_out.to(payload.device))

    # Adopt the source quantizer — the half of the native ``copy_`` contract
    # that torch_gcu's quantized copy silently drops, and the part ATen exposes
    # no setter for.  The values are already in ``out``'s storage, so only the
    # metadata moves: a quantized alias is built over the very same storage with
    # the requested scales / zero-points / axis and installed through ``.data``,
    # which keeps the caller's tensor object and layout untouched.
    #
    # When the same qparam vector state was already installed on this very
    # destination (a quantize loop reusing one ``out``), the rebuild is a no-op
    # that only costs dispatcher calls, so it is skipped.  The signature holds
    # the qparam objects themselves, so identity is exact and any in-place
    # qparam update bumps ``_version`` and forces the rebuild.
    signature = (
        scales,
        zero_points,
        scales._version,
        zero_points._version,
        axis,
        dtype,
        out.untyped_storage().data_ptr(),
    )
    cached = getattr(out, _QUANTIZER_SIG_ATTR, None)
    # Element-wise ``is`` comparison: comparing the tensors with ``==`` would
    # build a boolean tensor and raise on the multi-element case.
    if (
        cached is not None
        and cached[0] is signature[0]
        and cached[1] is signature[1]
        and cached[2:] == signature[2:]
    ):
        return out
    alias = _empty_per_channel_quantized(
        out.shape,
        *_qparam_tensors(scales, zero_points, out.device, use_float_zero_point),
        axis,
        dtype,
        out.device,
    )
    _SET_STORAGE_OP.redispatch(
        _QCPU_KEYSET,
        alias,
        out.untyped_storage(),
        out.storage_offset(),
        tuple(out.shape),
        out.stride(),
    )
    out.data = alias
    try:
        setattr(out, _QUANTIZER_SIG_ATTR, signature)
    except (AttributeError, TypeError):
        pass
    return out


# ---------------------------------------------------------------------------
# torch.quantize_per_channel entry correction
# ---------------------------------------------------------------------------
# The history of this section: it used to install torch.library IMPL
# registrations for the aten per-channel entries (the "quantized bridge"),
# which vendor commit 5fe27c58e ("revert register to aten") removed again.
# The registrations are not needed on the current stack -- torch_gcu itself
# serves the quantized factories and copies through its QuantizedPrivateUse1
# backend fallbacks (RegisterQuantizedGCU.cpp), so the device-native wrapper
# construction below just works.
#
# What is still broken without help is the *public Python entry*
# ``torch.quantize_per_channel``.  Two layers sit in front of it:
#
#   * the per-tensor sibling's reference shim (quantize_per_tensor.py,
#     ``_per_channel_safe``), which computes a CPU reference when FlagGems is
#     not active, and otherwise defers to the entry it captured at install
#     time;
#   * torch_gcu's transfer layer, which routes the captured C++ binding on GCU
#     arguments to its aten_cpu_fallback instead of the dispatcher.
#
# The second point is the trap: under ``use_gems()`` the aten op has this
# backend's Triton kernel registered on PrivateUse1, but the deferred binding
# never consults the dispatcher, so the call silently pays a full
# D2H -> CPU quantize -> H2D round trip (measured 205 ms vs 0.15 ms for the
# Triton kernel on a 4096x4096 quint8 input).  ``torch.ops.aten.
# quantize_per_channel`` does dispatch and reaches the kernel, which is why
# only the ``out`` benchmark (an ops-packet entry) ever exercised the kernel.
# The sibling fixed the same trap for its own entry by routing to the backend
# implementation while the FlagGems registration is active; the wrapper below
# installs the mirrored correction for the per-channel entry.
#
# Composition: this module imports the sibling at the top, so the sibling's
# shim is always installed first and the wrapper chains onto it.  When
# FlagGems is inactive the call still falls through to the sibling's CPU
# reference (the accuracy-reference path the accuracy suite relies on).  The
# wrapper deliberately does not carry the sibling's
# ``_flaggems_gcu_quantize_ref`` marker, so a sibling (re)install after this
# one keeps wrapping this entry instead of skipping it, and its active-branch
# defer then lands back here -- both install orders compose.  No dispatch
# keys are touched, matching the "revert register to aten" policy.

_ENTRY_FLAG = "_flaggems_gcu400_quantize_per_channel_entry"


def _install_quantize_per_channel_entry():
    previous = torch.quantize_per_channel
    if getattr(previous, _ENTRY_FLAG, False):
        return

    def _quantize_per_channel_entry(
        input, scales, zero_points, axis, dtype=None, *args, **kwargs
    ):
        if (
            isinstance(input, torch.Tensor)
            and input.device.type == "gcu"
            and _gems_registration_active()
        ):
            return quantize_per_channel(input, scales, zero_points, axis, dtype)
        return previous(input, scales, zero_points, axis, dtype, *args, **kwargs)

    setattr(_quantize_per_channel_entry, _ENTRY_FLAG, True)
    try:
        torch.quantize_per_channel = _quantize_per_channel_entry
    except (AttributeError, TypeError):
        pass


_install_quantize_per_channel_entry()
