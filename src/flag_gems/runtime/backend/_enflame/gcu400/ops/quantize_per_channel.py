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

3. **The quantized wrapper stays on the device.**  torch_gcu's raw
   ``Quantizedgcu`` key is empty, so the per-channel factory is served by this
   backend through the ``QuantizedPrivateUse1`` bridge
   (``ops/empty_per_channel_affine_quantized.py`` + ``ops/quantized_max_pool2d.py``),
   which redispatches the device-generic native allocator to ``QuantizedCPU``
   with ``device=gcu``.  The resulting wrapper is a real quantized ``Tensor``
   whose payload lives in GCU storage, so it can be addressed by the Triton
   kernel directly and no host round trip is needed: the kernel writes the
   integer payload *in place* through a storage alias (``_payload_view``), both
   for the allocating path and for the ``out=`` path.

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
    ALIGNED_CHANNEL: tl.constexpr,
    POW2_CHANNEL: tl.constexpr,
    EVEN: tl.constexpr,
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

    if EVEN:
        scale = tl.load(scales_ptr + axis_coord).to(tl.float32)
    else:
        scale = tl.load(scales_ptr + axis_coord, mask=mask, other=1.0).to(tl.float32)

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
            zero_point_i = tl.load(zero_points_ptr + axis_coord, mask=mask, other=0).to(
                tl.int32
            )
        q = libdevice.nearbyint(x_val * inv_scale) + zero_point_i.to(tl.float32)

    # nearbyint implements round-half-to-even (PyTorch's default rounding mode).
    q = tl.minimum(tl.maximum(q, q_min), q_max)
    q = q.to(out_ptr.dtype.element_ty)

    if EVEN:
        tl.store(out_ptr + offsets, q)
    else:
        tl.store(out_ptr + offsets, q, mask=mask)


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
    return torch.ops.aten._empty_per_channel_affine_quantized(  # allow-torch-compute
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
    # 4096 elements per program keeps the CTA count low enough that the block
    # bookkeeping stops competing with the memory pipeline (1024 measured 2.4x
    # slower at equal traffic); 8192 regressed, so 4096 is the plateau.
    BLOCK_SIZE = 4096
    shape_axis = input.shape[axis]
    stride_axis = input.stride(axis)
    # A block that starts channel-aligned and stays inside one channel row can
    # derive the channel index from a scalar instead of a per-element modulo.
    aligned_channel = stride_axis == 1 and shape_axis % BLOCK_SIZE == 0
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
            ALIGNED_CHANNEL=aligned_channel,
            POW2_CHANNEL=pow2_channel,
            EVEN=n_elements % BLOCK_SIZE == 0,
            BLOCK_SIZE=BLOCK_SIZE,
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
    # ``amin`` and die.
    _check_zero_points_on_host(zero_points.cpu(), q_min, q_max, use_float_zero_point)

    # Kernel operands stay in the fp32 / int32 safe domains on the device:
    # scales as fp32 (the fp32 reciprocal chain is the ATen CPU semantics),
    # zero-points as int32 (the host range check above confines them to
    # [q_min, q_max], int32-representable for all three quantized dtypes).
    scales_k = scales.to(torch.float32).contiguous()
    if use_float_zero_point:
        zero_points_k = zero_points.to(torch.float32).contiguous()
    else:
        zero_points_k = zero_points.to(torch.int32).contiguous()
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

    payload = _payload_view(out, int_dtype)
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
    return out


# ---------------------------------------------------------------------------
# torch_gcu quantized-backend bridging
# ---------------------------------------------------------------------------
# torch_gcu (a +cpu torch build with the gcu backend on PrivateUse1) ships no
# kernels under the QuantizedPrivateUse1 dispatch key — its display name
# "Quantizedgcu" is the one every "Could not run 'aten::empty_quantized' ...
# NotImplementedError" failure reports — and FlagGems' generic registrations
# for the quantized factories target QuantizedCUDA, a dead key on this build.
# Without the bridge below, every per-channel quantized call path with gcu
# tensors dies inside torch_gcu before reaching FlagGems code:
#
#   * ``torch.quantize_per_channel(gcu, ...)`` -> PrivateUse1 CPU fallback ->
#     native CPU kernel -> ``empty_quantized(device=gcu)`` -> BackendSelect ->
#     Quantizedgcu -> NotImplementedError (the reference path the accuracy
#     tests exercise).
#   * ``torch.ops.aten._empty_per_channel_affine_quantized(..., device=gcu)``
#     -> BackendSelect -> QuantizedPrivateUse1 -> served device-natively by the
#     backend bridge, so it needs no intervention from this module.
#   * ``torch._make_per_channel_quantized_tensor(gcu_int, ...)`` under
#     ``use_gems`` -> the generic FlagGems impl builds a gcu tensor over the
#     wrapper's host storage and launches a Triton kernel on it — a wild
#     device pointer.  The bridge keeps that op on the host instead.
#
# The registrations are part of this op's contract on gcu: the aten entry
# points route gcu-argument calls into this module, and the payload is computed
# on the device into the destination wrapper's own storage.  The keys are
# otherwise unoccupied (FlagGems only registers ``dequantize.self`` on
# QuantizedPrivateUse1), and the PrivateUse1 slots hold the same functions
# ``use_gems()`` would register from ``_FULL_CONFIG``, so a later
# ``use_gems()`` merely warns about the duplicates and keeps these kernels.

_QUANT_BRIDGE_FLAG = "_flag_gems_gcu400_per_channel_quantized_bridge"


def _install_per_channel_quantized_bridge():
    """Register the per-channel quantized aten entries for the gcu keys.

    Idempotent across every module name this file may be imported under: the
    library handle lives on torch-attached state (the values_copy precedent),
    so exactly one import performs the registration.
    """
    if getattr(torch, _QUANT_BRIDGE_FLAG, None) is not None:
        return
    bridge_lib = torch.library.Library("aten", "IMPL")

    def _make_per_channel_quantized_tensor_host(self, scale, zero_point, axis):
        # Host-side wrap of an integer payload: values verbatim, the aten
        # factory's own semantics.  Keeps the generic FlagGems impl — which
        # would hand a gcu tensor a host storage — away from gcu arguments.
        logger.debug("GEMS_ENFLAME MAKE_PER_CHANNEL_QUANTIZED_TENSOR_HOST")
        self_host = self.cpu() if isinstance(self, torch.Tensor) else self
        scale_host = scale.cpu() if isinstance(scale, torch.Tensor) else scale
        zero_point_host = (
            zero_point.cpu() if isinstance(zero_point, torch.Tensor) else zero_point
        )
        return torch._make_per_channel_quantized_tensor(
            self_host, scale_host, zero_point_host, axis
        )

    registrations = (
        ("quantize_per_channel", quantize_per_channel, "PrivateUse1"),
        ("quantize_per_channel.out", quantize_per_channel_out, "PrivateUse1"),
        # The quantized factories are deliberately not overridden here: the
        # QuantizedPrivateUse1 bridge (``empty_per_channel_affine_quantized.py``)
        # already serves them device-natively, which is what the implementation
        # above relies on for its zero-copy payload writes.
        (
            "_make_per_channel_quantized_tensor",
            _make_per_channel_quantized_tensor_host,
            "PrivateUse1",
        ),
    )
    for op_name, kernel, dispatch_key in registrations:
        try:
            bridge_lib.impl(op_name, kernel, dispatch_key)
        except RuntimeError as error:
            logger.warning(
                "quantize_per_channel: bridge registration %s@%s skipped: %s",
                op_name,
                dispatch_key,
                error,
            )
    setattr(torch, _QUANT_BRIDGE_FLAG, bridge_lib)


_install_per_channel_quantized_bridge()
