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

3. **torch_gcu ships no QuantizedPrivateUse1 kernels.**  Every aten quantized
   factory dies with ``NotImplementedError ... 'Quantizedgcu' backend`` before
   any FlagGems code runs, and FlagGems' generic registrations for those
   factories target ``QuantizedCUDA`` — a dead key on this build.  The integer
   payload is therefore computed on the GCU by the Triton kernel, and the
   quantized wrapper — pure metadata plus the integer storage, whose only
   accessor kernels (``qscheme``/``int_repr``/``q_per_channel_*``, quantized
   ``copy_``) exist for QuantizedCPU — is materialized on the host.  The aten
   entry points are bridged at the bottom of this module so torch-level calls
   with gcu tensors route here instead of into the NotImplementedError.

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
    q_min: tl.constexpr,
    q_max: tl.constexpr,
    USE_FLOAT_ZERO_POINT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Channel index along `axis` for a contiguous (or axis-aligned) tensor:
    #   axis_coord = (flat_index // stride_axis) % shape_axis
    # When ``stride_axis == 1`` (slicing along the trailing dim, the common
    # case) the ``// stride_axis`` divide-by-one drops out.
    if stride_axis == 1:
        axis_coord = offsets % shape_axis
    else:
        axis_coord = (offsets // stride_axis) % shape_axis

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
        zero_point_f = tl.load(zero_points_ptr + axis_coord, mask=mask, other=0.0).to(
            tl.float32
        )
        q = libdevice.nearbyint(libdevice.fma(x_val, inv_scale, zero_point_f))
    else:
        # per_channel_affine: the zero-point is a genuine integer, validated
        # on the host against the target dtype's range (which is
        # int32-representable), so it loads as int32.  Both the CPU and the
        # CUDA kernel round the scaled value *first* and then add the
        # zero-point, so the addition sits outside ``nearbyint``.
        zero_point_i = tl.load(zero_points_ptr + axis_coord, mask=mask, other=0).to(
            tl.int32
        )
        q = libdevice.nearbyint(x_val * inv_scale) + zero_point_i.to(tl.float32)

    # nearbyint implements round-half-to-even (PyTorch's default rounding mode).
    q = tl.minimum(tl.maximum(q, q_min), q_max)
    q = q.to(out_ptr.dtype.element_ty)

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


def _wrap_per_channel_quantized_host(
    int_out, scales, zero_points, axis, use_float_zero_point
):
    """Materialize the per-channel quantized wrapper on the host.

    torch_gcu has no QuantizedPrivateUse1 kernels, so no aten quantized
    accessor (``qscheme``, ``int_repr``, ``q_per_channel_*``, quantized
    ``copy_``) can run against a gcu-stored quantized tensor.  The payload was
    computed on the GCU; the wrapper — metadata plus the integer storage — is
    built on the host, where the native QuantizedCPU kernels serve every
    accessor the tests and downstream code rely on.  The transfers are plain
    aten copies (value-level, never bit-level lane copies).

    The stored qparam dtypes mirror the native quantizers: the affine scheme
    keeps scales in float64 and zero-points in int64, the float-qparams scheme
    keeps both in float32.
    """
    int_out_host = int_out.cpu()
    scales_host = scales.cpu()
    zero_points_host = zero_points.cpu()
    if use_float_zero_point:
        scales_q = scales_host.to(torch.float32)
        zero_points_q = zero_points_host.to(torch.float32)
    else:
        scales_q = scales_host.to(torch.float64)
        zero_points_q = zero_points_host.to(torch.int64)
    return torch._make_per_channel_quantized_tensor(
        int_out_host, scales_q, zero_points_q, axis
    )


def _quantize_per_channel_impl(input, scales, zero_points, axis, dtype):
    use_float_zero_point = _validate_qparams(input, scales, zero_points, axis, dtype)
    int_dtype, q_min, q_max = _QRANGE[dtype]

    input = input.contiguous()
    # The zero-point range check syncs on the original dtype values, before
    # any conversion, like ATen's ``check_zero_points_cuda``.  The vector is
    # read on its host copy: under ``use_gems`` a device-side ``min`` over an
    # int32 zero-point vector would dispatch into FlagGems' float-only
    # ``amin`` and die, and the host copy is needed for the wrap anyway.
    zero_points_host = zero_points.cpu()
    _check_zero_points_on_host(zero_points_host, q_min, q_max, use_float_zero_point)

    # Kernel operands stay in the fp32 / int32 safe domains on the device:
    # scales as fp32 (the fp32 reciprocal chain is the ATen CPU semantics),
    # zero-points as int32 (the host range check above confines them to
    # [q_min, q_max], int32-representable for all three quantized dtypes).
    scales_k = scales.to(torch.float32).contiguous()
    if use_float_zero_point:
        zero_points_k = zero_points.to(torch.float32).contiguous()
    else:
        zero_points_k = zero_points.to(torch.int32).contiguous()

    shape_axis = input.shape[axis]
    stride_axis = input.stride(axis)

    int_out = torch.empty_strided(
        input.shape,
        input.stride(),
        dtype=int_dtype,
        device=input.device,
    )

    n_elements = input.numel()
    if n_elements > 0:
        # Power-of-2 block size for coalesced element-wise access.
        BLOCK_SIZE = 1024
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
                q_min,
                q_max,
                USE_FLOAT_ZERO_POINT=use_float_zero_point,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return _wrap_per_channel_quantized_host(
        int_out, scales, zero_points_host, axis, use_float_zero_point
    )


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

    result = _quantize_per_channel_impl(input, scales, zero_points, axis, dtype)
    # ``copy_`` replaces the destination's integer storage and adopts the
    # source scale/zero-point/axis, which is what the native out= does.
    # Both tensors are host-side quantized wrappers (see the module
    # docstring), so the native QuantizedCPU copy kernel serves this.
    out.copy_(result)
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
#     -> BackendSelect -> Quantizedgcu -> NotImplementedError (the out=
#     helper the accuracy tests use).
#   * ``torch._make_per_channel_quantized_tensor(gcu_int, ...)`` under
#     ``use_gems`` -> the generic FlagGems impl builds a gcu tensor over the
#     wrapper's host storage and launches a Triton kernel on it — a wild
#     device pointer.  The bridge keeps that op on the host instead.
#
# The registrations are part of this op's contract on gcu: the payload is
# computed on the device, the quantized wrapper is a host-side shell, and the
# aten entry points route gcu-argument calls into this module.  The keys are
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

    def _empty_per_channel_affine_quantized_host(
        size,
        *,
        scales,
        zero_points,
        axis,
        dtype=None,
        layout=None,
        device=None,
        pin_memory=None,
        memory_format=torch.contiguous_format,
    ):
        # A gcu device is requested but no Quantizedgcu kernels exist; the
        # factory serves the only representable form on this build — the
        # host-side wrapper — with the qparams moved to the host.  The CPU
        # redispatch below never re-enters this kernel (CPU arguments
        # dispatch on the CPU/QuantizedCPU keys).
        logger.debug("GEMS_ENFLAME EMPTY_PER_CHANNEL_AFFINE_QUANTIZED_HOST")
        scales_host = scales.cpu() if isinstance(scales, torch.Tensor) else scales
        zero_points_host = (
            zero_points.cpu() if isinstance(zero_points, torch.Tensor) else zero_points
        )
        return torch.ops.aten._empty_per_channel_affine_quantized(
            size,
            scales=scales_host,
            zero_points=zero_points_host,
            axis=axis,
            dtype=dtype,
            layout=layout,
            device=None,
            pin_memory=pin_memory,
            memory_format=memory_format,
        )

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
        (
            "_empty_per_channel_affine_quantized",
            _empty_per_channel_affine_quantized_host,
            "QuantizedPrivateUse1",
        ),
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
