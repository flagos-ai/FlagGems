# Copyright 2026, The FlagOS Contributors.
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
"""Quantized-tensor support for the enflame GCU backend.

``torch_gcu`` registers no kernels for the ``QuantizedPrivateUse1`` dispatch
key, so *no* quantized tensor can be created, moved or inspected on a GCU
device through native ATen: every factory dispatch computes the quantized
dispatch key from the quantized dtype and fails with
``NotImplementedError: Could not run 'aten::empty_quantized' with arguments
from the 'Quantizedgcu' backend``.

The quantized pooling operators do not need a ``TensorImpl``-level quantizer,
though: their arithmetic runs entirely on the integer representation and the
quantizer is only metadata carried from the input to the output. A GCU
quantized tensor is therefore represented here as

* a real GCU storage addressed through a quantized dtype view
  (``uint8_storage.view(torch.quint8)`` -- same item size, so the view keeps
  the storage, sizes, strides and offset), and
* a ``QuantizedGcuTensor`` instance whose Python attributes hold the
  quantizer (scale / zero_point / qscheme).

``torch.Tensor`` method overrides (``q_scale``, ``q_zero_point``, ``qscheme``,
``int_repr``, ``contiguous``, ``clone``, ``to``, ``cpu``) provide the
quantized surface, while view ops (``slice``, ``permute``, ``__getitem__``,
...) fall through to ``__torch_dispatch__`` and are rewrapped with the
source's quantizer. Moving to the CPU builds a *real* ``QuantizedCPU`` tensor,
so reference kernels and value comparisons keep working natively.

So that harnesses can build inputs with ``torch.quantize_per_tensor`` /
``torch.quantize_per_channel`` on a GCU float tensor (every accuracy test and
benchmark does), ATen kernels for these two factories are registered at the
backend's device dispatch key when this module is imported; an existing
registration (e.g. from ``flag_gems.enable()``) is left untouched.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_QINT_TO_INT = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}
_INT_TO_QINT = {v: k for k, v in _QINT_TO_INT.items()}

# Quantized dtype -> (storage dtype, qmin, qmax) used for clamping.
_QUANT_INFO = {
    torch.quint8: (torch.uint8, 0, 255),
    torch.qint8: (torch.int8, -128, 127),
    torch.qint32: (torch.int32, -(2**31), 2**31 - 1),
}


def int_view(qtensor):
    """Alias a GCU quantized tensor's storage as a plain integer tensor.

    Works for :class:`QuantizedGcuTensor` as well as for natively quantized
    tensors: only storage / offset / sizes / strides are read, never the
    quantizer, and the result shares memory with ``qtensor``.
    """
    dtype = _QINT_TO_INT.get(qtensor.dtype)
    if dtype is None:
        raise RuntimeError(f"unsupported quantized dtype: {qtensor.dtype}")
    return torch.empty(0, dtype=dtype, device=qtensor.device).set_(
        qtensor.untyped_storage(),
        qtensor.storage_offset(),
        tuple(qtensor.shape),
        qtensor.stride(),
    )


def is_quantized_gcu_tensor(t):
    return isinstance(t, QuantizedGcuTensor)


def _copy_qmeta(dst, src):
    dst._q_scheme = src._q_scheme
    dst._q_scale = src._q_scale
    dst._q_zero_point = src._q_zero_point
    dst._q_scales = getattr(src, "_q_scales", None)
    dst._q_zero_points = getattr(src, "_q_zero_points", None)
    dst._q_axis = getattr(src, "_q_axis", None)


class QuantizedGcuTensor(torch.Tensor):
    """A GCU-resident quantized tensor (see the module docstring)."""

    __torch_function__ = torch._C._disabled_torch_function_impl

    def q_scale(self):
        return self._q_scale

    def q_zero_point(self):
        return self._q_zero_point

    def qscheme(self):
        return self._q_scheme

    def q_per_channel_scales(self):
        return self._q_scales

    def q_per_channel_zero_points(self):
        return self._q_zero_points

    def q_per_channel_axis(self):
        return self._q_axis

    def int_repr(self):
        return int_view(self).clone()

    def contiguous(self, memory_format=torch.contiguous_format):
        if self.is_contiguous(memory_format=memory_format):
            return self
        return wrap_int_tensor(
            int_view(self).contiguous(memory_format=memory_format), src=self
        )

    def clone(self, *, memory_format=None):
        return wrap_int_tensor(int_view(self).clone(), src=self)

    def cpu(self):
        return self.to("cpu")

    def to(self, *args, **kwargs):
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device, dtype = arg.device, arg.dtype
            elif isinstance(arg, (torch.device, str)):
                device = torch.device(arg)
            elif isinstance(arg, torch.dtype):
                dtype = arg
        if dtype is not None and dtype != self.dtype:
            raise NotImplementedError(
                "QuantizedGcuTensor.to: dtype conversion is not supported"
            )
        if device is None or device.type == self.device.type:
            return self
        if device.type != "cpu":
            raise NotImplementedError(
                "QuantizedGcuTensor.to: only the cpu device is supported"
            )
        # Build a *real* QuantizedCPU tensor carrying the same integers and
        # quantizer, so reference kernels run natively on the result.
        if self._q_scheme == torch.per_tensor_affine:
            wrapped = torch._empty_affine_quantized(
                0,
                scale=self._q_scale,
                zero_point=self._q_zero_point,
                dtype=self.dtype,
                device="cpu",
            )
        else:
            wrapped = torch._empty_per_channel_affine_quantized(
                0,
                scales=self._q_scales.detach().cpu(),
                zero_points=self._q_zero_points.detach().cpu(),
                axis=self._q_axis,
                dtype=self.dtype,
                device="cpu",
            )
        int_cpu = int_view(self).to("cpu")
        wrapped.set_(
            int_cpu.untyped_storage(),
            int_cpu.storage_offset(),
            tuple(int_cpu.shape),
            int_cpu.stride(),
        )
        return wrapped

    @staticmethod
    def __torch_dispatch__(func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Plain redispatch: the tensor is a real GCU tensor addressed through
        # a quantized dtype, so metadata-only ops work unchanged on it.
        with torch._C._DisableTorchDispatch():
            result = func(*args, **kwargs)

        def maybe_rewrap(r):
            if (
                isinstance(r, torch.Tensor)
                and r.dtype in _QINT_TO_INT
                and type(r) is not QuantizedGcuTensor
            ):
                for arg in args:
                    if isinstance(arg, QuantizedGcuTensor) and (
                        r.untyped_storage().data_ptr()
                        == arg.untyped_storage().data_ptr()
                    ):
                        return wrap_int_tensor(r, src=arg)
            return r

        if isinstance(result, (list, tuple)):
            return type(result)(maybe_rewrap(r) for r in result)
        return maybe_rewrap(result)


def wrap_int_tensor(int_tensor, scale=None, zero_point=None, src=None):
    """Wrap a plain integer GCU tensor as a per-tensor quantized tensor.

    The storage, sizes, strides and offset of ``int_tensor`` are kept as they
    are (the quantized dtype has the same item size), so any layout survives.
    With ``src`` (another quantized tensor) the quantizer is copied from it;
    otherwise ``scale`` / ``zero_point`` describe a per-tensor affine one.
    """
    if int_tensor.dtype in _QINT_TO_INT:
        # Already addressed through a quantized dtype (e.g. a slice of
        # another GCU quantized tensor): nothing to re-view.
        viewed = int_tensor
    else:
        viewed = int_tensor.view(_INT_TO_QINT[int_tensor.dtype])
    # ``torch.Tensor._make_subclass`` (the machinery ``torch.nn.Parameter`` is
    # built on) shares the storage, sizes, strides and offset of ``viewed``
    # while re-labeling it as ``QuantizedGcuTensor``; unlike
    # ``Tensor.as_subclass`` it also works after ``import flag_gems`` on the
    # enflame backend, where as_subclass's shallow-copy compatibility check
    # fails for PrivateUse1 tensors.
    qtensor = torch.Tensor._make_subclass(QuantizedGcuTensor, viewed, False)
    qtensor._q_scheme = torch.per_tensor_affine
    qtensor._q_scale = 0.0 if scale is None else float(scale)
    qtensor._q_zero_point = 0 if zero_point is None else int(zero_point)
    qtensor._q_scales = None
    qtensor._q_zero_points = None
    qtensor._q_axis = None
    if src is not None:
        _copy_qmeta(qtensor, src)
    return qtensor


def wrap_per_channel_int_tensor(int_tensor, scales, zero_points, axis):
    """Wrap a plain integer GCU tensor as a per-channel quantized tensor."""
    qtensor = wrap_int_tensor(int_tensor, 1.0, 0)
    qtensor._q_scheme = torch.per_channel_affine
    qtensor._q_scales = scales
    qtensor._q_zero_points = zero_points
    qtensor._q_axis = int(axis)
    return qtensor


def adopt_per_tensor_qparams(dst, src):
    """Overwrite ``dst``'s quantizer with ``src``'s, in place."""
    dst._q_scale = float(src.q_scale())
    dst._q_zero_point = int(src.q_zero_point())


def _suggest_memory_format(a):
    """Mirror of ``TensorImpl::suggest_memory_format`` for the dense cases."""
    if (
        a.dim() == 4
        and a.shape[1] != 1
        and a.is_contiguous(memory_format=torch.channels_last)
    ):
        return torch.channels_last
    if (
        a.dim() == 5
        and a.shape[1] != 1
        and a.is_contiguous(memory_format=torch.channels_last_3d)
    ):
        return torch.channels_last_3d
    return torch.contiguous_format


def _aten_quantize_per_tensor(a, scale, zero_point, dtype):
    """``aten::quantize_per_tensor`` on a GCU float tensor."""
    if dtype not in _QUANT_INFO:
        raise RuntimeError(
            f"quantize_per_tensor: unsupported quantized dtype {dtype}, expected "
            "one of torch.quint8, torch.qint8, torch.qint32"
        )
    storage_dtype, qmin, qmax = _QUANT_INFO[dtype]
    if a.dtype != torch.float32:
        raise RuntimeError(
            f"quantize_per_tensor: expected input of dtype torch.float32, got {a.dtype}"
        )
    memory_format = _suggest_memory_format(a)
    if memory_format is torch.contiguous_format or a.dim() < 4:
        out = torch.empty(a.shape, dtype=storage_dtype, device=a.device)
        a = a.contiguous()
    else:
        out = torch.empty(
            a.shape, dtype=storage_dtype, device=a.device, memory_format=memory_format
        )
        a = a.contiguous(memory_format=memory_format)
    # clamp(rint(x / scale) + zero_point, qmin, qmax); the fp32 clamp is exact
    # for the 8-bit ranges, and the integer clamp below repairs the qint32
    # edges that fp32 cannot represent.
    q = torch.round(a / scale) + int(zero_point)
    q = torch.clamp(q, float(qmin), float(qmax)).to(storage_dtype)
    out.copy_(torch.clamp(q, qmin, qmax))
    return wrap_int_tensor(out, scale, zero_point)


def _aten_quantize_per_channel(a, scales, zero_points, axis, dtype):
    """``aten::quantize_per_channel`` on a GCU float tensor."""
    if dtype not in _QUANT_INFO:
        raise RuntimeError(
            f"quantize_per_channel: unsupported quantized dtype {dtype}, expected "
            "one of torch.quint8, torch.qint8, torch.qint32"
        )
    storage_dtype, qmin, qmax = _QUANT_INFO[dtype]
    if a.dtype != torch.float32:
        raise RuntimeError(
            f"quantize_per_channel: expected input of dtype torch.float32, got {a.dtype}"
        )
    broadcast_shape = [1] * a.dim()
    broadcast_shape[axis] = scales.numel()
    scales_b = scales.reshape(broadcast_shape)
    zero_points_b = zero_points.reshape(broadcast_shape)
    q = torch.round(a / scales_b) + zero_points_b.to(torch.float32)
    q = torch.clamp(q, float(qmin), float(qmax)).to(storage_dtype)
    q = torch.clamp(q, qmin, qmax)
    out = torch.empty(a.shape, dtype=storage_dtype, device=a.device)
    out.copy_(q)
    return wrap_per_channel_int_tensor(out, scales, zero_points, axis)


_LIB = None
_REGISTERED_SCHEMAS = ()

# Set once the module body has fully executed; see _alias_to_first_copy.
_QUANTIZED_TENSOR_GCU_READY = True

_SHARED_ATTRS = (
    "QuantizedGcuTensor",
    "int_view",
    "is_quantized_gcu_tensor",
    "wrap_int_tensor",
    "wrap_per_channel_int_tensor",
    "adopt_per_tensor_qparams",
    "register_quantized_aten_kernels",
)


def _alias_to_first_copy():
    """Keep one class identity across the dual import of this module.

    The enflame arch ops package is reachable under two module names: the
    arch event imports it as the top-level ``gcu400.ops`` (a sys.path trick
    in ``runtime.backend.BackendArchEvent.get_arch_module``), while the
    regular package chain imports
    ``flag_gems.runtime.backend._enflame.gcu400.ops``. Both then import this
    helper under their own name, which would create two distinct
    ``QuantizedGcuTensor`` classes and break ``isinstance`` between them.
    Whichever copy is imported *second* therefore rebinds its public names
    to the first fully-initialized copy.
    """
    import sys

    me = sys.modules[__name__]
    for name, module in list(sys.modules.items()):
        if (
            name != __name__
            and name.endswith("_quantized_tensor_gcu")
            and getattr(module, "_QUANTIZED_TENSOR_GCU_READY", False)
        ):
            for attr in _SHARED_ATTRS:
                setattr(me, attr, getattr(module, attr))
            return


def register_quantized_aten_kernels():
    """Register the quantized factories for the current backend's device key.

    Imported automatically with the enflame ops package; runs once. A kernel
    already registered for the same schema and key (e.g. by an earlier
    ``flag_gems.enable()``) is left in place.
    """
    global _LIB, _REGISTERED_SCHEMAS
    if _LIB is not None:
        return _REGISTERED_SCHEMAS
    from flag_gems import runtime

    dispatch_key = runtime.device.dispatch_key
    lib = torch.library.Library("aten", "IMPL")
    registered = []
    for schema, impl in (
        ("quantize_per_tensor", _aten_quantize_per_tensor),
        ("quantize_per_channel", _aten_quantize_per_channel),
    ):
        try:
            lib.impl(schema, impl, dispatch_key)
            registered.append(schema)
        except RuntimeError as exc:
            logger.debug(
                "skip registering aten::%s at %s: %s", schema, dispatch_key, exc
            )
    _LIB = lib
    _REGISTERED_SCHEMAS = tuple(registered)
    if registered:
        logger.debug("registered GCU quantized factories: %s", registered)
    return _REGISTERED_SCHEMAS


_alias_to_first_copy()
register_quantized_aten_kernels()
