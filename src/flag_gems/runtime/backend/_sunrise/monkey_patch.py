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

import functools
import inspect
import json
import logging
import math
import numbers
import os
import time

import torch
import torch.nn.functional as F

_PTPU_DEVICE = "ptpu"
_LOGGER = logging.getLogger(__name__)


def _is_ptpu_tensor(value):
    return isinstance(value, torch.Tensor) and value.device.type == _PTPU_DEVICE


def _is_ptpu_device(device):
    if device is None:
        return False
    if isinstance(device, torch.device):
        return device.type == _PTPU_DEVICE
    if isinstance(device, str):
        return device.split(":", 1)[0] == _PTPU_DEVICE
    return False


def _is_cpu_device(device):
    if device is None:
        return False
    if isinstance(device, torch.device):
        return device.type == "cpu"
    if isinstance(device, str):
        return device.split(":", 1)[0] == "cpu"
    return False


def _has_tensor_base_view(tensor):
    return (
        isinstance(tensor, torch.Tensor) and getattr(tensor, "_base", None) is not None
    )


def _to_cpu_if_ptpu(value):
    if _is_ptpu_tensor(value):
        return value.cpu()
    return value


def _to_device_if_tensor(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_to_device_if_tensor(item, device) for item in value)
    return value


def _should_fallback_to_cpu(exc, tensor, aten_op):
    if not _is_ptpu_tensor(tensor):
        return False
    message = str(exc).lower()
    return aten_op.lower() in message and _PTPU_DEVICE in message


def _copy_cpu_result_to_out(result, out):
    if isinstance(out, torch.Tensor):
        out.copy_(_to_device_if_tensor(result, out.device))
        return out
    if isinstance(out, tuple):
        for result_item, out_item in zip(result, out):
            _copy_cpu_result_to_out(result_item, out_item)
        return out
    return None


def _finalize_cpu_result(result, out, device):
    copied_out = _copy_cpu_result_to_out(result, out)
    if copied_out is not None:
        return copied_out
    return _to_device_if_tensor(result, device)


def _copy_result_to_tensor(result, tensor):
    tensor.copy_(_to_device_if_tensor(result, tensor.device))
    return tensor


def _cpu_fallback(tensor, args, kwargs, original_fn):
    cpu_tensor = tensor.cpu()
    cpu_args = tuple(_to_cpu_if_ptpu(arg) for arg in args)
    cpu_kwargs = {key: _to_cpu_if_ptpu(value) for key, value in kwargs.items()}
    result = original_fn(cpu_tensor, *cpu_args, **cpu_kwargs)
    return _finalize_cpu_result(result, kwargs.get("out"), tensor.device)


def _inplace_cpu_fallback(tensor, args, kwargs, original_fn):
    cpu_tensor = tensor.cpu()
    cpu_args = tuple(_to_cpu_if_ptpu(arg) for arg in args)
    cpu_kwargs = {key: _to_cpu_if_ptpu(value) for key, value in kwargs.items()}
    result = original_fn(cpu_tensor, *cpu_args, **cpu_kwargs)
    return _copy_result_to_tensor(result, tensor)


def _torch_function_cpu_fallback(tensor, args, kwargs, original_fn):
    cpu_args = tuple(_to_cpu_if_ptpu(arg) for arg in args)
    cpu_kwargs = {key: _to_cpu_if_ptpu(value) for key, value in kwargs.items()}
    result = original_fn(*cpu_args, **cpu_kwargs)
    return _finalize_cpu_result(result, kwargs.get("out"), tensor.device)


def _torch_function_inplace_cpu_fallback(tensor, args, kwargs, original_fn):
    cpu_args = tuple(_to_cpu_if_ptpu(arg) for arg in args)
    cpu_kwargs = {key: _to_cpu_if_ptpu(value) for key, value in kwargs.items()}
    result = original_fn(*cpu_args, **cpu_kwargs)
    return _copy_result_to_tensor(result, tensor)


def _patch_tensor_copy_scalar_fill_fallback():
    patched_attr = "_flag_gems_sunrise_copy_scalar_fill_patched"
    if getattr(torch.Tensor, patched_attr, False):
        return

    original_fn = torch.Tensor.copy_

    def _scalar_fill_value(src):
        if isinstance(src, torch.Tensor):
            if src.ndim != 0:
                return None
            src = _to_cpu_if_ptpu(src)
            return src.item()
        if isinstance(src, numbers.Number):
            return src
        return None

    @functools.wraps(original_fn)
    def copy_with_scalar_fill_fallback(self, src, *args, **kwargs):
        try:
            return original_fn(self, src, *args, **kwargs)
        except RuntimeError as exc:
            if _flag_gems_use_gems_active() or not _is_ptpu_tensor(self):
                raise
            if "cannot copy src shape: []" not in str(exc):
                raise
            fill_value = _scalar_fill_value(src)
            if fill_value is None:
                raise
            return self.fill_(fill_value)

    torch.Tensor.copy_ = copy_with_scalar_fill_fallback
    setattr(torch.Tensor, patched_attr, True)


def _patch_tensor_method(name, aten_op, inplace=False):
    patched_attr = f"_flag_gems_sunrise_{name}_patched"
    if getattr(torch.Tensor, patched_attr, False):
        return

    original_fn = getattr(torch.Tensor, name)

    @functools.wraps(original_fn)
    def tensor_method_with_ptpu_cpu_fallback(self, *args, **kwargs):
        try:
            return original_fn(self, *args, **kwargs)
        except NotImplementedError as exc:
            if _flag_gems_use_gems_active():
                raise
            if not _should_fallback_to_cpu(exc, self, aten_op):
                raise
            if inplace:
                return _inplace_cpu_fallback(self, args, kwargs, original_fn)
            return _cpu_fallback(self, args, kwargs, original_fn)

    setattr(torch.Tensor, name, tensor_method_with_ptpu_cpu_fallback)
    setattr(torch.Tensor, patched_attr, True)


def _patch_tensor_property(name, aten_op):
    """Patch a `getset_descriptor` property on `torch.Tensor` (e.g. `real`, `imag`).

    Wrap only the getter. Re-raise on non-PTPU dispatches or unrelated aten ops.
    Keep the original setter intact so alias-write semantics (`t.real = ...`)
    still go through the C-side descriptor.
    """
    patched_attr = f"_flag_gems_sunrise_{name}_patched"
    if getattr(torch.Tensor, patched_attr, False):
        return

    original_descriptor = getattr(torch.Tensor, name)
    original_get = original_descriptor.__get__
    original_set = getattr(original_descriptor, "__set__", None)

    def getter(self):
        try:
            return original_get(self)
        except NotImplementedError as exc:
            if _flag_gems_use_gems_active():
                raise
            if not _should_fallback_to_cpu(exc, self, aten_op):
                raise
            cpu_result = original_get(self.cpu())
            device_result = _to_device_if_tensor(cpu_result, self.device)
            if isinstance(cpu_result, torch.Tensor) and cpu_result.is_neg():
                return torch._neg_view(device_result)
            return device_result

    if original_set is None:
        new_descriptor = property(getter)
    else:

        def setter(self, value):
            return original_set(self, value)

        new_descriptor = property(getter, setter)

    setattr(torch.Tensor, name, new_descriptor)
    setattr(torch.Tensor, patched_attr, True)


def _patch_torch_function(name, aten_op, inplace=False):
    patched_attr = f"_flag_gems_sunrise_{name}_patched"
    if getattr(torch, patched_attr, False):
        return

    original_fn = getattr(torch, name)

    @functools.wraps(original_fn)
    def function_with_ptpu_cpu_fallback(*args, **kwargs):
        tensor = args[0] if args else kwargs.get("input")
        try:
            return original_fn(*args, **kwargs)
        except NotImplementedError as exc:
            if _flag_gems_use_gems_active():
                raise
            if not _should_fallback_to_cpu(exc, tensor, aten_op):
                raise
            if inplace:
                return _torch_function_inplace_cpu_fallback(
                    tensor, args, kwargs, original_fn
                )
            return _torch_function_cpu_fallback(tensor, args, kwargs, original_fn)

    setattr(torch, name, function_with_ptpu_cpu_fallback)
    setattr(torch, patched_attr, True)


def _patch_torch_nn_functional(name, aten_op):
    """Patch `torch.nn.functional.<name>(...)` for PTPU CPU fallback.

    Use when the failing call site is inside a `torch.nn` module's `forward`
    that routes through `torch.nn.functional.<name>(...)` (e.g. `F.pad`,
    `F.interpolate`) and the C++ dispatcher does not surface in the Python
    `torch.ops.aten.<op>(...)` packet path.
    """
    patched_attr = f"_flag_gems_sunrise_nn_functional_{name}_patched"
    if getattr(F, patched_attr, False):
        return

    original_fn = getattr(F, name)

    @functools.wraps(original_fn)
    def functional_with_ptpu_cpu_fallback(*args, **kwargs):
        tensor = args[0] if args else kwargs.get("input")
        try:
            return original_fn(*args, **kwargs)
        except NotImplementedError as exc:
            if _flag_gems_use_gems_active():
                raise
            if not _should_fallback_to_cpu(exc, tensor, aten_op):
                raise
            return _torch_function_cpu_fallback(tensor, args, kwargs, original_fn)

    setattr(F, name, functional_with_ptpu_cpu_fallback)
    setattr(F, patched_attr, True)


def _vector_norm_arg(args, kwargs, index, name, default=None):
    return args[index] if len(args) > index else kwargs.get(name, default)


def _normalize_vector_norm_dims(tensor, dim):
    if dim is None:
        return tuple(range(tensor.ndim))
    if isinstance(dim, int):
        return (dim % tensor.ndim,)
    return tuple(d % tensor.ndim for d in dim)


def _maybe_stable_cpu_vector_norm_reference(args, kwargs):
    """Use an explicit high-precision CPU reference for long finite norms.

    PyTorch CPU `torch.linalg.vector_norm` can undercount long float32
    reductions on this environment, especially for multi-dim reductions over
    non-unit-stride slices. The Sunrise/PTPU Triton kernel is much closer to a
    double-precision reference, so keep the device path native and only correct
    the CPU reference helper path outside `flag_gems.use_gems()`.
    """
    tensor = args[0] if args else kwargs.get("input") or kwargs.get("x")
    if (
        _flag_gems_use_gems_active()
        or not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or not tensor.is_floating_point()
        or tensor.dtype not in (torch.float16, torch.float32, torch.bfloat16)
    ):
        return None

    ord_value = _vector_norm_arg(args, kwargs, 1, "ord", 2)
    if ord_value not in (1, 2):
        return None

    dim = _vector_norm_arg(args, kwargs, 2, "dim", None)
    dims = _normalize_vector_norm_dims(tensor, dim)
    if not dims:
        return None

    reduction_numel = math.prod(tensor.shape[d] for d in dims)
    if reduction_numel < 2048:
        return None

    keepdim = _vector_norm_arg(args, kwargs, 3, "keepdim", False)
    dtype = kwargs.get("dtype", None) or tensor.dtype
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    out = kwargs.get("out", None)

    work = tensor.to(torch.float64)
    if ord_value == 1:
        result = work.abs().sum(dim=dims, keepdim=keepdim)
    else:
        result = torch.sqrt((work * work).sum(dim=dims, keepdim=keepdim))
    result = result.to(dtype=dtype)

    if out is not None:
        out.copy_(result)
        return out
    return result


def _patch_torch_linalg_function(name, aten_op):
    """Patch `torch.linalg.<name>(...)` for Sunrise reference/fallback quirks."""
    patched_attr = f"_flag_gems_sunrise_linalg_{name}_patched"
    if getattr(torch.linalg, patched_attr, False):
        return

    original_fn = getattr(torch.linalg, name)

    @functools.wraps(original_fn)
    def linalg_with_ptpu_cpu_fallback(*args, **kwargs):
        tensor = args[0] if args else kwargs.get("input")
        if name == "vector_norm":
            stable_result = _maybe_stable_cpu_vector_norm_reference(args, kwargs)
            if stable_result is not None:
                return stable_result
        try:
            return original_fn(*args, **kwargs)
        except NotImplementedError as exc:
            if _flag_gems_use_gems_active():
                raise
            if not _should_fallback_to_cpu(exc, tensor, aten_op):
                raise
            return _torch_function_cpu_fallback(tensor, args, kwargs, original_fn)

    setattr(torch.linalg, name, linalg_with_ptpu_cpu_fallback)
    setattr(torch.linalg, patched_attr, True)


def _patch_torch_tensor_out(packet_name, aten_op):
    packet = getattr(torch.ops.aten, packet_name)
    patched_attr = "_flag_gems_sunrise_tensor_out_patched"
    if getattr(packet, patched_attr, False):
        return

    original_fn = packet.Tensor_out

    @functools.wraps(original_fn)
    def tensor_out_with_ptpu_cpu_fallback(*args, **kwargs):
        tensor = args[0] if args else kwargs.get("self")
        try:
            return original_fn(*args, **kwargs)
        except NotImplementedError as exc:
            if _flag_gems_use_gems_active():
                raise
            if not _should_fallback_to_cpu(exc, tensor, aten_op):
                raise
            return _torch_function_cpu_fallback(tensor, args, kwargs, original_fn)

    packet.Tensor_out = tensor_out_with_ptpu_cpu_fallback
    setattr(packet, patched_attr, True)


def _patch_torch_out(packet_name, aten_op):
    packet = getattr(torch.ops.aten, packet_name)
    patched_attr = "_flag_gems_sunrise_out_patched"
    if getattr(packet, patched_attr, False):
        return

    original_fn = packet.out

    @functools.wraps(original_fn)
    def out_with_ptpu_cpu_fallback(*args, **kwargs):
        tensor = args[0] if args else kwargs.get("self") or kwargs.get("input")
        try:
            return original_fn(*args, **kwargs)
        except NotImplementedError as exc:
            if _flag_gems_use_gems_active():
                raise
            if not _should_fallback_to_cpu(exc, tensor, aten_op):
                raise
            return _torch_function_cpu_fallback(tensor, args, kwargs, original_fn)

    packet.out = out_with_ptpu_cpu_fallback
    setattr(packet, patched_attr, True)


def _patch_torch_creation_function(name, aten_op):
    """Patch a `torch.<name>(...)` creation op (no dispatch-driving tensor input).

    Detect a PTPU target via the `device=` kwarg, fall back by calling the
    original function on CPU, then move the result to the requested device.
    """
    patched_attr = f"_flag_gems_sunrise_{name}_patched"
    if getattr(torch, patched_attr, False):
        return

    original_fn = getattr(torch, name)

    @functools.wraps(original_fn)
    def creation_with_ptpu_cpu_fallback(*args, **kwargs):
        device = kwargs.get("device")
        try:
            return original_fn(*args, **kwargs)
        except NotImplementedError as exc:
            if _flag_gems_use_gems_active():
                raise
            if not _is_ptpu_device(device):
                raise
            message = str(exc).lower()
            if aten_op.lower() not in message or _PTPU_DEVICE not in message:
                raise
            cpu_kwargs = dict(kwargs)
            cpu_kwargs["device"] = "cpu"
            out = kwargs.get("out")
            if isinstance(out, torch.Tensor) and _is_ptpu_tensor(out):
                cpu_kwargs["out"] = None
            result = original_fn(*args, **cpu_kwargs)
            return _finalize_cpu_result(
                result,
                kwargs.get("out"),
                (
                    torch.device(device)
                    if not isinstance(device, torch.device)
                    else device
                ),
            )

    setattr(torch, name, creation_with_ptpu_cpu_fallback)
    setattr(torch, patched_attr, True)


def _patch_torch_randn_complex_dtype():
    """Generate complex-dtype `torch.randn(...)` on CPU when targeting PTPU.

    PTPU's `randn` implementation calls `normal_` internally, which raises
    `RuntimeError: normal_ does not support complex tensors on PTPU, but got
    c10::complex<...>` for any complex dtype. This is a quirk: the failure
    text is a plain `RuntimeError`, not `NotImplementedError`, and it does
    not name an `aten::...` symbol, so `_should_fallback_to_cpu(...)` and
    `_patch_torch_creation_function(...)` do not fit.

    Narrow guard:

    - Wrap only `torch.randn`
    - Only divert when `dtype` is a complex dtype AND `device` is PTPU
    - Only divert when the raised `RuntimeError` matches the known quirk text
    - Real-dtype `torch.randn(..., device='ptpu')` is untouched
    """
    patched_attr = "_flag_gems_sunrise_randn_complex_dtype_patched"
    if getattr(torch, patched_attr, False):
        return

    original_fn = torch.randn
    complex_quirk_marker = "normal_ does not support complex tensors"
    float64_quirk_marker = "supports only float16, bfloat16 and float32 tensors"

    @functools.wraps(original_fn)
    def randn_with_ptpu_complex_cpu_fallback(*args, **kwargs):
        dtype = kwargs.get("dtype")
        device = kwargs.get("device")
        if (
            isinstance(dtype, torch.dtype)
            and dtype == torch.float64
            and _is_ptpu_device(device)
            and not _flag_gems_use_gems_active()
        ):
            cpu_kwargs = dict(kwargs)
            cpu_kwargs["device"] = "cpu"
            result = original_fn(*args, **cpu_kwargs)
            target_device = (
                device if isinstance(device, torch.device) else torch.device(device)
            )
            return _to_device_if_tensor(result, target_device)
        if (
            isinstance(dtype, torch.dtype)
            and dtype.is_complex
            and _is_ptpu_device(device)
        ):
            try:
                return original_fn(*args, **kwargs)
            except RuntimeError as exc:
                if _flag_gems_use_gems_active():
                    raise
                if complex_quirk_marker not in str(
                    exc
                ) and float64_quirk_marker not in str(exc):
                    raise
                cpu_kwargs = dict(kwargs)
                cpu_kwargs["device"] = "cpu"
                result = original_fn(*args, **cpu_kwargs)
                target_device = (
                    device if isinstance(device, torch.device) else torch.device(device)
                )
                return _to_device_if_tensor(result, target_device)
        return original_fn(*args, **kwargs)

    torch.randn = randn_with_ptpu_complex_cpu_fallback
    setattr(torch, patched_attr, True)


def _patch_torch_cudnn_convolution():
    """Run `torch.cudnn_convolution(...)` on CPU via `F.conv{1,2,3}d` for PTPU.

    `aten::cudnn_convolution` is a CUDA/cuDNN-only op — it is unimplemented on
    PTPU AND on CPU, so the usual "bounce the same call to CPU" trick fails.
    The math is plain (bias-free) convolution, which CPU *does* support through
    `torch.nn.functional.conv{1,2,3}d`. So the fallback both moves to CPU and
    re-expresses the op as the corresponding functional conv, then moves the
    result back to the PTPU device.

    Signature mapping (note `cudnn_convolution` has no bias arg, and its
    `benchmark` / `deterministic` / `allow_tf32` tuning flags have no CPU
    analogue and are dropped):

        cudnn_convolution(input, weight, *, padding, stride, dilation, groups,
                          benchmark, deterministic, allow_tf32)
        -> F.conv{1,2,3}d(input, weight, bias=None,
                          stride=stride, padding=padding,
                          dilation=dilation, groups=groups)

    The conv rank is selected by `input.dim()` (3->1d, 4->2d, 5->3d).
    """
    patched_attr = "_flag_gems_sunrise_cudnn_convolution_patched"
    if getattr(torch, patched_attr, False):
        return

    original_fn = torch.cudnn_convolution
    conv_by_rank = {
        3: F.conv1d,
        4: F.conv2d,
        5: F.conv3d,
    }

    @functools.wraps(original_fn)
    def cudnn_convolution_with_ptpu_cpu_fallback(*args, **kwargs):
        tensor = args[0] if args else kwargs.get("input") or kwargs.get("self")
        try:
            return original_fn(*args, **kwargs)
        except NotImplementedError as exc:
            if _flag_gems_use_gems_active():
                raise
            if not _should_fallback_to_cpu(exc, tensor, "aten::cudnn_convolution"):
                raise

            call_args = list(args)
            call_kwargs = dict(kwargs)

            def _take(name, position):
                if len(call_args) > position:
                    return call_args[position]
                return call_kwargs.get(name)

            inp = _take("input", 0)
            weight = _take("weight", 1)
            padding = _take("padding", 2)
            stride = _take("stride", 3)
            dilation = _take("dilation", 4)
            groups = _take("groups", 5)

            conv_fn = conv_by_rank.get(inp.dim())
            if conv_fn is None:
                raise
            cpu_out = conv_fn(
                _to_cpu_if_ptpu(inp),
                _to_cpu_if_ptpu(weight),
                bias=None,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            return _to_device_if_tensor(cpu_out, tensor.device)

    torch.cudnn_convolution = cudnn_convolution_with_ptpu_cpu_fallback
    setattr(torch, patched_attr, True)


def _patch_torch_div_floor_trunc_integer_dtype():
    """Force `torch.div(int_tensor, ..., rounding_mode='floor'|'trunc')` to
    return an integer dtype on PTPU.

    PTPU's `aten::div.Tensor` returns float for integer-typed inputs even when
    `rounding_mode` requests integer-style rounding (CPU returns int). This is
    a wrong-dtype quirk, not a NotImplementedError, so it does not fit any of
    the `_should_fallback_to_cpu` helpers above.

    Narrow guard:

    - Wrap only `torch.div`
    - Only divert when `rounding_mode` is `'floor'` / `'trunc'`
    - Only divert when at least one participating operand is a PTPU integer
      (non-floating, non-complex) tensor and every participating operand keeps
      integer floor/trunc semantics
    - True division (`rounding_mode=None`) is left untouched even for int inputs
      (returning float there is the correct PyTorch semantics)
    """
    patched_attr = "_flag_gems_sunrise_div_floor_trunc_dtype_patched"
    if getattr(torch, patched_attr, False):
        return

    original_fn = torch.div

    def _is_integer_like_div_operand(value):
        if isinstance(value, torch.Tensor):
            return not value.is_floating_point() and not value.is_complex()
        return isinstance(value, (bool, int))

    def _find_ptpu_integer_tensor(args, kwargs):
        candidates = []
        if len(args) > 0:
            candidates.append(args[0])
        if len(args) > 1:
            candidates.append(args[1])
        candidates.extend(
            [
                kwargs.get("input"),
                kwargs.get("other"),
                kwargs.get("tensor"),
                kwargs.get("value"),
            ]
        )
        for value in candidates:
            if (
                isinstance(value, torch.Tensor)
                and value.device.type == _PTPU_DEVICE
                and _is_integer_like_div_operand(value)
            ):
                return value
        return None

    @functools.wraps(original_fn)
    def div_with_ptpu_integer_dtype_fix(*args, **kwargs):
        rounding_mode = kwargs.get("rounding_mode")
        if rounding_mode in ("floor", "trunc"):
            if _flag_gems_use_gems_active():
                return original_fn(*args, **kwargs)
            tensor = _find_ptpu_integer_tensor(args, kwargs)
            operands = (
                args[:2]
                if len(args) >= 2
                else (
                    tuple(args)
                    + tuple(
                        value
                        for value in (kwargs.get("input"), kwargs.get("other"))
                        if value is not None
                    )
                )
            )
            if (
                tensor is not None
                and operands
                and all(_is_integer_like_div_operand(value) for value in operands)
            ):
                return _torch_function_cpu_fallback(tensor, args, kwargs, original_fn)
        return original_fn(*args, **kwargs)

    torch.div = div_with_ptpu_integer_dtype_fix
    setattr(torch, patched_attr, True)


def _patch_tensor_to_cpu_for_complex_views():
    """Route complex PTPU view copies to CPU through the base tensor safely.

    Sunrise/PTPU has two related host-copy gaps for complex tensors:

    - conjugate views can segfault on `.cpu()` / `.to('cpu')`
    - sliced / non-contiguous complex views can fail with
      `direct_copy_kernel_ptpu ... failed to dispatch data type ComplexFloat`

    For these cases, copy the root base tensor to CPU first, rebuild the
    original view metadata on CPU with `as_strided`, then reapply lazy conj/neg
    bits on the CPU tensor.
    """
    to_attr = "_flag_gems_sunrise_tensor_to_complex_view_cpu_patched"
    cpu_attr = "_flag_gems_sunrise_tensor_cpu_complex_view_patched"
    if getattr(torch.Tensor, to_attr, False) and getattr(torch.Tensor, cpu_attr, False):
        return

    original_to = torch.Tensor.to
    original_cpu = torch.Tensor.cpu

    def _should_route_through_base(self):
        return (
            isinstance(self, torch.Tensor)
            and self.device.type == _PTPU_DEVICE
            and self.is_complex()
            and (self.is_conj() or self.is_neg() or _has_tensor_base_view(self))
        )

    def _to_targets_cpu(args, kwargs):
        if _is_cpu_device(kwargs.get("device")):
            return True
        if not args:
            return False
        first = args[0]
        if _is_cpu_device(first):
            return True
        if isinstance(first, torch.Tensor):
            return first.device.type == "cpu"
        return False

    def _to_target_dtype(args, kwargs):
        dtype = kwargs.get("dtype")
        if isinstance(dtype, torch.dtype):
            return dtype
        if not args:
            return None
        first = args[0]
        if isinstance(first, torch.dtype):
            return first
        if isinstance(first, torch.Tensor):
            return first.dtype
        return None

    def _rebuild_complex_view_on_cpu(self):
        if self.is_conj():
            cpu_view = _rebuild_complex_view_on_cpu(self.conj()).conj()
            if self.is_neg():
                cpu_view = torch._neg_view(cpu_view)
            return cpu_view

        root = self
        while _has_tensor_base_view(root):
            root = root._base

        cpu_root = original_cpu(root)
        cpu_view = cpu_root
        if root is not self:
            cpu_view = torch.as_strided(
                cpu_root,
                self.size(),
                self.stride(),
                self.storage_offset(),
            )
        if self.is_neg():
            cpu_view = torch._neg_view(cpu_view)
        return cpu_view

    @functools.wraps(original_to)
    def to_with_complex_conj_cpu_route(self, *args, **kwargs):
        if _flag_gems_use_gems_active():
            return original_to(self, *args, **kwargs)
        if _should_route_through_base(self) and _to_targets_cpu(args, kwargs):
            cpu_view = _rebuild_complex_view_on_cpu(self)
            return original_to(cpu_view, *args, **kwargs)
        try:
            return original_to(self, *args, **kwargs)
        except RuntimeError as exc:
            target_dtype = _to_target_dtype(args, kwargs)
            if (
                not _is_ptpu_tensor(self)
                or self.is_complex()
                or not isinstance(target_dtype, torch.dtype)
                or not target_dtype.is_complex
                or "failed to dispatch data type complex" not in str(exc).lower()
            ):
                raise
            cpu_cast = original_to(original_cpu(self), *args, **kwargs)
            return original_to(cpu_cast, device=self.device)

    @functools.wraps(original_cpu)
    def cpu_with_complex_conj_cpu_route(self, *args, **kwargs):
        if _flag_gems_use_gems_active():
            return original_cpu(self, *args, **kwargs)
        if _should_route_through_base(self):
            return _rebuild_complex_view_on_cpu(self)
        return original_cpu(self, *args, **kwargs)

    torch.Tensor.to = to_with_complex_conj_cpu_route
    torch.Tensor.cpu = cpu_with_complex_conj_cpu_route
    setattr(torch.Tensor, to_attr, True)
    setattr(torch.Tensor, cpu_attr, True)


def _patch_complex_tensor_scalar_mul_runtime_error():
    """Fallback complex-tensor scalar mul to CPU on the PTPU runtime quirk.

    Sunrise/PTPU currently fails outside `flag_gems.use_gems()` for:

    - `x * 2.0`
    - `x.mul(2.0)`
    - `torch.mul(x, 2.0)`

    when `x` is a PTPU complex tensor. The failure is a plain `RuntimeError`
    whose text looks like:

        `...BINARY_MUL... failed to dispatch data type ComplexFloat`

    This is not a `NotImplementedError` and does not name an `aten::...`
    symbol, so the generic `_should_fallback_to_cpu(...)` helpers do not fit.

    Narrow guard:

    - only `torch.mul`, `Tensor.mul`, and `Tensor.__mul__`
    - only when the left-hand side is a PTPU complex tensor
    - only when the right-hand side is a non-tensor scalar
    - only on the known runtime error substring
    """
    tensor_mul_attr = "_flag_gems_sunrise_tensor_mul_complex_scalar_patched"
    function_mul_attr = "_flag_gems_sunrise_function_mul_complex_scalar_patched"
    if getattr(torch.Tensor, tensor_mul_attr, False) and getattr(
        torch, function_mul_attr, False
    ):
        return

    quirk_marker = "failed to dispatch data type complex"

    def _should_fallback_complex_scalar_mul(tensor, other):
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.device.type == _PTPU_DEVICE
            and not isinstance(other, torch.Tensor)
            and (tensor.is_complex() or isinstance(other, complex))
        )

    def _ptpu_mul_reference_tensor(*values):
        scalar_complex = any(isinstance(value, complex) for value in values)
        for value in values:
            if not isinstance(value, torch.Tensor) or value.device.type != _PTPU_DEVICE:
                continue
            if value.is_complex() or value.dtype == torch.float64:
                return value
            if scalar_complex:
                return value
        return None

    original_tensor_mul = torch.Tensor.mul
    original_tensor_dunder_mul = torch.Tensor.__mul__
    original_tensor_dunder_rmul = torch.Tensor.__rmul__
    original_function_mul = torch.mul

    @functools.wraps(original_tensor_mul)
    def tensor_mul_with_complex_scalar_cpu_fallback(self, other):
        reference_tensor = _ptpu_mul_reference_tensor(self, other)
        if reference_tensor is not None and not _flag_gems_use_gems_active():
            return original_tensor_mul(
                _to_cpu_if_ptpu(self), _to_cpu_if_ptpu(other)
            ).to(reference_tensor.device)
        try:
            return original_tensor_mul(self, other)
        except RuntimeError as exc:
            if _flag_gems_use_gems_active():
                raise
            if not _should_fallback_complex_scalar_mul(self, other):
                raise
            if quirk_marker not in str(exc).lower():
                raise
            return original_tensor_mul(self.cpu(), other).to(self.device)

    @functools.wraps(original_tensor_dunder_mul)
    def tensor_dunder_mul_with_complex_scalar_cpu_fallback(self, other):
        reference_tensor = _ptpu_mul_reference_tensor(self, other)
        if reference_tensor is not None and not _flag_gems_use_gems_active():
            return original_tensor_dunder_mul(
                _to_cpu_if_ptpu(self), _to_cpu_if_ptpu(other)
            ).to(reference_tensor.device)
        try:
            return original_tensor_dunder_mul(self, other)
        except RuntimeError as exc:
            if _flag_gems_use_gems_active():
                raise
            if not _should_fallback_complex_scalar_mul(self, other):
                raise
            if quirk_marker not in str(exc).lower():
                raise
            return original_tensor_dunder_mul(self.cpu(), other).to(self.device)

    @functools.wraps(original_tensor_dunder_rmul)
    def tensor_dunder_rmul_with_complex_scalar_cpu_fallback(self, other):
        reference_tensor = _ptpu_mul_reference_tensor(self, other)
        if reference_tensor is not None and not _flag_gems_use_gems_active():
            return original_tensor_dunder_rmul(
                _to_cpu_if_ptpu(self), _to_cpu_if_ptpu(other)
            ).to(reference_tensor.device)
        try:
            return original_tensor_dunder_rmul(self, other)
        except RuntimeError as exc:
            if _flag_gems_use_gems_active():
                raise
            if not _should_fallback_complex_scalar_mul(self, other):
                raise
            if quirk_marker not in str(exc).lower():
                raise
            return original_tensor_dunder_rmul(self.cpu(), other).to(self.device)

    @functools.wraps(original_function_mul)
    def function_mul_with_complex_scalar_cpu_fallback(*args, **kwargs):
        tensor = next((arg for arg in args[:2] if isinstance(arg, torch.Tensor)), None)
        if tensor is None:
            tensor = kwargs.get("input")
        if tensor is None:
            tensor = kwargs.get("other")
        other = None
        if len(args) > 1:
            other = args[1] if tensor is args[0] else args[0]
        else:
            other = (
                kwargs.get("other")
                if tensor is kwargs.get("input")
                else kwargs.get("input")
            )
        reference_tensor = _ptpu_mul_reference_tensor(tensor, other)
        if reference_tensor is not None and not _flag_gems_use_gems_active():
            cpu_args = tuple(_to_cpu_if_ptpu(arg) for arg in args)
            cpu_kwargs = {key: _to_cpu_if_ptpu(value) for key, value in kwargs.items()}
            result = original_function_mul(*cpu_args, **cpu_kwargs)
            return _to_device_if_tensor(result, reference_tensor.device)
        try:
            return original_function_mul(*args, **kwargs)
        except RuntimeError as exc:
            if _flag_gems_use_gems_active():
                raise
            if not _should_fallback_complex_scalar_mul(tensor, other):
                raise
            if quirk_marker not in str(exc).lower():
                raise
            cpu_args = tuple(_to_cpu_if_ptpu(arg) for arg in args)
            cpu_kwargs = {key: _to_cpu_if_ptpu(value) for key, value in kwargs.items()}
            result = original_function_mul(*cpu_args, **cpu_kwargs)
            return _to_device_if_tensor(result, tensor.device)

    torch.Tensor.mul = tensor_mul_with_complex_scalar_cpu_fallback
    torch.Tensor.__mul__ = tensor_dunder_mul_with_complex_scalar_cpu_fallback
    torch.Tensor.__rmul__ = tensor_dunder_rmul_with_complex_scalar_cpu_fallback
    torch.mul = function_mul_with_complex_scalar_cpu_fallback
    setattr(torch.Tensor, tensor_mul_attr, True)
    setattr(torch, function_mul_attr, True)


def _patch_complex_tensor_add_runtime_error():
    """Fallback complex add to CPU on the Sunrise/PTPU runtime quirk.

    Outside `flag_gems.use_gems()`, raw complex add can fail with a plain
    runtime error like:

        `...BINARY_ADD... failed to dispatch data type ComplexFloat`

    This typically shows up in reference expressions such as `a + b * alpha`
    inside tests. Keep the guard narrow so the real device add path under
    `use_gems()` remains visible.
    """
    tensor_add_attr = "_flag_gems_sunrise_tensor_add_complex_patched"
    function_add_attr = "_flag_gems_sunrise_function_add_complex_patched"
    if getattr(torch.Tensor, tensor_add_attr, False) and getattr(
        torch, function_add_attr, False
    ):
        return

    quirk_marker = "failed to dispatch data type complex"

    def _first_ptpu_complex_tensor(*values):
        for value in values:
            if (
                isinstance(value, torch.Tensor)
                and value.device.type == _PTPU_DEVICE
                and value.is_complex()
            ):
                return value
        return None

    def _should_route_complex_scalar_add(tensor, other):
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.device.type == _PTPU_DEVICE
            and tensor.is_complex()
            and isinstance(other, complex)
        )

    def _ptpu_add_reference_tensor(*values):
        for value in values:
            if (
                isinstance(value, torch.Tensor)
                and value.device.type == _PTPU_DEVICE
                and (value.is_complex() or value.dtype == torch.float64)
            ):
                return value
        return None

    original_tensor_add = torch.Tensor.add
    original_tensor_dunder_add = torch.Tensor.__add__
    original_function_add = torch.add

    @functools.wraps(original_tensor_add)
    def tensor_add_with_complex_cpu_fallback(self, other, *args, **kwargs):
        reference_tensor = _ptpu_add_reference_tensor(self, other)
        if reference_tensor is not None and not _flag_gems_use_gems_active():
            return original_tensor_add(
                _to_cpu_if_ptpu(self), _to_cpu_if_ptpu(other), *args, **kwargs
            ).to(reference_tensor.device)
        if not _flag_gems_use_gems_active() and _should_route_complex_scalar_add(
            self, other
        ):
            return original_tensor_add(self.cpu(), other, *args, **kwargs).to(
                self.device
            )
        try:
            return original_tensor_add(self, other, *args, **kwargs)
        except RuntimeError as exc:
            if _flag_gems_use_gems_active():
                raise
            tensor = _first_ptpu_complex_tensor(self, other)
            if tensor is None or quirk_marker not in str(exc).lower():
                raise
            cpu_self = _to_cpu_if_ptpu(self)
            cpu_other = _to_cpu_if_ptpu(other)
            result = original_tensor_add(cpu_self, cpu_other, *args, **kwargs)
            return _to_device_if_tensor(result, tensor.device)

    @functools.wraps(original_tensor_dunder_add)
    def tensor_dunder_add_with_complex_cpu_fallback(self, other):
        reference_tensor = _ptpu_add_reference_tensor(self, other)
        if reference_tensor is not None and not _flag_gems_use_gems_active():
            return original_tensor_dunder_add(
                _to_cpu_if_ptpu(self), _to_cpu_if_ptpu(other)
            ).to(reference_tensor.device)
        if not _flag_gems_use_gems_active() and _should_route_complex_scalar_add(
            self, other
        ):
            return original_tensor_dunder_add(self.cpu(), other).to(self.device)
        try:
            return original_tensor_dunder_add(self, other)
        except RuntimeError as exc:
            if _flag_gems_use_gems_active():
                raise
            tensor = _first_ptpu_complex_tensor(self, other)
            if tensor is None or quirk_marker not in str(exc).lower():
                raise
            cpu_self = _to_cpu_if_ptpu(self)
            cpu_other = _to_cpu_if_ptpu(other)
            result = original_tensor_dunder_add(cpu_self, cpu_other)
            return _to_device_if_tensor(result, tensor.device)

    @functools.wraps(original_function_add)
    def function_add_with_complex_cpu_fallback(*args, **kwargs):
        tensor = _first_ptpu_complex_tensor(
            *(
                args[:2]
                if len(args) >= 2
                else (kwargs.get("input"), kwargs.get("other"))
            )
        )
        other = args[1] if len(args) > 1 else kwargs.get("other")
        reference_tensor = _ptpu_add_reference_tensor(
            *(
                args[:2]
                if len(args) >= 2
                else (kwargs.get("input"), kwargs.get("other"))
            )
        )
        if reference_tensor is not None and not _flag_gems_use_gems_active():
            cpu_args = tuple(_to_cpu_if_ptpu(arg) for arg in args)
            cpu_kwargs = {key: _to_cpu_if_ptpu(value) for key, value in kwargs.items()}
            result = original_function_add(*cpu_args, **cpu_kwargs)
            return _to_device_if_tensor(result, reference_tensor.device)
        if not _flag_gems_use_gems_active() and _should_route_complex_scalar_add(
            tensor, other
        ):
            cpu_args = tuple(_to_cpu_if_ptpu(arg) for arg in args)
            cpu_kwargs = {key: _to_cpu_if_ptpu(value) for key, value in kwargs.items()}
            result = original_function_add(*cpu_args, **cpu_kwargs)
            return _to_device_if_tensor(result, tensor.device)
        try:
            return original_function_add(*args, **kwargs)
        except RuntimeError as exc:
            if _flag_gems_use_gems_active():
                raise
            if tensor is None or quirk_marker not in str(exc).lower():
                raise
            cpu_args = tuple(_to_cpu_if_ptpu(arg) for arg in args)
            cpu_kwargs = {key: _to_cpu_if_ptpu(value) for key, value in kwargs.items()}
            result = original_function_add(*cpu_args, **cpu_kwargs)
            return _to_device_if_tensor(result, tensor.device)

    torch.Tensor.add = tensor_add_with_complex_cpu_fallback
    torch.Tensor.__add__ = tensor_dunder_add_with_complex_cpu_fallback
    torch.add = function_add_with_complex_cpu_fallback
    setattr(torch.Tensor, tensor_add_attr, True)
    setattr(torch, function_add_attr, True)


def _patch_torch_isclose_allclose_complex_dtype():
    """Fallback `torch.isclose` / `torch.allclose` for PTPU complex/fp64 tensors.

    `torch.testing.assert_close(...)` on Sunrise/PTPU complex tensors reaches
    `torch.isclose(...)`, which can raise:

        `RuntimeError: unsupported scalar type: ComplexFloat`

    This is a plain runtime quirk outside `flag_gems.use_gems()`, not an
    `aten::...`-tagged `NotImplementedError`, so the normal helper path does
    not catch it.

    Narrow guard:

    - only `torch.isclose` and `torch.allclose`
    - only when the first argument is a PTPU complex/fp64 tensor
    - only on the known runtime error substring for the complex case
    """
    patched_attr = "_flag_gems_sunrise_isclose_allclose_complex_dtype_patched"
    if getattr(torch, patched_attr, False):
        return

    quirk_marker = "unsupported scalar type: complex"
    original_isclose = torch.isclose
    original_allclose = torch.allclose

    def _should_fallback_compare(tensor):
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.device.type == _PTPU_DEVICE
            and (tensor.is_complex() or tensor.dtype == torch.float64)
        )

    @functools.wraps(original_isclose)
    def isclose_with_complex_cpu_fallback(*args, **kwargs):
        tensor = args[0] if args else kwargs.get("input")
        if not _flag_gems_use_gems_active() and _should_fallback_compare(tensor):
            cpu_args = tuple(_to_cpu_if_ptpu(arg) for arg in args)
            cpu_kwargs = {key: _to_cpu_if_ptpu(value) for key, value in kwargs.items()}
            result = original_isclose(*cpu_args, **cpu_kwargs)
            return _to_device_if_tensor(result, tensor.device)
        try:
            return original_isclose(*args, **kwargs)
        except RuntimeError as exc:
            if _flag_gems_use_gems_active():
                raise
            if not _should_fallback_compare(tensor):
                raise
            if tensor.is_complex() and quirk_marker not in str(exc).lower():
                raise
            cpu_args = tuple(_to_cpu_if_ptpu(arg) for arg in args)
            cpu_kwargs = {key: _to_cpu_if_ptpu(value) for key, value in kwargs.items()}
            result = original_isclose(*cpu_args, **cpu_kwargs)
            return _to_device_if_tensor(result, tensor.device)

    @functools.wraps(original_allclose)
    def allclose_with_complex_cpu_fallback(*args, **kwargs):
        tensor = args[0] if args else kwargs.get("input")
        if not _flag_gems_use_gems_active() and _should_fallback_compare(tensor):
            cpu_args = tuple(_to_cpu_if_ptpu(arg) for arg in args)
            cpu_kwargs = {key: _to_cpu_if_ptpu(value) for key, value in kwargs.items()}
            return original_allclose(*cpu_args, **cpu_kwargs)
        try:
            return original_allclose(*args, **kwargs)
        except RuntimeError as exc:
            if _flag_gems_use_gems_active():
                raise
            if not _should_fallback_compare(tensor):
                raise
            if tensor.is_complex() and quirk_marker not in str(exc).lower():
                raise
            cpu_args = tuple(_to_cpu_if_ptpu(arg) for arg in args)
            cpu_kwargs = {key: _to_cpu_if_ptpu(value) for key, value in kwargs.items()}
            return original_allclose(*cpu_args, **cpu_kwargs)

    torch.isclose = isclose_with_complex_cpu_fallback
    torch.allclose = allclose_with_complex_cpu_fallback
    setattr(torch, patched_attr, True)


def _patch_complex_matmul_runtime_error():
    """Fallback reference matmul-family ops to CPU on Sunrise/PTPU.

    Complex reconstruction paths such as `u @ diag(s) @ v.mH` can fail outside
    `flag_gems.use_gems()` with runtime errors from lowerings like:

    - `addbmm_out not implemented for ComplexFloat`
    - `baddbmm_out only supports float/half/bfloat16, got ComplexFloat`

    Separately, some real-valued *degenerate batched matmuls* such as
    `(..., 17, 1) @ (..., 1, 1)` can silently produce garbage on PTPU in the
    same reference-style reconstruction path, even though the upstream SVD
    factors themselves are correct. Route those narrow cases to CPU too.

    This is a reference-path/runtime gap rather than a FlagGems kernel bug.
    Keep the guard tight:

    - only outside `flag_gems.use_gems()`
    - always for PTPU complex/fp64 tensors
    - additionally for PTPU real floating batched matmuls where at least one
      tensor has a singleton matrix dimension (`min(shape[-2:]) == 1`)
    - wrap matmul-family entry points that the Python surface can hit during
      reconstruction: `Tensor.__matmul__`, `Tensor.matmul`, `torch.matmul`,
      `torch.bmm`, `torch.addbmm`, `torch.baddbmm`
    """
    tensor_attr = "_flag_gems_sunrise_tensor_matmul_complex_patched"
    function_attr = "_flag_gems_sunrise_function_matmul_complex_patched"
    if getattr(torch.Tensor, tensor_attr, False) and getattr(
        torch, function_attr, False
    ):
        return

    quirk_markers = (
        "addbmm_out not implemented for complex",
        "baddbmm_out only supports float/half/bfloat16, got complex",
        "unsupported scalar type: complex",
    )

    def _ptpu_matmul_reference_tensor(*values):
        first_ptpu_tensor = None
        for value in values:
            if not isinstance(value, torch.Tensor) or value.device.type != _PTPU_DEVICE:
                continue
            if first_ptpu_tensor is None:
                first_ptpu_tensor = value
            if value.is_complex() or value.dtype == torch.float64:
                return value
        return first_ptpu_tensor

    def _ptpu_tensor_args(*values):
        return [
            value
            for value in values
            if isinstance(value, torch.Tensor) and value.device.type == _PTPU_DEVICE
        ]

    def _should_route_reference_matmul(*values):
        tensors = _ptpu_tensor_args(*values)
        if not tensors:
            return False
        if any(t.is_complex() or t.dtype == torch.float64 for t in tensors):
            return True
        return any(
            t.ndim >= 3
            and t.is_floating_point()
            and not t.is_complex()
            and min(t.shape[-2:]) == 1
            for t in tensors
        )

    def _cpu_dispatch_to_reference_device(reference_tensor, original_fn, args, kwargs):
        cpu_args = tuple(_to_cpu_if_ptpu(arg) for arg in args)
        cpu_kwargs = {key: _to_cpu_if_ptpu(value) for key, value in kwargs.items()}
        result = original_fn(*cpu_args, **cpu_kwargs)
        out = kwargs.get("out")
        return _finalize_cpu_result(result, out, reference_tensor.device)

    original_tensor_matmul = torch.Tensor.matmul
    original_tensor_dunder_matmul = torch.Tensor.__matmul__
    original_function_matmul = torch.matmul
    original_function_bmm = torch.bmm
    original_function_addbmm = torch.addbmm
    original_function_baddbmm = torch.baddbmm

    @functools.wraps(original_tensor_matmul)
    def tensor_matmul_with_complex_cpu_fallback(self, other):
        reference_tensor = _ptpu_matmul_reference_tensor(self, other)
        if not _flag_gems_use_gems_active() and _should_route_reference_matmul(
            self, other
        ):
            return _cpu_dispatch_to_reference_device(
                reference_tensor, original_tensor_matmul, (self, other), {}
            )
        try:
            return original_tensor_matmul(self, other)
        except RuntimeError as exc:
            if _flag_gems_use_gems_active():
                raise
            if reference_tensor is None or not any(
                marker in str(exc).lower() for marker in quirk_markers
            ):
                raise
            return _cpu_dispatch_to_reference_device(
                reference_tensor, original_tensor_matmul, (self, other), {}
            )

    @functools.wraps(original_tensor_dunder_matmul)
    def tensor_dunder_matmul_with_complex_cpu_fallback(self, other):
        reference_tensor = _ptpu_matmul_reference_tensor(self, other)
        if not _flag_gems_use_gems_active() and _should_route_reference_matmul(
            self, other
        ):
            return _cpu_dispatch_to_reference_device(
                reference_tensor, original_tensor_dunder_matmul, (self, other), {}
            )
        try:
            return original_tensor_dunder_matmul(self, other)
        except RuntimeError as exc:
            if _flag_gems_use_gems_active():
                raise
            if reference_tensor is None or not any(
                marker in str(exc).lower() for marker in quirk_markers
            ):
                raise
            return _cpu_dispatch_to_reference_device(
                reference_tensor, original_tensor_dunder_matmul, (self, other), {}
            )

    def _patch_torch_matmul_like(name, original_fn):
        @functools.wraps(original_fn)
        def fn_with_complex_cpu_fallback(*args, **kwargs):
            reference_tensor = _ptpu_matmul_reference_tensor(*args, *kwargs.values())
            if not _flag_gems_use_gems_active() and _should_route_reference_matmul(
                *args, *kwargs.values()
            ):
                return _cpu_dispatch_to_reference_device(
                    reference_tensor, original_fn, args, kwargs
                )
            try:
                return original_fn(*args, **kwargs)
            except RuntimeError as exc:
                if _flag_gems_use_gems_active():
                    raise
                if reference_tensor is None or not any(
                    marker in str(exc).lower() for marker in quirk_markers
                ):
                    raise
                return _cpu_dispatch_to_reference_device(
                    reference_tensor, original_fn, args, kwargs
                )

        setattr(torch, name, fn_with_complex_cpu_fallback)

    torch.Tensor.matmul = tensor_matmul_with_complex_cpu_fallback
    torch.Tensor.__matmul__ = tensor_dunder_matmul_with_complex_cpu_fallback
    _patch_torch_matmul_like("matmul", original_function_matmul)
    _patch_torch_matmul_like("bmm", original_function_bmm)
    _patch_torch_matmul_like("addbmm", original_function_addbmm)
    _patch_torch_matmul_like("baddbmm", original_function_baddbmm)
    setattr(torch.Tensor, tensor_attr, True)
    setattr(torch, function_attr, True)


def _flag_gems_use_gems_active():
    """Return True while a `flag_gems.use_gems()` context is active.

    `use_gems()` sets the module-level `current_work_registrar` on enter and
    `del`s it on exit, so `getattr(flag_gems, "current_work_registrar", None)`
    is a reliable, side-effect-free signal for "are we currently dispatching
    aten ops through FlagGems device kernels?".
    """
    import flag_gems

    return getattr(flag_gems, "current_work_registrar", None) is not None


def _patch_torch_einsum_low_precision_reference():
    """Compute low-precision `torch.einsum(...)` reference matmuls on CPU.

    This is a precision quirk, not a `NotImplementedError`. `torch.einsum`
    lowers its contraction to a matmul/bmm. On Sunrise/PTPU the fp16 / bf16
    matmul accumulates in low precision, while CPU (and CUDA) accumulate fp16
    matmuls internally in fp32. Tests such as `test_flash_attn_varlen_func.py`
    build their CPU "golden" reference with raw `torch.einsum("hqk,khd->qhd",
    attn, v)` on tensors that happen to live on PTPU (the test wraps setup in
    `with torch.device("ptpu")` and never routes the reference through
    `accuracy_utils.to_reference()`), so the *reference itself* drifts by up to
    ~0.5 versus the true CPU result and the assertion fails even though the
    Sunrise flash-attention kernel under test is correct (~1e-3).

    The fix mirrors the "wrong ref operator → CPU" rule: redirect only the
    reference-path einsum to CPU. The guard is intentionally tight so that the
    real device-under-test einsum (`test_einsum.py`, `test_fp8_einsum.py`, ...)
    is never diverted:

    - Skip entirely while `flag_gems.use_gems()` is active. The device path in
      `test_einsum.py` runs einsum under `use_gems()`; the reference paths do
      not. (No FlagGems op implementation calls `torch.einsum`, so this never
      touches kernel internals.)
    - Only divert when at least one operand is a PTPU tensor.
    - Only divert when the contraction dtype is fp16 / bf16. fp32 / fp64
      references (e.g. `to_reference(.., upcast=True)`, `q.float()`) already
      match CPU and are left on device.
    - Equivalent to upcasting the einsum to fp32 on device, but computing on
      CPU keeps the reference identical to a `--ref cpu` golden value.
    """
    patched_attr = "_flag_gems_sunrise_einsum_low_precision_patched"
    if getattr(torch, patched_attr, False):
        return

    original_fn = torch.einsum
    low_precision_dtypes = (torch.float16, torch.bfloat16)

    def _operand_tensors(operands):
        # torch.einsum accepts either (equation, *tensors) or
        # (equation, [tensors]); flatten the sublist form too.
        for operand in operands:
            if isinstance(operand, torch.Tensor):
                yield operand
            elif isinstance(operand, (list, tuple)):
                for item in operand:
                    if isinstance(item, torch.Tensor):
                        yield item

    @functools.wraps(original_fn)
    def einsum_with_ptpu_low_precision_cpu_reference(equation, *operands):
        if not _flag_gems_use_gems_active():
            tensors = list(_operand_tensors(operands))
            if any(_is_ptpu_tensor(t) for t in tensors) and any(
                t.dtype in low_precision_dtypes for t in tensors
            ):
                cpu_operands = tuple(
                    (
                        _to_cpu_if_ptpu(operand)
                        if isinstance(operand, torch.Tensor)
                        else (
                            [_to_cpu_if_ptpu(item) for item in operand]
                            if isinstance(operand, (list, tuple))
                            else operand
                        )
                    )
                    for operand in operands
                )
                device = next((t.device for t in tensors if _is_ptpu_tensor(t)), None)
                result = original_fn(equation, *cpu_operands)
                return _to_device_if_tensor(result, device)
        return original_fn(equation, *operands)

    torch.einsum = einsum_with_ptpu_low_precision_cpu_reference
    setattr(torch, patched_attr, True)


def _patch_bool_sum_cpu_reference():
    """Compute PTPU bool-tensor `sum` reductions on CPU outside `use_gems()`.

    Sunrise/PTPU occasionally returns the wrong population count for boolean
    masks in test-setup code such as `numel = mask.sum().item()`. This is a
    silent semantic quirk rather than a `NotImplementedError`, so we cannot
    rely on the normal exception-driven CPU fallback helpers.

    Keep the guard intentionally tight:

    - only `torch.Tensor.sum` / `torch.sum`
    - only when the input tensor is a PTPU bool tensor
    - only outside `flag_gems.use_gems()`, so the real reduction kernels under
      test are still exercised inside the device path
    """
    tensor_attr = "_flag_gems_sunrise_tensor_bool_sum_cpu_patched"
    function_attr = "_flag_gems_sunrise_function_bool_sum_cpu_patched"
    if getattr(torch.Tensor, tensor_attr, False) and getattr(
        torch, function_attr, False
    ):
        return

    original_tensor_sum = torch.Tensor.sum
    original_function_sum = torch.sum

    def _should_route_bool_sum(tensor):
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.device.type == _PTPU_DEVICE
            and tensor.dtype == torch.bool
        )

    @functools.wraps(original_tensor_sum)
    def tensor_sum_with_bool_cpu_fallback(self, *args, **kwargs):
        if not _flag_gems_use_gems_active() and _should_route_bool_sum(self):
            return _cpu_fallback(self, args, kwargs, original_tensor_sum)
        return original_tensor_sum(self, *args, **kwargs)

    @functools.wraps(original_function_sum)
    def function_sum_with_bool_cpu_fallback(*args, **kwargs):
        tensor = args[0] if args else kwargs.get("input")
        if not _flag_gems_use_gems_active() and _should_route_bool_sum(tensor):
            return _torch_function_cpu_fallback(
                tensor, args, kwargs, original_function_sum
            )
        return original_function_sum(*args, **kwargs)

    torch.Tensor.sum = tensor_sum_with_bool_cpu_fallback
    torch.sum = function_sum_with_bool_cpu_fallback
    setattr(torch.Tensor, tensor_attr, True)
    setattr(torch, function_attr, True)


def _patch_torch_nn_functional_one_hot_cpu_reference():
    """Compute `torch.nn.functional.one_hot(...)` on CPU for PTPU inputs.

    Tests such as `test_multinomial.py` build reference counts with
    `torch.nn.functional.one_hot(...)` directly on tensors that may live on
    PTPU. Route only that reference-style path to CPU:

    - only `torch.nn.functional.one_hot`
    - only when the input tensor is on PTPU
    - only outside `flag_gems.use_gems()`, so the real backend one_hot path
      remains available inside the device-under-test region
    """
    patched_attr = "_flag_gems_sunrise_nn_functional_one_hot_cpu_patched"
    if getattr(F, patched_attr, False):
        return

    original_fn = F.one_hot

    @functools.wraps(original_fn)
    def one_hot_with_ptpu_cpu_reference(*args, **kwargs):
        tensor = args[0] if args else kwargs.get("tensor") or kwargs.get("input")
        if not _flag_gems_use_gems_active() and _is_ptpu_tensor(tensor):
            return _torch_function_cpu_fallback(tensor, args, kwargs, original_fn)
        return original_fn(*args, **kwargs)

    F.one_hot = one_hot_with_ptpu_cpu_reference
    setattr(F, patched_attr, True)


def _patch_torch_packet(packet_name, aten_op):
    packet = getattr(torch.ops.aten, packet_name)
    patched_attr = "_flag_gems_sunrise_packet_patched"
    if getattr(packet, patched_attr, False):
        return

    original_fn = packet._op

    @functools.wraps(original_fn)
    def packet_with_ptpu_cpu_fallback(*args, **kwargs):
        tensor = args[0] if args else kwargs.get("self") or kwargs.get("input")
        try:
            return original_fn(*args, **kwargs)
        except NotImplementedError as exc:
            if _flag_gems_use_gems_active():
                raise
            if not _should_fallback_to_cpu(exc, tensor, aten_op):
                raise
            return _torch_function_cpu_fallback(tensor, args, kwargs, original_fn)

    packet._op = packet_with_ptpu_cpu_fallback
    setattr(packet, patched_attr, True)


def _patch_torch_ptpu_get_device_index():
    """Work around torch_ptpu's `_get_device_index()` choking on an index-less
    `torch.device('ptpu')`: `device.index` is None, so the trailing
    `device >= 0` raises `TypeError: '>=' not supported between NoneType and
    int`. flag_gems constructor/RNG ops pass exactly such a device into
    `torch_device_fn.device(device)` under `use_gems()`. Coerce a None index to
    `current_device()`. Every torch_ptpu device helper and the device guard
    resolve `_get_device_index` from the `torch_ptpu.ptpu` module globals, so
    rebinding it there fixes them all.
    """
    try:
        import torch_ptpu.ptpu as _ptpu
    except Exception:
        return

    if getattr(_ptpu, "_flag_gems_sunrise_gdi_patched", False):
        return

    original_fn = getattr(_ptpu, "_get_device_index", None)
    if original_fn is None:
        return

    @functools.wraps(original_fn)
    def get_device_index_with_index_fallback(device):
        if isinstance(device, torch.device) and device.index is None:
            device = _ptpu.current_device()
        return original_fn(device)

    _ptpu._get_device_index = get_device_index_with_index_fallback
    _ptpu._flag_gems_sunrise_gdi_patched = True


def _pytest_terminal_summary_frame():
    for frame_info in inspect.stack(context=0):
        frame_path = os.path.normpath(frame_info.filename)
        if frame_info.function == "pytest_terminal_summary" and frame_path.endswith(
            os.path.join("tests", "conftest.py")
        ):
            return frame_info
    return None


def _backup_corrupt_accuracy_report(frame_info, payload):
    if not payload:
        return None
    frame = frame_info.frame
    json_file = frame.f_locals.get("json_file")
    report_path = getattr(json_file, "name", None) or frame.f_globals.get("REPORT_FILE")
    if not report_path:
        return None
    report_path = os.path.abspath(os.fspath(report_path))
    backup_path = (
        f"{report_path}.corrupt." f"{os.getpid()}." f"{int(time.time() * 1000)}"
    )
    with open(backup_path, "w", encoding="utf-8") as backup_file:
        backup_file.write(payload)
    return backup_path


def _sanitize_accuracy_report_json(value):
    if isinstance(value, torch.Tensor):
        return (
            {
                "__tensor__": True,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "device": str(value.device),
                "requires_grad": bool(value.requires_grad),
            },
            1,
        )
    if isinstance(value, dict):
        sanitized = {}
        replaced = 0
        for key, item in value.items():
            if isinstance(key, (str, int, float, bool)) or key is None:
                safe_key = key
            else:
                safe_key = str(key)
            safe_item, item_replaced = _sanitize_accuracy_report_json(item)
            sanitized[safe_key] = safe_item
            replaced += item_replaced
        return sanitized, replaced
    if isinstance(value, (list, tuple)):
        sanitized = []
        replaced = 0
        for item in value:
            safe_item, item_replaced = _sanitize_accuracy_report_json(item)
            sanitized.append(safe_item)
            replaced += item_replaced
        return sanitized, replaced
    if isinstance(value, (set, frozenset)):
        sanitized = []
        replaced = 0
        for item in value:
            safe_item, item_replaced = _sanitize_accuracy_report_json(item)
            sanitized.append(safe_item)
            replaced += item_replaced
        return sanitized, replaced
    return value, 0


def _patch_json_loads_for_accuracy_result():
    """Ignore a truncated `accuracy_result.json` in test summary on Sunrise.

    Some CI jobs finish all pytest cases successfully, then fail in
    `tests/conftest.py::pytest_terminal_summary` while merging the accumulated
    `accuracy_result.json`. The failure is a plain `json.JSONDecodeError` on a
    previously truncated file, so keep the fix narrow and Sunrise-local:

    - patch only `json.loads`
    - only intercept `json.JSONDecodeError`
    - only when the caller is `tests/conftest.py::pytest_terminal_summary`
    - backup the corrupt payload before falling back to `{}`
    """
    patched_attr = "_flag_gems_sunrise_accuracy_json_loads_patched"
    if getattr(json, patched_attr, False):
        return

    original_fn = json.loads

    @functools.wraps(original_fn)
    def loads_with_accuracy_result_fallback(*args, **kwargs):
        try:
            return original_fn(*args, **kwargs)
        except json.JSONDecodeError:
            frame_info = _pytest_terminal_summary_frame()
            if frame_info is None:
                raise
            payload = args[0] if args else kwargs.get("s")
            backup_path = None
            try:
                backup_path = _backup_corrupt_accuracy_report(frame_info, payload)
            except OSError as backup_exc:
                _LOGGER.warning(
                    "GEMS_SUNRISE skipped corrupt accuracy_result backup: %s",
                    backup_exc,
                )
            if backup_path is not None:
                _LOGGER.warning(
                    "GEMS_SUNRISE ignored corrupt accuracy_result JSON and backed it up to %s",
                    backup_path,
                )
            else:
                _LOGGER.warning("GEMS_SUNRISE ignored corrupt accuracy_result JSON")
            return {}

    json.loads = loads_with_accuracy_result_fallback
    setattr(json, patched_attr, True)


def _patch_json_dump_for_accuracy_result():
    """Sanitize tensor payloads before pytest summary writes JSON on Sunrise."""
    patched_attr = "_flag_gems_sunrise_accuracy_json_dump_patched"
    if getattr(json, patched_attr, False):
        return

    original_fn = json.dump

    @functools.wraps(original_fn)
    def dump_with_accuracy_result_sanitize(*args, **kwargs):
        frame_info = _pytest_terminal_summary_frame()
        if frame_info is None or not args:
            return original_fn(*args, **kwargs)
        payload = args[0]
        safe_payload, replaced = _sanitize_accuracy_report_json(payload)
        if replaced:
            _LOGGER.warning(
                "GEMS_SUNRISE sanitized %d tensor value(s) before writing accuracy_result JSON",
                replaced,
            )
            args = (safe_payload, *args[1:])
        return original_fn(*args, **kwargs)

    json.dump = dump_with_accuracy_result_sanitize
    setattr(json, patched_attr, True)


def apply_sunrise_monkey_patches():
    _patch_torch_ptpu_get_device_index()
    _patch_json_loads_for_accuracy_result()
    _patch_json_dump_for_accuracy_result()
    _patch_tensor_copy_scalar_fill_fallback()
    # triu
    _patch_tensor_method("triu", "aten::triu.out")
    _patch_tensor_method("triu_", "aten::triu.out", inplace=True)
    _patch_torch_function("triu", "aten::triu.out")

    # tanh
    _patch_tensor_method("tanh", "aten::tanh.out")
    _patch_tensor_method("tanh_", "aten::tanh.out", inplace=True)
    _patch_torch_function("tanh", "aten::tanh.out")

    # relu
    _patch_tensor_method("relu", "aten::relu")
    _patch_tensor_method("relu_", "aten::relu", inplace=True)
    _patch_torch_function("relu", "aten::relu")

    # clamp_min
    _patch_tensor_method("clamp_min", "aten::clamp_min")
    _patch_tensor_method("clamp_min_", "aten::clamp_min", inplace=True)
    _patch_torch_function("clamp_min", "aten::clamp_min")
    _patch_torch_function("clamp_min_", "aten::clamp_min", inplace=True)
    _patch_torch_tensor_out("clamp_min", "aten::clamp_min.Tensor_out")

    # remainder / mod
    _patch_tensor_method("__mod__", "aten::remainder")
    _patch_tensor_method("remainder", "aten::remainder")
    _patch_tensor_method("remainder_", "aten::remainder", inplace=True)
    _patch_torch_function("remainder", "aten::remainder")
    _patch_torch_tensor_out("remainder", "aten::remainder.Tensor_out")

    # floor_divide
    _patch_tensor_method("__floordiv__", "aten::floor_divide")
    _patch_tensor_method("floor_divide", "aten::floor_divide")
    _patch_tensor_method("floor_divide_", "aten::floor_divide", inplace=True)
    _patch_torch_function("floor_divide", "aten::floor_divide")
    _patch_bool_sum_cpu_reference()

    # reductions used in tests
    _patch_torch_function("amin", "aten::amin")
    _patch_torch_function("amax", "aten::amax")
    _patch_tensor_method("min", "aten::min")
    _patch_torch_function("min", "aten::min")
    _patch_tensor_method("median", "aten::median")
    _patch_torch_function("median", "aten::median")
    _patch_tensor_method("amax", "aten::amax.out")
    _patch_tensor_method("logsumexp", "aten::amax.out")
    _patch_torch_function("logsumexp", "aten::amax.out")
    _patch_tensor_method("mean", "aten::mean")
    _patch_torch_function("mean", "aten::mean")
    _patch_torch_function("norm", "aten::linalg_vector_norm.out")
    _patch_torch_linalg_function("vector_norm", "aten::linalg_vector_norm.out")
    _patch_torch_linalg_function("qr", "aten::linalg_qr.out")
    _patch_torch_function("unique_consecutive", "aten::unique_consecutive")

    # misc test helpers
    _patch_tensor_method("__invert__", "aten::bitwise_not.out")
    _patch_tensor_method("bitwise_not", "aten::bitwise_not.out")
    _patch_tensor_method("bitwise_not_", "aten::bitwise_not.out", inplace=True)
    _patch_torch_function("bitwise_not", "aten::bitwise_not.out")
    _patch_tensor_method("__and__", "aten::bitwise_and")
    _patch_tensor_method("bitwise_and", "aten::bitwise_and")
    _patch_tensor_method("bitwise_and_", "aten::bitwise_and", inplace=True)
    _patch_torch_function("bitwise_and", "aten::bitwise_and")
    _patch_torch_tensor_out("bitwise_and", "aten::bitwise_and.Tensor_out")
    _patch_tensor_method("masked_select", "aten::masked_select")
    _patch_torch_function("masked_select", "aten::masked_select")
    _patch_tensor_method("__or__", "aten::bitwise_or")
    _patch_torch_function("bitwise_or", "aten::bitwise_or")
    _patch_torch_tensor_out("bitwise_or", "aten::bitwise_or.Tensor_out")
    _patch_torch_function("isclose", "aten::bitwise_and.Tensor_out")
    _patch_torch_function("allclose", "aten::bitwise_and.Tensor_out")
    _patch_torch_function("complex", "aten::complex.out")
    _patch_torch_creation_function("eye", "aten::eye.m_out")
    _patch_torch_creation_function("linspace", "aten::linspace.out")
    _patch_torch_out("hypot", "aten::hypot.out")
    _patch_torch_creation_function("randperm", "aten::randperm.generator_out")
    _patch_tensor_property("real", "aten::view_as_real")
    _patch_tensor_property("imag", "aten::view_as_real")
    _patch_torch_nn_functional("pad", "aten::replication_pad3d.out")
    _patch_torch_nn_functional("logsigmoid", "aten::log_sigmoid_forward")
    _patch_torch_nn_functional_one_hot_cpu_reference()
    _patch_torch_randn_complex_dtype()
    _patch_torch_cudnn_convolution()
    _patch_torch_div_floor_trunc_integer_dtype()
    _patch_tensor_to_cpu_for_complex_views()
    _patch_complex_tensor_scalar_mul_runtime_error()
    _patch_complex_tensor_add_runtime_error()
    _patch_complex_tensor_add_runtime_error()
    _patch_complex_matmul_runtime_error()
    _patch_torch_isclose_allclose_complex_dtype()
    _patch_torch_einsum_low_precision_reference()
    _patch_torch_packet("elu", "aten::elu.out")
