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

import os

import torch_ptpu  # noqa: F401
from backend_utils import VendorDescriptor

from .monkey_patch import apply_sunrise_monkey_patches

vendor_info = VendorDescriptor(
    vendor_name="sunrise",
    device_name="ptpu",
    device_query_cmd="pt_smi",
    triton_extra_name="tang",
    dispatch_key="PrivateUse1",
    fp64_enabled=False,
    bf16_enabled=False,
    int64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = ()


def _sunrise_rebuild_ptpu_tensor_from_cpu(
    tensor_cls, cpu_tensor, device, requires_grad
):
    import torch
    from torch.nn.parameter import Parameter

    tensor = cpu_tensor.to(device=device)
    if tensor_cls == Parameter:
        return Parameter(tensor, requires_grad=requires_grad)
    if tensor_cls not in (torch.Tensor, type(tensor)):
        try:
            tensor = tensor.as_subclass(tensor_cls)
        except Exception:
            pass
    tensor.requires_grad = requires_grad
    return tensor


def _should_stage_ptpu_tensor_for_multiprocessing(tensor):
    import torch

    return (
        isinstance(tensor, torch.Tensor)
        and tensor.device.type == "ptpu"
        and tensor.layout == torch.strided
        and not tensor.is_nested
    )


def _sunrise_monkey_patch_enabled():
    value = os.getenv("FLAG_GEMS_SUNRISE_ENABLE_MONKEY_PATCH", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


if _sunrise_monkey_patch_enabled():
    apply_sunrise_monkey_patches()

# [Sunrise fix] Aten lower needed.
CUSTOMIZED_AUTOGRAD_OPS = (
    "absolute",
    "arcsinh_",
    "arcsinh",
    "arcsinh.out",
    "arctanh_",
    "clip",
    "clip_",
    "concatenate",
    "conj_physical",
    "diag",
    "diff",
    "__ior__.Tensor",
    "__ior__.Scalar",
    "__or__.Tensor",
    "__or__.Scalar",
    "embedding_backward",
    "feature_dropout",
    "feature_dropout_",
    "gather_backward",
    "greater.Tensor",
    "greater.Scalar",
    "greater.Scalar_out",
    "hstack",
    "isclose",
    "isfinite",
    "kron",
    "log_sigmoid",
    "margin_ranking_loss",
    "nonzero_numpy",
    "pad",
    "prelu",
    "quantile",
    "relu6",
    "repeat_interleave.self_int",
    "repeat_interleave.self_Tensor",
    "resolve_conj",
    "resolve_neg",
    "selu",
    "selu_",
    "square",
    "square_",
    "square.out",
    "svd",
    "tile",
    "vstack",
)


def _sunrise_extra_config_entries():  # 有些公共库也没有注册的op，只能先放在这里了。使得tests能过
    from .ops import (
        amax_out,
        amin,
        amin_out,
        aminmax_out,
        clamp_min,
        clamp_min_,
        clamp_min_out,
        hypot_out,
    )

    return (
        ("amax.out", amax_out),
        ("amin", amin),
        ("amin.out", amin_out),
        ("aminmax.out", aminmax_out),
        ("clamp_min.Tensor", clamp_min),
        ("clamp_min.Tensor_out", clamp_min_out),
        ("clamp_min_.Tensor", clamp_min_),
        ("hypot.out", hypot_out),
    )


def _install_autograd_dispatch_patch():
    import torch

    from flag_gems.runtime.op_registrar import GeneralOpRegistrar

    register_cls = GeneralOpRegistrar

    if getattr(register_cls, "_sunrise_autograd_dispatch_patched", False):
        return

    original_register_impl = register_cls.register_impl
    autograd_key = torch._C.DispatchKey.Autograd.name
    autograd_ops = frozenset(CUSTOMIZED_AUTOGRAD_OPS)

    def register_impl(self, key, fn, extra_dispatch_keys=()):
        if self.device.vendor_name == vendor_info.vendor_name and key in autograd_ops:
            all_dispatch_keys = list(extra_dispatch_keys)
            if autograd_key not in all_dispatch_keys:
                all_dispatch_keys.append(autograd_key)
            extra_dispatch_keys = tuple(all_dispatch_keys)

        return original_register_impl(self, key, fn, extra_dispatch_keys)

    register_cls.register_impl = register_impl
    register_cls._sunrise_autograd_dispatch_patched = True
    register_cls._sunrise_original_register_impl = original_register_impl


def _install_register_config_patch():
    from flag_gems.runtime.op_registrar import GeneralOpRegistrar

    register_cls = GeneralOpRegistrar

    if getattr(register_cls, "_sunrise_config_patched", False):
        return

    original_init = register_cls.__init__

    def _extend_config(config, full_config_by_func):
        extra_entries = _sunrise_extra_config_entries()
        existing_keys = {item[0] for item in config}
        merged_config = tuple(config) + tuple(
            item for item in extra_entries if item[0] not in existing_keys
        )

        if full_config_by_func is None:
            return merged_config, None

        merged_map = {key: list(value) for key, value in full_config_by_func.items()}
        for item in extra_entries:
            fn = item[1]
            func_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)
            merged_map.setdefault(func_name, [])
            if item not in merged_map[func_name]:
                merged_map[func_name].append(item)
        return merged_config, merged_map

    def __init__(
        self,
        config,
        user_include_ops=None,
        user_exclude_ops=None,
        cpp_patched_ops=None,
        lib=None,
        full_config_by_func=None,
    ):
        config, full_config_by_func = _extend_config(config, full_config_by_func)
        return original_init(
            self,
            config,
            user_include_ops=user_include_ops,
            user_exclude_ops=user_exclude_ops,
            cpp_patched_ops=cpp_patched_ops,
            lib=lib,
            full_config_by_func=full_config_by_func,
        )

    register_cls.__init__ = __init__
    register_cls._sunrise_config_patched = True
    register_cls._sunrise_original_init = original_init


def _install_typed_ptr_device_patch():
    from flag_gems.utils.tensor_wrapper import TypedPtr

    if getattr(TypedPtr, "_sunrise_device_patched", False):
        return

    def __init__(self, ptr, dtype, device=None):
        self.ptr = ptr
        self.dtype = dtype
        self.device = device

    @classmethod
    def from_tensor(cls, tensor, offset=0):
        return cls(
            tensor.data_ptr() + tensor.element_size() * offset,
            tensor.dtype,
            tensor.device,
        )

    @classmethod
    def reinterpret_tensor(cls, tensor, dtype, offset=0):
        return cls(tensor.data_ptr() + dtype.itemsize * offset, dtype, tensor.device)

    TypedPtr.__init__ = __init__
    TypedPtr.from_tensor = from_tensor
    TypedPtr.reinterpret_tensor = reinterpret_tensor
    TypedPtr._sunrise_device_patched = True


def _install_pointwise_dynamic_complex_patch():
    import torch

    from flag_gems.utils.pointwise_dynamic import ComplexMode, PointwiseDynamicFunction
    from flag_gems.utils.shape_utils import all_the_same_shape, all_the_same_stride
    from flag_gems.utils.tensor_wrapper import StridedBuffer

    if getattr(PointwiseDynamicFunction, "_sunrise_complex_patched", False):
        return

    def _tensor_is_contiguous(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.is_contiguous()
        expected_stride = 1
        for size, stride in zip(reversed(tensor.shape), reversed(tensor.stride())):
            if size == 1:
                continue
            if stride != expected_stride:
                return False
            expected_stride *= size
        return True

    if not hasattr(StridedBuffer, "is_contiguous"):
        StridedBuffer.is_contiguous = _tensor_is_contiguous

    def _call_real_impl(self, *args, _skip_tensor_check=False, **kwargs):
        from flag_gems import runtime

        if not runtime.device.support_fp64:
            ptpu_tensor = next(
                (
                    arg
                    for arg in args
                    if isinstance(arg, torch.Tensor)
                    and arg.device.type == "ptpu"
                    and arg.dtype == torch.float64
                ),
                None,
            )
            if ptpu_tensor is None:
                ptpu_tensor = next(
                    (
                        value
                        for value in kwargs.values()
                        if isinstance(value, torch.Tensor)
                        and value.device.type == "ptpu"
                        and value.dtype == torch.float64
                    ),
                    None,
                )
            if ptpu_tensor is not None:
                cpu_args = tuple(
                    arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args
                )
                cpu_kwargs = {
                    key: (value.cpu() if isinstance(value, torch.Tensor) else value)
                    for key, value in kwargs.items()
                    if not key.startswith("out")
                }
                py_fn = getattr(self._scalar_fn, "fn", self._scalar_fn)
                result = py_fn(*cpu_args, **cpu_kwargs)
                out = kwargs.get("out0")
                if out is not None:
                    out.copy_(result.to(out.device))
                    return out
                if isinstance(result, tuple):
                    return tuple(
                        (
                            item.to(ptpu_tensor.device)
                            if isinstance(item, torch.Tensor)
                            else item
                        )
                        for item in result
                    )
                if isinstance(result, torch.Tensor):
                    return result.to(ptpu_tensor.device)
                return result

        ndim, args, kwargs = self.prepare_args(
            *args, _skip_tensor_check=_skip_tensor_check, **kwargs
        )
        overload = self.instantiate(ndim)
        out = overload(*args, **kwargs)
        return self._unwrap(out)

    def _is_missing_backend_view_op(self, exc, aten_op):
        message = str(exc)
        return (
            isinstance(exc, NotImplementedError)
            and aten_op in message
            and "ptpu" in message
        )

    def _complex_real_view_buffer(self, tensor):
        real_dtype = tensor.dtype.to_real()
        shape = tuple(tensor.shape) + (2,)
        strides = tuple(stride * 2 for stride in tensor.stride()) + (1,)
        return StridedBuffer(tensor, shape=shape, strides=strides, dtype=real_dtype)

    def _complex_component_buffers(self, tensor):
        real_dtype = tensor.dtype.to_real()
        strides = tuple(stride * 2 for stride in tensor.stride())
        real = StridedBuffer(
            tensor, shape=tensor.shape, strides=strides, dtype=real_dtype
        )
        imag = StridedBuffer(
            tensor,
            shape=tensor.shape,
            strides=strides,
            dtype=real_dtype,
            offset=1,
        )
        return real, imag

    def _view_as_real_for_kernel(self, tensor):
        try:
            return torch.view_as_real(tensor)
        except Exception as exc:
            if self._is_missing_backend_view_op(exc, "aten::view_as_real"):
                return self._complex_real_view_buffer(tensor)
            raise

    def _split_complex_components(self, tensor):
        try:
            real_view = torch.view_as_real(tensor)
            return real_view[..., 0], real_view[..., 1]
        except Exception as exc:
            if self._is_missing_backend_view_op(exc, "aten::view_as_real"):
                return self._complex_component_buffers(tensor)
            raise

    def _view_as_complex_result(self, tensor):
        try:
            return torch.view_as_complex(tensor.contiguous())
        except Exception as exc:
            if self._is_missing_backend_view_op(exc, "aten::view_as_complex"):
                return torch.view_as_complex(tensor.cpu().contiguous()).to(
                    tensor.device
                )
            raise

    def _cross_components_for_kernel(self, tensor):
        try:
            real_view = torch.view_as_real(tensor)
        except Exception as exc:
            if not self._is_missing_backend_view_op(exc, "aten::view_as_real"):
                raise
            real_view = torch.view_as_real(tensor.cpu()).to(tensor.device)
        return real_view[..., 0], real_view[..., 1]

    def _cpu_fallback_value(self, value):
        if isinstance(value, torch.Tensor):
            return value.cpu()
        return value

    def _should_cpu_fallback_complex(self, result_dtype, device):
        if device is None or device.type == "cpu":
            return False
        if result_dtype != torch.complex128:
            return False
        from flag_gems import runtime

        return not runtime.device.support_fp64

    def _cpu_fallback_complex_dispatch(self, args, kwargs, device):
        cpu_args = tuple(self._cpu_fallback_value(arg) for arg in args)
        out = kwargs.get("out0")
        py_fn = getattr(self._scalar_fn, "fn", self._scalar_fn)
        result = py_fn(*cpu_args)
        if out is not None:
            out.copy_(result.to(out.device))
            return out
        if isinstance(result, torch.Tensor):
            return result.to(device)
        if isinstance(result, tuple):
            return tuple(
                item.to(device) if isinstance(item, torch.Tensor) else item
                for item in result
            )
        return result

    def _call_complex_dispatch(self, *args, **kwargs):
        strategy = self.complex_strategy
        operands, others = self._split_args(args)

        device = self._infer_device(operands)
        result_dtype = self._infer_complex_dtype(operands)

        if self._should_cpu_fallback_complex(result_dtype, device):
            return self._cpu_fallback_complex_dispatch(args, kwargs, device)

        if strategy.tensorize_scalars and strategy.fallback_target is not None:
            operands = self._tensorize_scalar_operands(operands, result_dtype, device)
            new_args = self._merge_args(operands, others)
            return strategy.fallback_target(*new_args, **kwargs)

        for i in list(operands.keys()):
            operands[i] = self._to_complex_tensor(operands[i], result_dtype, device)

        complex_tensors = [operands[i] for i in sorted(operands.keys())]
        complex_tensors = torch.broadcast_tensors(*complex_tensors)
        for idx, key in enumerate(sorted(operands.keys())):
            operands[key] = complex_tensors[idx]

        classification = self._classify_complex_inputs(operands)

        if strategy.mode == ComplexMode.CROSS and classification == "all_complex":
            return self._call_complex_cross(operands, result_dtype)
        if classification in ("all_complex", "mixed"):
            return self._call_complex_elementwise(
                operands, others, result_dtype, kwargs
            )
        new_args = self._merge_args(operands, others)
        return self._call_real_impl(*new_args, **kwargs)

    def _call_complex_elementwise(self, operands, others, result_dtype, kwargs):
        real_tensors = {
            i: self._view_as_real_for_kernel(t) for i, t in operands.items()
        }
        out_kwargs = dict(kwargs)
        out_complex = out_kwargs.get("out0")
        if out_complex is None:
            first_operand = operands[sorted(operands.keys())[0]]
            out_complex = torch.empty(
                first_operand.shape,
                dtype=result_dtype,
                device=first_operand.device,
            )
            out_kwargs["out0"] = out_complex

        out_kwargs["out0"] = self._view_as_real_for_kernel(out_complex)
        new_args = self._merge_args(real_tensors, others)
        self._call_real_impl(*new_args, _skip_tensor_check=True, **out_kwargs)
        return out_complex

    def _call_complex_cross(self, operands, result_dtype):
        sorted_keys = sorted(operands.keys())
        a_tensor, b_tensor = operands[sorted_keys[0]], operands[sorted_keys[1]]
        ar, ai = self._cross_components_for_kernel(a_tensor)
        br, bi = self._cross_components_for_kernel(b_tensor)

        common_dtype = torch.promote_types(ar.dtype, br.dtype)
        if ar.dtype != common_dtype:
            ar, ai = ar.to(common_dtype), ai.to(common_dtype)
        if br.dtype != common_dtype:
            br, bi = br.to(common_dtype), bi.to(common_dtype)

        cross_kernel = self.complex_strategy.cross_kernel
        real, imag = cross_kernel._call_real_impl(
            ar, ai, br, bi, _skip_tensor_check=True
        )
        out = torch.stack((real, imag), dim=-1)
        return self._view_as_complex_result(out).to(result_dtype)

    def use_fast_path(tensors):
        if not all_the_same_shape(tensors):
            return False
        if all(_tensor_is_contiguous(tensor) for tensor in tensors):
            return True
        return (
            all(isinstance(tensor, torch.Tensor) for tensor in tensors)
            and all_the_same_stride(tensors)
            and torch.ops.aten.is_non_overlapping_and_dense(tensors[0])
        )

    PointwiseDynamicFunction._call_real_impl = _call_real_impl
    PointwiseDynamicFunction._is_missing_backend_view_op = _is_missing_backend_view_op
    PointwiseDynamicFunction._complex_real_view_buffer = _complex_real_view_buffer
    PointwiseDynamicFunction._complex_component_buffers = _complex_component_buffers
    PointwiseDynamicFunction._view_as_real_for_kernel = _view_as_real_for_kernel
    PointwiseDynamicFunction._split_complex_components = _split_complex_components
    PointwiseDynamicFunction._view_as_complex_result = _view_as_complex_result
    PointwiseDynamicFunction._cross_components_for_kernel = _cross_components_for_kernel
    PointwiseDynamicFunction._cpu_fallback_value = _cpu_fallback_value
    PointwiseDynamicFunction._should_cpu_fallback_complex = _should_cpu_fallback_complex
    PointwiseDynamicFunction._cpu_fallback_complex_dispatch = (
        _cpu_fallback_complex_dispatch
    )
    PointwiseDynamicFunction._call_complex_dispatch = _call_complex_dispatch
    PointwiseDynamicFunction._call_complex_elementwise = _call_complex_elementwise
    PointwiseDynamicFunction._call_complex_cross = _call_complex_cross
    PointwiseDynamicFunction.use_fast_path = staticmethod(use_fast_path)
    PointwiseDynamicFunction._sunrise_complex_patched = True


def _install_pointwise_dynamic_post_import_hook():
    import builtins
    import sys

    if getattr(builtins, "_sunrise_pointwise_import_hook_installed", False):
        return

    original_import = builtins.__import__

    def maybe_patch():
        module = sys.modules.get("flag_gems.utils.pointwise_dynamic")
        if module is None or getattr(module, "_sunrise_complex_patch_attempted", False):
            return
        module._sunrise_complex_patch_attempted = True
        builtins.__import__ = original_import
        builtins._sunrise_pointwise_import_hook_installed = False
        _install_pointwise_dynamic_complex_patch()

    def import_with_sunrise_pointwise_patch(
        name, globals=None, locals=None, fromlist=(), level=0
    ):
        module = original_import(name, globals, locals, fromlist, level)
        if name == "flag_gems.utils.pointwise_dynamic" or (
            name == "flag_gems.utils" and "pointwise_dynamic" in fromlist
        ):
            maybe_patch()
        return module

    builtins.__import__ = import_with_sunrise_pointwise_patch
    builtins._sunrise_pointwise_import_hook_installed = True


def _install_ptpu_manual_seed_patch():
    import torch

    ptpu_mod = getattr(torch, "ptpu", None)
    if ptpu_mod is None or getattr(ptpu_mod, "_sunrise_manual_seed_patched", False):
        return

    def _is_in_bad_fork():
        return False

    def manual_seed_all(seed):
        seed = int(seed) & 0xFFFFFFFFFFFFFFFF

        # PTPU exposes RNG state get/set but not a Python seed API. The runtime
        # state is a 16-byte blob where the low 8 bytes act as the seed and the
        # high 8 bytes can be reset to zero for a fresh sequence start.
        # `torch.manual_seed()` can be called under `with torch.device("ptpu")`,
        # so build the state explicitly on CPU for `torch.ptpu.set_rng_state()`.
        state = torch.zeros(16, dtype=torch.uint8, device="cpu")
        for i in range(8):
            state[i] = (seed >> (8 * i)) & 0xFF

        for device_idx in range(ptpu_mod.device_count()):
            ptpu_mod.set_rng_state(state, device_idx)

    ptpu_mod._is_in_bad_fork = _is_in_bad_fork
    ptpu_mod.manual_seed_all = manual_seed_all
    ptpu_mod._sunrise_manual_seed_patched = True


def _install_ptpu_default_generators_patch():
    import torch

    ptpu_mod = getattr(torch, "ptpu", None)
    if (
        ptpu_mod is None
        or hasattr(ptpu_mod, "default_generators")
        or getattr(ptpu_mod, "_sunrise_default_generators_patched", False)
    ):
        return

    class _SunrisePtpuGenerator:
        def __init__(self, device_idx):
            self.device_idx = int(device_idx)
            self.device = torch.device("ptpu", self.device_idx)

        def get_state(self):
            return ptpu_mod.get_rng_state(self.device_idx).detach().cpu().clone()

        def set_state(self, state):
            if not isinstance(state, torch.Tensor):
                raise TypeError("PTPU RNG state must be a torch.Tensor")
            if state.dtype != torch.uint8:
                raise TypeError("PTPU RNG state must be a torch.uint8 tensor")
            ptpu_mod.set_rng_state(state.detach().cpu().contiguous(), self.device_idx)

        def manual_seed(self, seed):
            seed = int(seed) & 0xFFFFFFFFFFFFFFFF
            state = torch.zeros(16, dtype=torch.uint8, device="cpu")
            for i in range(8):
                state[i] = (seed >> (8 * i)) & 0xFF
            self.set_state(state)
            return self

    class _SunrisePtpuDefaultGenerators:
        def __init__(self):
            self._generators = {}

        def _normalize_device(self, device):
            if device is None:
                return int(ptpu_mod.current_device())
            if isinstance(device, torch.device):
                if device.type != "ptpu":
                    raise RuntimeError(f"Expected a ptpu device, got {device}")
                return (
                    int(ptpu_mod.current_device())
                    if device.index is None
                    else int(device.index)
                )
            if isinstance(device, str):
                return self._normalize_device(torch.device(device))
            return int(device)

        def __getitem__(self, device):
            device_idx = self._normalize_device(device)
            device_count = int(ptpu_mod.device_count())
            if device_idx < 0:
                device_idx += device_count
            if device_idx < 0 or device_idx >= device_count:
                raise IndexError(
                    f"PTPU device index {device_idx} is out of range "
                    f"for {device_count} devices"
                )
            if device_idx not in self._generators:
                self._generators[device_idx] = _SunrisePtpuGenerator(device_idx)
            return self._generators[device_idx]

        def __iter__(self):
            for device_idx in range(len(self)):
                yield self[device_idx]

        def __len__(self):
            return int(ptpu_mod.device_count())

    ptpu_mod.default_generators = _SunrisePtpuDefaultGenerators()
    ptpu_mod._sunrise_default_generators_patched = True


def _install_ptpu_multiprocessing_reduction_patch():
    import multiprocessing.reduction as mp_reduction

    import torch
    import torch.multiprocessing.reductions as reductions
    from torch.nn.parameter import Parameter

    if getattr(reductions, "_sunrise_ptpu_reduce_tensor_patched", False):
        return

    original_reduce_tensor = reductions.reduce_tensor

    # Keep this in `_sunrise/__init__.py` instead of `monkey_patch.py` because
    # multiprocessing reducers are registered eagerly in Python's global
    # pickling table. This is import-time runtime wiring, not a call-site-level
    # torch API fallback that can be caught and retried after a NotImplemented.
    def reduce_tensor_with_ptpu_cpu_staging(tensor):
        if not _should_stage_ptpu_tensor_for_multiprocessing(tensor):
            return original_reduce_tensor(tensor)

        if tensor.requires_grad and not tensor.is_leaf:
            raise RuntimeError(
                "Cowardly refusing to serialize non-leaf tensor which requires_grad, "
                "since autograd does not support crossing process boundaries.  "
                "If you just want to transfer the data, call detach() on the tensor "
                "before serializing (e.g., putting it on the queue)."
            )

        reductions.check_serializing_named_tensor(tensor)
        torch.utils.hooks.warn_if_has_hooks(tensor)

        return (
            reductions._sunrise_rebuild_ptpu_tensor_from_cpu,
            (
                type(tensor),
                tensor.detach().cpu(),
                tensor.device,
                tensor.requires_grad,
            ),
        )

    reductions._sunrise_rebuild_ptpu_tensor_from_cpu = (
        _sunrise_rebuild_ptpu_tensor_from_cpu
    )
    reductions._sunrise_original_reduce_tensor = original_reduce_tensor
    reductions.reduce_tensor = reduce_tensor_with_ptpu_cpu_staging
    for tensor_cls in torch._tensor_classes:
        mp_reduction.register(tensor_cls, reduce_tensor_with_ptpu_cpu_staging)
    mp_reduction.register(torch.Tensor, reduce_tensor_with_ptpu_cpu_staging)
    mp_reduction.register(Parameter, reduce_tensor_with_ptpu_cpu_staging)
    reductions._sunrise_ptpu_reduce_tensor_patched = True


_install_ptpu_default_generators_patch()
_install_ptpu_manual_seed_patch()
_install_autograd_dispatch_patch()
_install_register_config_patch()  # 有些公共库也没有注册的op，只能先放在这里了。使得tests能过
_install_pointwise_dynamic_post_import_hook()


__all__ = ["*"]
