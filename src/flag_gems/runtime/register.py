import functools
import warnings

import torch

from ..logging_utils import (
    compare_outputs,
    get_precision_logger,
    get_tensor_info,
    precision_config,
)
from . import backend, common, error
from .backend.device import DeviceDetector


class Register:
    def __init__(
        self,
        config,
        user_include_ops=None,
        user_exclude_ops=None,
        cpp_patched_ops=None,
        lib=None,
        full_config_by_func=None,
    ):
        self.device = DeviceDetector()

        # lib is a instance of torch.library.Library
        # Some inference chips may not support the backward implementation of operators
        self.lib = lib

        # reg_key like 'CUDA'
        self.reg_key = self.device.dispatch_key
        self.all_ops = []
        self.all_keys = []

        # optional mapping func_name -> list of config entries
        self.full_config_by_func = full_config_by_func
        self.cpp_patched_ops = set(cpp_patched_ops or [])

        if user_include_ops:
            self.include_ops = list(user_include_ops or [])
            self.exclude_ops = []
            self.config = config
            self.extract_include_config()
            # Use the filtered include config to avoid registering all ops.
            self.config = self.include_config
            self.for_each()
        else:
            self.vendor_unused_ops_list = self.get_vendor_unused_op()
            self.exclude_ops = (
                list(user_exclude_ops or []) + self.vendor_unused_ops_list
            )
            self.config = config
            self.config_filter()
            self.for_each()

    def extract_include_config(self):
        # Simple fast path: if we have a full_config_by_func mapping, iterate
        # over the requested function names and collect matching config items.
        self.include_config = []

        if self.full_config_by_func:
            for name in self.include_ops:
                for config_item in self.full_config_by_func.get(name, []):
                    op_name, func = config_item[0], config_item[1]
                    # respect optional condition functions
                    if len(config_item) > 2:
                        condition_func = config_item[2]
                        if not condition_func():
                            continue
                    if op_name in self.cpp_patched_ops:
                        continue
                    self.include_config.append((op_name, func))
        else:
            # fallback: scan provided config and match by func name or op name
            for config_item in self.config:
                op_name, func = config_item[0], config_item[1]
                func_name = func.__name__ if hasattr(func, "__name__") else str(func)
                if (
                    func_name not in self.include_ops
                    and op_name not in self.include_ops
                ):
                    continue
                if len(config_item) > 2:
                    condition_func = config_item[2]
                    if not condition_func():
                        continue
                if op_name in self.cpp_patched_ops:
                    continue
                self.include_config.append((op_name, func))

        if not self.include_config:
            warnings.warn(
                "only_enable failed: No op to register. Check if include is correct."
            )
            return

    def config_filter(self):
        def enabled(item):
            return len(item) < 3 or bool(item[2]())

        self.config = [
            (item[0], item[1])
            for item in self.config
            if enabled(item)
            and item[1].__name__ not in self.exclude_ops
            and item[0] not in self.cpp_patched_ops
        ]

    def get_vendor_unused_op(self):
        if self.device.vendor != common.vendors.NVIDIA:
            return backend.get_curent_device_unused_op(self.device.vendor_name)
        return []

    def register_impl(self, key, fn):
        if self.lib is None:
            raise ValueError("Library instance is not provided.")
        device_key = self.reg_key
        self.all_ops.append(fn.__name__)
        self.all_keys.append(key)
        self.lib.impl(key, fn, device_key)

    def for_each(self):
        for key, func in self.config:
            try:
                self.register_impl(key, func)
            except Exception as e:
                error.register_error(e)

    def get_all_ops(self):
        return self.all_ops

    def get_all_keys(self):
        return self.all_keys

    def get_unused_ops(self):
        return self.exclude_ops

    def get_vendor_name(self):
        return self.device.vendor_name

    def get_current_device(self):
        return self.device.name


# Maximum tensor element count allowed for precision check (skip if exceeded to avoid large tensor copy overhead)
_MAX_NUMEL_FOR_CHECK = 1 * 1024 * 1024  # 1M elements


def _get_dtype_tolerance(args, default_rtol, default_atol):
    """Automatically adjust tolerance based on the dtype of input tensors"""
    for a in args:
        if isinstance(a, torch.Tensor) and a.is_floating_point():
            if a.dtype in (torch.bfloat16, torch.float16):
                return (max(default_rtol, 1e-2), max(default_atol, 1e-2))
            break
    return (default_rtol, default_atol)


def _to_cpu(x):
    """Recursively move tensors to CPU"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    elif isinstance(x, (list, tuple)):
        return type(x)(_to_cpu(i) for i in x)
    elif isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    return x


def _max_tensor_numel(args):
    """Return the element count of the largest tensor in the arguments"""
    max_n = 0
    for a in args:
        if isinstance(a, torch.Tensor):
            max_n = max(max_n, a.numel())
    return max_n


def _wrap_op_with_precision_check(op_key, fn):
    """Wrap a FlagGems operator to compare its output against the native PyTorch CPU implementation after execution.

    Since FlagGems replaces the CUDA dispatch, the native implementation cannot be called on GPU,
    so inputs are copied to CPU to compute the reference result. Performance overhead is controlled by:
    - max_checks: only check the first N calls per operator (default 10)
    - skip large tensors (over 1M elements)
    - once a failure is logged, that operator is no longer checked
    """
    _call_count = 0

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal _call_count

        # Execute the FlagGems implementation first
        fg_result = fn(*args, **kwargs)

        cfg = precision_config
        if not cfg["enabled"]:
            return fg_result

        # Skip operators that have already logged a failure
        if op_key in cfg["logged_ops"]:
            return fg_result

        # Sampling: only check the first N calls per operator
        _call_count += 1
        if _call_count > cfg.get("max_checks", 10):
            return fg_result

        op_name = (
            op_key.split("::")[-1].split(".")[0]
            if "::" in op_key
            else op_key.split(".")[0]
        )

        # Skip out variants (different argument signatures, prone to errors when calling on CPU)
        overload_part = op_key.split(".")[-1] if "." in op_key else ""
        if overload_part == "out" or op_name.endswith("_out"):
            return fg_result

        # Skip operators that do not need checking
        skip_ops = {
            # Pure layout / memory operations
            "copy_",
            "_to_copy",
            "view",
            "reshape",
            "expand",
            "permute",
            "transpose",
            "contiguous",
            "clone",
            "to",
            "empty",
            "zeros",
            "ones",
            "full",
            "masked_fill_",
            # Random sampling operators (GPU/CPU RNGs differ, results will inevitably mismatch)
            "exponential_",
            "normal_",
            "uniform_",
            "bernoulli_",
            "random_",
            "multinomial",
            "randperm",
            # Sorting/selection operators (order of equal values may differ, not suitable for element-wise comparison)
            "sort",
            "topk",
            "argsort",
        }
        if op_name in skip_ops:
            return fg_result

        # Skip large tensors to avoid copy overhead
        if _max_tensor_numel(args) > _MAX_NUMEL_FOR_CHECK:
            return fg_result

        try:
            # Parse op_key to get the correct overload
            # op_key is in the form "add.Tensor", "mm.default", "softmax.int", etc.
            parts = op_key.split(".")
            base_name = parts[0]
            overload_name = parts[1] if len(parts) > 1 else "default"

            aten_packet = getattr(torch.ops.aten, base_name, None)
            if aten_packet is None:
                return fg_result
            aten_overload = getattr(aten_packet, overload_name, None)
            if aten_overload is None:
                return fg_result

            # Copy inputs to CPU and call the native aten implementation (unaffected by FlagGems on CPU)
            cpu_args = [_to_cpu(a) for a in args]
            cpu_kwargs = {k: _to_cpu(v) for k, v in kwargs.items()}

            with torch.no_grad():
                pt_result_cpu = aten_overload(*cpu_args, **cpu_kwargs)

            # Also copy FlagGems result to CPU for comparison (avoid moving CPU result back to GPU)
            fg_result_cpu = _to_cpu(fg_result)

            # Automatically adjust tolerance based on dtype
            rtol, atol = _get_dtype_tolerance(args, cfg["rtol"], cfg["atol"])

            is_close, info = compare_outputs(fg_result_cpu, pt_result_cpu, rtol, atol)

            if not is_close:
                cfg["logged_ops"].add(op_key)
                logger = get_precision_logger()
                input_info = [get_tensor_info(a) for a in args if get_tensor_info(a)]
                output_info = get_tensor_info(fg_result)

                msg = f"Op: {op_key} | FAIL | in: {input_info} | out: {output_info}"
                if "error" in info:
                    msg += f" | {info['error']}: fg={info['fg']}, pt={info['pt']}"
                else:
                    msg += f" | max_abs: {info['max_abs']:.6e} | max_rel: {info['max_rel']:.6e}"
                msg += f" | rtol={rtol}, atol={atol}"
                logger.warning(msg)

        except Exception:
            pass

        return fg_result

    return wrapper


class PrecisionCheckRegister(Register):
    def register_impl(self, key, fn):
        if self.lib is None:
            raise ValueError("Library instance is not provided.")

        wrapped_fn = _wrap_op_with_precision_check(key, fn)

        device_key = self.reg_key
        self.all_ops.append(fn.__name__)
        self.all_keys.append(key)
        self.lib.impl(key, wrapped_fn, device_key)
