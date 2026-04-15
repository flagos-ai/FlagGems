"""Precision-checking register – loaded only when precision checking is enabled.

This module is NOT imported on the normal execution path.  It is lazily
imported by ``register.py`` only when the user explicitly requests
``PrecisionCheckRegister``.
"""

import functools

import torch

from ..logging_utils import (
    compare_outputs,
    get_precision_logger,
    get_tensor_info,
    precision_config,
)
from .register import Register

# Maximum tensor element count allowed for precision check
# (skip if exceeded to avoid large tensor copy overhead)
_MAX_NUMEL_FOR_CHECK = 1 * 1024 * 1024  # 1M elements


def _get_dtype_tolerance(args, default_rtol, default_atol):
    """Automatically adjust tolerance based on the dtype of input tensors."""
    for a in args:
        if isinstance(a, torch.Tensor) and a.is_floating_point():
            if a.dtype in (torch.bfloat16, torch.float16):
                return (max(default_rtol, 1e-2), max(default_atol, 1e-2))
            break
    return (default_rtol, default_atol)


def _to_cpu(x):
    """Recursively move tensors to CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    elif isinstance(x, (list, tuple)):
        return type(x)(_to_cpu(i) for i in x)
    elif isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    return x


def _max_tensor_numel(args):
    """Return the element count of the largest tensor in the arguments."""
    max_n = 0
    for a in args:
        if isinstance(a, torch.Tensor):
            max_n = max(max_n, a.numel())
    return max_n


# Operators that should never be precision-checked
_SKIP_OPS = frozenset(
    {
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
        # Random sampling operators (GPU/CPU RNGs differ)
        "exponential_",
        "normal_",
        "uniform_",
        "bernoulli_",
        "random_",
        "multinomial",
        "randperm",
        # Sorting/selection operators (order of equal values may differ)
        "sort",
        "topk",
        "argsort",
    }
)


def _wrap_op_with_precision_check(op_key, fn):
    """Wrap a FlagGems operator to compare its output against native PyTorch.

    Since FlagGems replaces the CUDA dispatch, the native implementation
    cannot be called on GPU, so inputs are copied to CPU to compute the
    reference result.  Performance overhead is controlled by:
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

        # Skip out variants
        overload_part = op_key.split(".")[-1] if "." in op_key else ""
        if overload_part == "out" or op_name.endswith("_out"):
            return fg_result

        # Skip operators that do not need checking
        if op_name in _SKIP_OPS:
            return fg_result

        # Skip large tensors to avoid copy overhead
        if _max_tensor_numel(args) > _MAX_NUMEL_FOR_CHECK:
            return fg_result

        try:
            parts = op_key.split(".")
            base_name = parts[0]
            overload_name = parts[1] if len(parts) > 1 else "default"

            aten_packet = getattr(torch.ops.aten, base_name, None)
            if aten_packet is None:
                return fg_result
            aten_overload = getattr(aten_packet, overload_name, None)
            if aten_overload is None:
                return fg_result

            cpu_args = [_to_cpu(a) for a in args]
            cpu_kwargs = {k: _to_cpu(v) for k, v in kwargs.items()}

            with torch.no_grad():
                pt_result_cpu = aten_overload(*cpu_args, **cpu_kwargs)

            fg_result_cpu = _to_cpu(fg_result)

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
                    msg += (
                        f" | max_abs: {info['max_abs']:.6e}"
                        f" | max_rel: {info['max_rel']:.6e}"
                    )
                msg += f" | rtol={rtol}, atol={atol}"
                logger.warning(msg)

        except Exception:
            pass

        return fg_result

    return wrapper


class PrecisionCheckRegister(Register):
    """Register subclass that wraps every operator with precision checking.

    This class is only instantiated when the user has explicitly called
    ``enable_precision_check()`` before ``enable()`` / ``only_enable()``.
    It is never on the normal execution path.
    """

    def register_impl(self, key, fn):
        if self.lib is None:
            raise ValueError("Library instance is not provided.")

        wrapped_fn = _wrap_op_with_precision_check(key, fn)

        device_key = self.reg_key
        self.all_ops.append(fn.__name__)
        self.all_keys.append(key)
        self.lib.impl(key, wrapped_fn, device_key)
