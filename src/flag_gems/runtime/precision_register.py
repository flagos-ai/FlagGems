"""Precision-checking register – loaded only when precision checking is enabled.

This module is NOT imported on the normal execution path.  It is lazily
imported by ``register.py`` only when the user explicitly requests
``PrecisionCheckRegister``.
"""

import functools

import torch

from ..config import get_skip_precision_check_ops
from ..logging_utils import (
    compare_outputs,
    get_tensor_info,
    precision_config,
    write_precision_result,
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


# Operators that should never be precision-checked – loaded from conf/operators.yaml
_SKIP_OPS = get_skip_precision_check_ops()


def _parse_op_key(op_key):
    """Parse op_key once and return (op_name, overload_name, should_skip).

    This avoids repeated string splitting inside the hot wrapper.
    """
    # Strip namespace prefix (e.g. "aten::add.Tensor" -> "add.Tensor")
    bare_key = op_key.split("::")[-1] if "::" in op_key else op_key

    # Split into base op name and overload
    dot_pos = bare_key.find(".")
    if dot_pos >= 0:
        op_name = bare_key[:dot_pos]
        overload_name = bare_key[dot_pos + 1 :]
    else:
        op_name = bare_key
        overload_name = "default"

    # Determine if this op should be skipped entirely (never checked)
    should_skip = (
        overload_name == "out" or op_name.endswith("_out") or op_name in _SKIP_OPS
    )

    return op_name, overload_name, should_skip


def _wrap_op_with_precision_check(op_key, fn):
    """Wrap a FlagGems operator to compare its output against native PyTorch.

    Since FlagGems replaces the CUDA dispatch, the native implementation
    cannot be called on GPU, so inputs are copied to CPU to compute the
    reference result.  Performance overhead is controlled by:
    - max_checks: only check the first N calls per operator (default 10)
    - skip large tensors (over 1M elements)
    - once a failure is logged, that operator is no longer checked
    """
    # --- Pre-compute everything derivable from op_key at wrap time ---
    op_name, overload_name, should_skip = _parse_op_key(op_key)

    # If this op can never be checked, return the unwrapped function directly
    if should_skip:
        return fn

    # Pre-resolve the aten overload so we don't do getattr on every call
    aten_packet = getattr(torch.ops.aten, op_name, None)
    aten_overload = getattr(aten_packet, overload_name, None) if aten_packet else None

    # If we can't find the native implementation, no point wrapping
    if aten_overload is None:
        return fn

    # Pre-fetch the set reference for fast membership test
    _logged_ops = precision_config["logged_ops"]
    _call_count = 0

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal _call_count

        # Skip operators that have already logged a failure
        if op_key in _logged_ops:
            return fn(*args, **kwargs)

        # Sampling: only check the first N calls per operator
        _call_count += 1
        if _call_count > precision_config.get("max_checks", 10):
            return fn(*args, **kwargs)

        # Skip large tensors to avoid copy overhead
        if _max_tensor_numel(args) > _MAX_NUMEL_FOR_CHECK:
            return fn(*args, **kwargs)

        # Execute the FlagGems implementation FIRST with no interference
        fg_result = fn(*args, **kwargs)

        try:
            # Copy inputs and output to CPU for comparison.
            # The .cpu() call implicitly synchronizes the CUDA stream.
            # For in-place ops (op_name ends with '_'), inputs may have been
            # modified, but those ops are typically skipped via _SKIP_OPS.
            cpu_args = [_to_cpu(a) for a in args]
            cpu_kwargs = {k: _to_cpu(v) for k, v in kwargs.items()}
            fg_result_cpu = _to_cpu(fg_result)

            with torch.no_grad():
                pt_result_cpu = aten_overload(*cpu_args, **cpu_kwargs)

            cfg = precision_config
            rtol, atol = _get_dtype_tolerance(args, cfg["rtol"], cfg["atol"])
            is_close, info = compare_outputs(fg_result_cpu, pt_result_cpu, rtol, atol)

            if not is_close:
                _logged_ops.add(op_key)
                input_info = [get_tensor_info(a) for a in args if get_tensor_info(a)]
                output_info = get_tensor_info(fg_result)

                record = {
                    "op": op_key,
                    "status": "FAIL",
                    "inputs": input_info,
                    "output": output_info,
                    "rtol": rtol,
                    "atol": atol,
                }
                if "error" in info:
                    record["error"] = info["error"]
                    record["fg_value"] = info["fg"]
                    record["pt_value"] = info["pt"]
                else:
                    record["max_abs_diff"] = info["max_abs"]
                    record["max_rel_diff"] = info["max_rel"]
                write_precision_result(record)

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

    def register_impl(self, key, fn, extra_dispatch_keys=()):
        if self.lib is None:
            raise ValueError("Library instance is not provided.")

        wrapped_fn = _wrap_op_with_precision_check(key, fn)

        device_key = self.reg_key
        self.all_ops.append(fn.__name__)
        self.all_keys.append(key)
        self.lib.impl(key, wrapped_fn, device_key)
        for dispatch_key in extra_dispatch_keys:
            self.lib.impl(key, wrapped_fn, dispatch_key)
