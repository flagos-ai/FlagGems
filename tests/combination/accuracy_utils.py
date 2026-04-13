"""
Accuracy utilities for combination tests.

This module provides reference computation and assertion functions
tailored for multi-operator combination testing.  It follows the same
three-level reference strategy as the single-operator tests:

  1. GPU fp64 (default, when device supports fp64 and --ref is not cpu)
  2. CPU fp32/fp64 (when --ref cpu or device lacks fp64)
  3. try/except fallback to CPU fp32 (when an operator does not support fp64)

Tolerance is automatically scaled by dtype and the number of chained
operators (sqrt scaling for independent error accumulation).
"""

import copy
import logging
import math

import torch

import flag_gems
from flag_gems.testing import RESOLUTION

fp64_is_supported = flag_gems.runtime.device.support_fp64
bf16_is_supported = flag_gems.runtime.device.support_bf16

# Combination test float dtypes – adapts to backend capabilities.
COMBO_FLOAT_DTYPES = [torch.float16, torch.float32]
if bf16_is_supported:
    COMBO_FLOAT_DTYPES.append(torch.bfloat16)

# Low-precision only (for mixed-precision accumulation tests that compare against fp32).
COMBO_LOW_PRECISION_DTYPES = [torch.float16]
if bf16_is_supported:
    COMBO_LOW_PRECISION_DTYPES.append(torch.bfloat16)

try:
    from ..conftest import TO_CPU
except ImportError:
    TO_CPU = False

device = flag_gems.device

logger = logging.getLogger("flag_gems.combination_test")

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------

# Base absolute tolerance per dtype for combination tests.
# These are intentionally a bit more relaxed than single-operator values
# because multi-operator chains accumulate rounding errors.
COMBO_BASE_ATOL = {
    torch.float16: 5e-3,
    torch.bfloat16: 1.5e-2,
    torch.float32: 1e-5,
    torch.float64: 1e-10,
}


def _get_combo_tolerance(dtype, num_ops=1, atol_override=None, rtol_override=None):
    """Return (rtol, atol) suitable for a chain of *num_ops* operators.

    * ``rtol`` comes from ``flag_gems.testing.RESOLUTION[dtype]``.
    * ``atol`` is ``COMBO_BASE_ATOL[dtype] * sqrt(num_ops)``.

    The sqrt scaling reflects that independent rounding errors accumulate
    proportionally to the square root of the number of steps – tighter
    than linear (catches real bugs) yet looser than a single-op tolerance
    (avoids false positives from normal accumulation).
    """
    rtol = rtol_override if rtol_override is not None else RESOLUTION.get(dtype, 1e-3)
    base_atol = COMBO_BASE_ATOL.get(dtype, 1e-4)
    atol = (
        atol_override
        if atol_override is not None
        else base_atol * math.sqrt(max(num_ops, 1))
    )
    return rtol, atol


# ---------------------------------------------------------------------------
# Reference computation (three-level strategy)
# ---------------------------------------------------------------------------


def _ref_device_and_dtype():
    """Decide the device / dtype to use for the reference computation."""
    if TO_CPU:
        # User passed ``--ref cpu``.  Prefer fp64 when the host supports it
        # (CPU almost always does, but stay safe).
        ref_dtype = torch.float64
        return "cpu", ref_dtype
    if fp64_is_supported:
        return device, torch.float64
    # Device does not support fp64 – fall back to CPU fp32.
    return "cpu", torch.float32


def _to_ref(x, ref_device, ref_dtype):
    """Move a single value to the reference device / dtype."""
    if isinstance(x, torch.Tensor):
        return x.detach().to(ref_device).to(ref_dtype)
    return x


def compute_reference(model, *inputs, **kwargs):
    """Run *model* on the reference device / dtype and return the output.

    The model is ``deepcopy``-ed so the original is not modified.
    All ``torch.Tensor`` positional and keyword arguments are moved to the
    reference device / dtype automatically.  The output tensor(s) are
    returned on the **original** device so that callers can compare
    directly.

    If the primary strategy (GPU fp64) raises a ``RuntimeError`` – e.g.
    because one operator in the combination does not support fp64 – we
    silently fall back to CPU fp32.
    """
    orig_device = None

    def _try_run(ref_device, ref_dtype):
        nonlocal orig_device
        ref_model = copy.deepcopy(model).to(ref_device).to(ref_dtype)
        ref_model.eval()
        ref_inputs = tuple(_to_ref(x, ref_device, ref_dtype) for x in inputs)
        ref_kwargs = {k: _to_ref(v, ref_device, ref_dtype) for k, v in kwargs.items()}
        # Remember the device of the first tensor input.
        for x in inputs:
            if isinstance(x, torch.Tensor):
                orig_device = x.device
                break
        with torch.no_grad():
            return ref_model(*ref_inputs, **ref_kwargs)

    ref_device, ref_dtype = _ref_device_and_dtype()
    try:
        output = _try_run(ref_device, ref_dtype)
    except RuntimeError:
        # Fallback: CPU fp32
        output = _try_run("cpu", torch.float32)

    # Move output back to the original device for easy comparison.
    if orig_device is not None and isinstance(output, torch.Tensor):
        return output.to(orig_device)
    if isinstance(output, (tuple, list)):
        cls = type(output)
        return cls(
            o.to(orig_device) if isinstance(o, torch.Tensor) else o for o in output
        )
    return output


def compute_reference_with_grad(model, *inputs, loss_fn=None, **kwargs):
    """Run forward + backward on the reference device and return results.

    Returns:
        (ref_output, ref_input_grads, ref_param_grads)

        * ``ref_output`` – forward output (on the original device).
        * ``ref_input_grads`` – dict mapping index to gradient for each
          positional input that had ``requires_grad``.
        * ``ref_param_grads`` – dict mapping parameter name to gradient.
    """
    ref_device, ref_dtype = _ref_device_and_dtype()
    orig_device = None

    # Identify which inputs need gradients.
    grad_input_indices = []
    for i, x in enumerate(inputs):
        if isinstance(x, torch.Tensor):
            if orig_device is None:
                orig_device = x.device
            if x.requires_grad:
                grad_input_indices.append(i)

    def _run(r_device, r_dtype):
        ref_model = copy.deepcopy(model).to(r_device).to(r_dtype)
        ref_model.train()  # need grad through dropout etc.
        ref_inputs = []
        for i, x in enumerate(inputs):
            if isinstance(x, torch.Tensor):
                t = x.detach().to(r_device).to(r_dtype)
                if i in grad_input_indices:
                    t.requires_grad_(True)
                ref_inputs.append(t)
            else:
                ref_inputs.append(x)
        ref_inputs = tuple(ref_inputs)
        ref_kwargs = {k: _to_ref(v, r_device, r_dtype) for k, v in kwargs.items()}

        ref_output = ref_model(*ref_inputs, **ref_kwargs)

        # Backward
        if loss_fn is not None:
            loss = loss_fn(ref_output)
        else:
            if isinstance(ref_output, torch.Tensor):
                loss = ref_output.sum()
            else:
                loss = ref_output[0].sum()
        loss.backward()

        # Collect gradients.
        input_grads = {}
        for i in grad_input_indices:
            if ref_inputs[i].grad is not None:
                input_grads[i] = ref_inputs[i].grad.to(orig_device)

        param_grads = {}
        for name, p in ref_model.named_parameters():
            if p.grad is not None:
                param_grads[name] = p.grad.to(orig_device)

        if isinstance(ref_output, torch.Tensor):
            ref_output = ref_output.detach().to(orig_device)
        elif isinstance(ref_output, (tuple, list)):
            cls = type(ref_output)
            ref_output = cls(
                o.detach().to(orig_device) if isinstance(o, torch.Tensor) else o
                for o in ref_output
            )

        return ref_output, input_grads, param_grads

    try:
        return _run(ref_device, ref_dtype)
    except RuntimeError:
        return _run("cpu", torch.float32)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def combo_assert_close(res, ref, dtype, num_ops=1, atol=None, rtol=None, name=""):
    """Assert *res* and *ref* are close, with tolerance scaled for *num_ops*.

    This is the primary assertion for **Mode 1 (standard forward comparison)**.
    """
    rtol_val, atol_val = _get_combo_tolerance(dtype, num_ops, atol, rtol)

    # Bring both to the same dtype for comparison.
    if res.dtype != ref.dtype:
        ref = ref.to(res.dtype)
    if res.device != ref.device:
        ref = ref.to(res.device)

    # Compute actual error for logging.
    with torch.no_grad():
        diff = (res.float() - ref.float()).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()

    passed = True
    try:
        torch.testing.assert_close(
            res,
            ref,
            atol=atol_val,
            rtol=rtol_val,
            msg=lambda m: f"{name}: {m}" if name else m,
        )
    except AssertionError:
        passed = False
        raise
    finally:
        logger.info(
            "accuracy_check",
            extra={
                "test_data": {
                    "event": "accuracy_check",
                    "check_type": "forward_comparison",
                    "name": name,
                    "dtype": str(dtype),
                    "num_ops": num_ops,
                    "expected_atol": atol_val,
                    "expected_rtol": rtol_val,
                    "actual_max_error": max_err,
                    "actual_mean_error": mean_err,
                    "passed": passed,
                }
            },
        )


def assert_accumulation_error(
    output_low, output_fp32, dtype, num_layers, max_relative_error=None
):
    """Assert mixed-precision accumulation error is within bounds.

    This is the primary assertion for **Mode 2 (mixed-precision accumulation)**.

    The maximum allowed relative error defaults to::

        RESOLUTION[dtype] * sqrt(num_layers)

    which lets the tolerance grow sub-linearly with depth.
    """
    output_cmp = output_low.to(torch.float32)
    if output_cmp.device != output_fp32.device:
        output_cmp = output_cmp.to(output_fp32.device)

    error = (output_cmp - output_fp32).abs()
    relative_error = (error / (output_fp32.abs() + 1e-6)).mean().item()

    if max_relative_error is None:
        base = RESOLUTION.get(dtype, 1e-3)
        max_relative_error = base * math.sqrt(max(num_layers, 1))
        # Clamp to a reasonable upper bound – even 32 layers should not
        # exceed ~15% relative error for fp16 / ~30% for bf16.
        max_relative_error = min(max_relative_error, 0.30)

    passed = relative_error < max_relative_error

    logger.info(
        "accuracy_check",
        extra={
            "test_data": {
                "event": "accuracy_check",
                "check_type": "accumulation_error",
                "dtype": str(dtype),
                "num_layers": num_layers,
                "relative_error": relative_error,
                "max_allowed": max_relative_error,
                "passed": passed,
            }
        },
    )

    assert passed, (
        f"Accumulation error too high for {dtype} with {num_layers} layers: "
        f"{relative_error:.4%} > {max_relative_error:.4%}"
    )


def assert_gradient_close(
    grad_gems, grad_ref, dtype, num_ops=1, atol=None, rtol=None, name=""
):
    """Assert two gradient tensors are close.

    This is the primary assertion for **Mode 3 (gradient comparison)**.
    Uses the same tolerance logic as ``combo_assert_close``.
    """
    # Log via combo_assert_close which already has logging.
    # Override check_type by logging separately.
    rtol_val, atol_val = _get_combo_tolerance(dtype, num_ops, atol, rtol)

    if grad_gems.dtype != grad_ref.dtype:
        grad_ref = grad_ref.to(grad_gems.dtype)
    if grad_gems.device != grad_ref.device:
        grad_ref = grad_ref.to(grad_gems.device)

    with torch.no_grad():
        diff = (grad_gems.float() - grad_ref.float()).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()

    passed = True
    try:
        torch.testing.assert_close(
            grad_gems,
            grad_ref,
            atol=atol_val,
            rtol=rtol_val,
            msg=lambda m: f"{name}: {m}" if name else m,
        )
    except AssertionError:
        passed = False
        raise
    finally:
        logger.info(
            "accuracy_check",
            extra={
                "test_data": {
                    "event": "accuracy_check",
                    "check_type": "gradient_comparison",
                    "name": name,
                    "dtype": str(dtype),
                    "num_ops": num_ops,
                    "expected_atol": atol_val,
                    "expected_rtol": rtol_val,
                    "actual_max_error": max_err,
                    "actual_mean_error": mean_err,
                    "passed": passed,
                }
            },
        )


def assert_numerical_consistency(output_gems, output_ref, name=""):
    """Assert FlagGems and PyTorch produce NaN/Inf in the same positions.

    This is the primary assertion for **Mode 4 (numerical stability +
    behaviour consistency)**.  For non-NaN positions the values must also
    be close (using a generous tolerance since edge-case inputs often
    amplify differences).

    Args:
        output_gems: Result from FlagGems (GPU).
        output_ref: Result from PyTorch reference.
        name: Human-readable label for error messages.
    """
    gems = output_gems.detach().float()
    ref = output_ref.detach().float()
    if gems.device != ref.device:
        ref = ref.to(gems.device)

    nan_gems = torch.isnan(gems)
    nan_ref = torch.isnan(ref)

    nan_match = True
    non_nan_close = True
    error_detail = ""

    # Where ref has NaN, gems should also have NaN.
    ref_nan_but_gems_not = nan_ref & ~nan_gems
    if ref_nan_but_gems_not.any():
        count = ref_nan_but_gems_not.sum().item()
        nan_match = False
        error_detail = f"Reference has NaN at {count} positions where FlagGems does not"

    # Where ref does NOT have NaN, gems should not have NaN either.
    if nan_match:
        gems_nan_but_ref_not = nan_gems & ~nan_ref
        if gems_nan_but_ref_not.any():
            count = gems_nan_but_ref_not.sum().item()
            nan_match = False
            error_detail = (
                f"FlagGems has NaN at {count} positions where reference does not"
            )

    # For non-NaN positions, values should be close.
    if nan_match:
        valid = ~nan_gems
        if valid.any():
            try:
                torch.testing.assert_close(
                    gems[valid],
                    ref[valid],
                    atol=1e-2,
                    rtol=1e-2,
                    msg=lambda m: f"{name} (non-NaN positions): {m}" if name else m,
                )
            except AssertionError:
                non_nan_close = False
                error_detail = "Non-NaN values differ beyond tolerance"

    passed = nan_match and non_nan_close

    logger.info(
        "accuracy_check",
        extra={
            "test_data": {
                "event": "accuracy_check",
                "check_type": "numerical_consistency",
                "name": name,
                "nan_match": nan_match,
                "non_nan_close": non_nan_close,
                "passed": passed,
            }
        },
    )

    if not nan_match:
        raise AssertionError(f"{name}: {error_detail}")
    if not non_nan_close:
        raise AssertionError(f"{name}: {error_detail}")


def assert_loss_close(loss_gems, loss_ref, dtype, name=""):
    """Assert two scalar loss values are close.

    This is the primary assertion for **Mode 5 (scalar loss comparison)**.
    """
    rtol = RESOLUTION.get(dtype, 1e-3)
    # For scalar comparison a fixed atol is sufficient.
    atol = COMBO_BASE_ATOL.get(dtype, 1e-4)

    g = loss_gems.detach().float()
    r = loss_ref.detach().float()
    if g.device != r.device:
        r = r.to(g.device)

    actual_diff = (g - r).abs().item()
    passed = True
    try:
        torch.testing.assert_close(
            g,
            r,
            atol=atol,
            rtol=rtol,
            msg=lambda m: f"{name}: {m}" if name else m,
        )
    except AssertionError:
        passed = False
        raise
    finally:
        logger.info(
            "accuracy_check",
            extra={
                "test_data": {
                    "event": "accuracy_check",
                    "check_type": "loss_comparison",
                    "name": name,
                    "dtype": str(dtype),
                    "expected_atol": atol,
                    "expected_rtol": rtol,
                    "actual_diff": actual_diff,
                    "passed": passed,
                }
            },
        )
