"""
ARM CPU fused_add_rms_norm wrapper.

Wraps the _arm/fused/fused_add_rms_norm.py Triton kernel so it can be used
as a drop-in replacement for flag_gems.fused_add_rms_norm on ARM64 CPU.

Standalone rms_norm (without residual add) was removed: A/B measurement on
Qwen3-1.7B INT8 decode showed no measurable benefit over ATen's native
Qwen3RMSNorm.forward. See _arm/fused/fused_add_rms_norm.py for the note.
"""

from flag_gems.runtime.backend._arm.fused.fused_add_rms_norm import (
    fused_add_rms_norm as _arm_fused_add_rms_norm,
)


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """
    ARM CPU drop-in for flag_gems.fused_add_rms_norm.

    In-place: residual = x + residual; x = rms_norm(residual) * weight.
    Returns (x, residual).
    """
    return _arm_fused_add_rms_norm(x, residual, normalized_shape, weight, eps)
