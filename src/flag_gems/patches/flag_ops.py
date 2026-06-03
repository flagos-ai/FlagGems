"""
flag_ops custom operator library registration.

This module provides enable_flag_ops() function to register custom operators
from flag_gems.fused to the torch.ops.flag_ops namespace.

Usage:
    import flag_gems
    flag_gems.enable_flag_ops()
    result = torch.ops.flag_ops.silu_and_mul(x, y)
"""

from flag_gems.patches.patch_vllm_all import enable_flag_ops

__all__ = ["enable_flag_ops"]
