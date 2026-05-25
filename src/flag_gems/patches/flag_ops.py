"""
flag_ops custom operator library registration.

This module provides enable_flag_ops() function to register three operators
from flag_gems.fused to the torch.ops.flag_ops namespace:
- silu_and_mul
- silu_and_mul_with_clamp
- cutlass_scaled_mm

Usage:
    import flag_gems
    flag_gems.enable_flag_ops()
    result = torch.ops.flag_ops.silu_and_mul(x, y)
"""

import torch

import flag_gems.runtime as runtime
from flag_gems.fused import (
    cutlass_scaled_mm,
    silu_and_mul,
    silu_and_mul_out,
    silu_and_mul_with_clamp,
    silu_and_mul_with_clamp_out,
)

_registered = False
_flag_ops_def_lib = None
_flag_ops_impl_lib = None


def enable_flag_ops():
    """
    Register flag_ops custom operators to torch.ops.flag_ops namespace.

    This function is idempotent - calling it multiple times will only register once.

    Registered operators:
    - torch.ops.flag_ops.silu_and_mul(A, B) -> Tensor
    - torch.ops.flag_ops.silu_and_mul_out(A, B, *, out) -> Tensor
    - torch.ops.flag_ops.silu_and_mul_with_clamp(x, y, limit) -> Tensor
    - torch.ops.flag_ops.silu_and_mul_with_clamp_out(x, y, *, out, limit) -> Tensor
    - torch.ops.flag_ops.cutlass_scaled_mm(c, a, b, a_scale, b_scale, bias=None) -> Tensor
    """
    global _registered, _flag_ops_def_lib, _flag_ops_impl_lib
    if _registered:
        return
    _registered = True

    # Create library instance for operator definition
    # Store as module-level variable to prevent garbage collection
    _flag_ops_def_lib = torch.library.Library("flag_ops", "DEF")

    # Define operator signatures
    _flag_ops_def_lib.define(
        "silu_and_mul(Tensor A, Tensor B) -> Tensor",
    )

    _flag_ops_def_lib.define(
        "silu_and_mul_out(Tensor A, Tensor B, *, Tensor(a!) out) -> Tensor(a!)",
    )

    _flag_ops_def_lib.define(
        "silu_and_mul_with_clamp(Tensor x, Tensor y, float limit) -> Tensor",
    )

    _flag_ops_def_lib.define(
        "silu_and_mul_with_clamp_out(Tensor x, Tensor y, *, Tensor(a!) out, float limit) -> Tensor(a!)",
    )

    _flag_ops_def_lib.define(
        "cutlass_scaled_mm(Tensor(a!) c, Tensor a, Tensor b, Tensor a_scale, "
        "Tensor b_scale, Tensor? bias=None) -> Tensor(a!)",
    )

    # Create library instance for implementation registration
    # Store as module-level variable to prevent garbage collection
    _flag_ops_impl_lib = torch.library.Library("flag_ops", "IMPL")

    # Get dispatch key from runtime
    dispatch_key = runtime.device.dispatch_key

    # Register implementations
    _flag_ops_impl_lib.impl("silu_and_mul", silu_and_mul, dispatch_key)
    _flag_ops_impl_lib.impl("silu_and_mul_out", silu_and_mul_out, dispatch_key)
    _flag_ops_impl_lib.impl(
        "silu_and_mul_with_clamp", silu_and_mul_with_clamp, dispatch_key
    )
    _flag_ops_impl_lib.impl(
        "silu_and_mul_with_clamp_out", silu_and_mul_with_clamp_out, dispatch_key
    )
    _flag_ops_impl_lib.impl("cutlass_scaled_mm", cutlass_scaled_mm, dispatch_key)


__all__ = ["enable_flag_ops"]
