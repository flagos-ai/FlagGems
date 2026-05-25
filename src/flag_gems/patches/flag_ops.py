"""
flag_ops custom operator library registration.

This module provides enable_flag_ops() function to register custom operators
from flag_gems.fused to the torch.ops.flag_ops namespace.

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

# Operator registry: (name, schema, impl_func)
_FLAG_OPS_REGISTRY = (
    (
        "silu_and_mul",
        "silu_and_mul(Tensor A, Tensor B) -> Tensor",
        silu_and_mul,
    ),
    (
        "silu_and_mul_out",
        "silu_and_mul_out(Tensor A, Tensor B, *, Tensor(a!) out) -> Tensor(a!)",
        silu_and_mul_out,
    ),
    (
        "silu_and_mul_with_clamp",
        "silu_and_mul_with_clamp(Tensor x, Tensor y, float limit) -> Tensor",
        silu_and_mul_with_clamp,
    ),
    (
        "silu_and_mul_with_clamp_out",
        "silu_and_mul_with_clamp_out("
        "Tensor x, Tensor y, *, Tensor(a!) out, float limit) -> Tensor(a!)",
        silu_and_mul_with_clamp_out,
    ),
    (
        "cutlass_scaled_mm",
        "cutlass_scaled_mm(Tensor(a!) c, Tensor a, Tensor b, Tensor a_scale, "
        "Tensor b_scale, Tensor? bias=None) -> Tensor(a!)",
        cutlass_scaled_mm,
    ),
)


def enable_flag_ops() -> None:
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

    # Create library instances
    # Store as module-level variables to prevent garbage collection
    _flag_ops_def_lib = torch.library.Library("flag_ops", "DEF")
    _flag_ops_impl_lib = torch.library.Library("flag_ops", "IMPL")

    dispatch_key = runtime.device.dispatch_key

    # Register all operators from registry
    for name, schema, impl_func in _FLAG_OPS_REGISTRY:
        _flag_ops_def_lib.define(schema)
        _flag_ops_impl_lib.impl(name, impl_func, dispatch_key)


__all__ = ["enable_flag_ops"]
