"""Hopper FA3 kernel suite.

The modules in this package provide the implementation backing the historical
``ops.flash_kernel_v3`` import path, plus script-facing registry helpers for
comparing warp-specialization variants.
"""

from .registry import (
    WSVariant,
    get_variant,
    iter_variants,
    resolve_variant_names,
    variant_names,
)

__all__ = [
    "WSVariant",
    "get_variant",
    "iter_variants",
    "resolve_variant_names",
    "variant_names",
]
