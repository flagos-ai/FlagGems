# SPDX-License-Identifier: Apache-2.0
# QC-GEM1: Compatibility module for GemLite interface
# This module provides backward compatibility with the original GemLite API

# Use relative imports to work within the flag_gems.ops package
from .qcgem import (
    QCGeMLinear as GemLiteLinearTriton,
    QCGeMLinear,
    DType,
    QCGEM_ACC_DTYPE as GEMLITE_ACC_DTYPE,
    QCGEM_TRITON_KERNELS as GEMLITE_TRITON_KERNELS,
    QCGEM_TRITON_MAPPING as GEMLITE_TRITON_MAPPING,
    QCGEM_MATMUL_TYPES as GEMLITE_MATMUL_TYPES,
    QCGEM_MATMUL_TYPES_MAPPING as GEMLITE_MATMUL_TYPES_MAPPING,
    QCGEM_TRITON_CONFIG_CACHE as GEMLITE_TRITON_CONFIG_CACHE,
    gemm,
    gemm_splitK,
    gemm_splitK_persistent,
    gemv,
    gemv_splitK,
    gemv_revsplitK,
    set_autotune,
    set_kernel_caching,
    AUTOTUNE,
    KERNEL,
    DTYPE_TO_TORCH,
    TORCH_TO_DTYPE,
    DTYPE_TO_TRITON,
    TORCH_DTYPE_TO_TRITON,
    PACKING_BITWIDTH_TO_TORCH_DTYPE,
    FP8_DTYPES,
    FP8_INT8_DTYPES,
    MX_DTYPES,
    MX_DTYPES_val,
    is_mx_dtype,
    pack_weights_over_cols,
)

# Import config functions
from .qcgem.config import set_autotune, set_kernel_caching, AUTOTUNE, KERNEL

# GemLite-specific functions
from .qcgem.core import get_matmul_type, qcgem_forward

# Additional compatibility exports
try:
    from .qcgem.gemm_kernels_optimized import (
        gemm_INT_kernel_optimized,
        gemm_forward_optimized,
    )
except ImportError:
    gemm_INT_kernel_optimized = None
    gemm_forward_optimized = None


# Placeholder functions for compatibility
def set_autotune_setting(setting):
    """Set autotune setting (compatibility function)."""
    AUTOTUNE.GEMM = setting
    AUTOTUNE.GEMV = setting
    AUTOTUNE.GEMM_SPLITK = setting
    AUTOTUNE.GEMV_SPLITK = setting
    AUTOTUNE.GEMV_REVSPLITK = setting


def set_packing_bitwidth(bitwidth):
    """Set packing bitwidth (compatibility function)."""
    pass  # Not needed in current implementation


def set_acc_dtype(dtype):
    """Set accumulator dtype (compatibility function)."""
    pass  # Not needed in current implementation


def forward_functional(*args, **kwargs):
    """Forward functional (compatibility function)."""
    return qcgem_forward(*args, **kwargs)


def get_default_gemv():
    """Get default GEMV kernel name."""
    return "GEMV"


def get_default_cache_config():
    """Get default cache configuration."""
    return {"enable": True, "max_size": 256}


def load_config(config):
    """Load configuration."""
    pass  # Placeholder


def cache_config(config):
    """Set cache configuration."""
    pass  # Placeholder


def reset_config():
    """Reset configuration to defaults."""
    pass  # Placeholder


def helper():
    """Helper function."""
    return "QC-GEM v1.0.0"


# Placeholder for quant_utils
class quant_utils:
    """Placeholder for quantization utilities."""
    pass


__version__ = "1.0.0"

__all__ = [
    "GemLiteLinearTriton",
    "GemLiteLinear",
    "DType",
    "GEMLITE_ACC_DTYPE",
    "GEMLITE_TRITON_KERNELS",
    "GEMLITE_TRITON_MAPPING",
    "GEMLITE_MATMUL_TYPES",
    "GEMLITE_MATMUL_TYPES_MAPPING",
    "GEMLITE_TRITON_CONFIG_CACHE",
    "set_autotune_setting",
    "set_packing_bitwidth",
    "set_acc_dtype",
    "set_autotune",
    "forward_functional",
    "get_matmul_type",
    "get_default_gemv",
    "get_default_cache_config",
    "load_config",
    "cache_config",
    "reset_config",
    "helper",
    "quant_utils",
    "gemm",
    "gemm_splitK",
    "gemm_splitK_persistent",
    "gemv",
    "gemv_splitK",
    "gemv_revsplitK",
    "set_kernel_caching",
    "AUTOTUNE",
    "KERNEL",
    "DTYPE_TO_TORCH",
    "TORCH_TO_DTYPE",
    "DTYPE_TO_TRITON",
    "TORCH_DTYPE_TO_TRITON",
    "PACKING_BITWIDTH_TO_TORCH_DTYPE",
    "FP8_DTYPES",
    "FP8_INT8_DTYPES",
    "MX_DTYPES",
    "MX_DTYPES_val",
    "is_mx_dtype",
    "pack_weights_over_cols",
    "gemm_INT_kernel_optimized",
    "gemm_forward_optimized",
]
