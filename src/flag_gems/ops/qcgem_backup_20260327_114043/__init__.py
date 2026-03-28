# SPDX-License-Identifier: Apache-2.0
# QC-GEM: Quantized Computing GEM library for FlagGems
# Based on GemLite by Mobius Labs GmbH

from .core import (
    QCGeMLinear,
    QCGeMLinear as GemLiteLinear,
    qcgem_forward,
    qcgem_forward_precomputed,
    qcgem_mm,
    qcgem_linear,
    DType,
    QCGEM_ACC_DTYPE,
    QCGEM_TRITON_CONFIG_CACHE,
    QCGEM_TRITON_MAPPING,
    QCGEM_MATMUL_TYPES,
    QCGEM_MATMUL_TYPES_MAPPING,
    # New optimization features
    SPLIT_K_THRESHOLD,
    get_split_k_factor,
    PrecomputedWeightCache,
    get_precomputed_cache,
    precompute_weights,
)

from .dtypes import (
    DType,
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
)

from .gemm_kernels import gemm
from .bitpack import pack_weights_over_cols

from .config import set_autotune, set_kernel_caching, AUTOTUNE, KERNEL

# Optimized kernels (NEW!)
try:
    from .gemm_kernels_optimized import (
        gemm_INT_kernel_optimized,
        gemm_forward_optimized,
        dequantize_optimized,
        PrecomputedWeightCache,
        get_precomputed_cache,
        get_fast_optimized_config_nvidia,
        get_splitk_autotune_config_nvidia,
        get_small_n_autotune_config_nvidia,
        IS_HOPPER,
        IS_BLACKWELL,
        gemm_INT_kernel_with_prefetch,  # NEW: Shared memory prefetch kernel
    )
    _HAS_OPTIMIZED_KERNELS = True
except ImportError as e:
    _HAS_OPTIMIZED_KERNELS = False


__version__ = "1.0.0"

__all__ = [
    # Core
    "QCGeMLinear",
    "GemLiteLinear",
    "qcgem_forward",
    "qcgem_forward_precomputed",
    "qcgem_mm",
    "qcgem_linear",
    # DTypes
    "DType",
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
    # Kernels
    "gemm",
    "pack_weights_over_cols",
    # Config
    "set_autotune",
    "set_kernel_caching",
    "AUTOTUNE",
    "KERNEL",
    # Constants
    "QCGEM_ACC_DTYPE",
    "QCGEM_TRITON_CONFIG_CACHE",
    "QCGEM_TRITON_MAPPING",
    "QCGEM_MATMUL_TYPES",
    "QCGEM_MATMUL_TYPES_MAPPING",
    # Optimized kernels
    "gemm_INT_kernel_optimized",
    "gemm_forward_optimized",
    "dequantize_optimized",
    "PrecomputedWeightCache",
    "get_precomputed_cache",
    "precompute_weights",
    "get_fast_optimized_config_nvidia",
    "get_splitk_autotune_config_nvidia",
    "get_small_n_autotune_config_nvidia",
    "IS_HOPPER",
    "IS_BLACKWELL",
    "gemm_INT_kernel_with_prefetch",
    "_HAS_OPTIMIZED_KERNELS",
    # Split-K features
    "SPLIT_K_THRESHOLD",
    "get_split_k_factor",
]
