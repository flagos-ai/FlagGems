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
from .gemv_kernels import gemv
from .gemv_splitK_kernels import gemv_splitK
from .gemv_revsplitK_kernels import gemv_revsplitK
from .gemm_splitK_kernels import gemm_splitK
from .gemm_splitK_persistent_kernels import gemm_splitK_persistent
from .bitpack import pack_weights_over_cols

from .config import set_autotune, set_kernel_caching, AUTOTUNE, KERNEL

# Optimized kernels from QCGemV2.0
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
        gemm_INT_kernel_with_prefetch,
        gemm_optimized,
    )
    _HAS_OPTIMIZED_KERNELS = True
except ImportError as e:
    _HAS_OPTIMIZED_KERNELS = False


# Quantization utilities from QCGemV2.0
from .quant_utils import (
    WeightQuantizerMXFP,
    scale_activations_per_token,
    scale_activations_mxfp8,
    scale_activations_mxfp4,
    scale_activations_per_token_triton,
    scale_activations_mxfp8_triton_v2,
    scale_activations_mxfp4_triton_v2,
)

# Helper / model-patching utilities from QCGemV2.0
from . import helper
from .helper import (
    A16W8,
    A16Wn,
    A16W8_INT8,
    A16W8_FP8,
    A16Wn_HQQ_INT,
    A16W8_HQQ_INT,
    A16W4_HQQ_INT,
    A16W2_HQQ_INT,
    A16W1_HQQ_INT,
    A16Wn_MXFP,
    A16W8_MXFP,
    A16W4_MXFP,
    A8W8_dynamic,
    A8W8_int8_dynamic,
    A8W8_fp8_dynamic,
    A8W8_INT8_dynamic,
    A8W8_FP8_dynamic,
    A8Wn_HQQ_INT_dynamic,
    A8W4_HQQ_INT_dynamic,
    A8W2_HQQ_INT_dynamic,
    A8W8_MXFP_dynamic,
    A8Wn_MXFP_dynamic,
    A8W8_MXFP_dynamic,
    A8W4_MXFP_dynamic,
    A4W4_MXFP_dynamic,
    A4W4_NVFP_dynamic,
    A16W158_INT,
    A8W158_INT_dynamic,
    cleanup_linear,
    patch_model,
    warmup,
)

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
    "gemv",
    "gemv_splitK",
    "gemv_revsplitK",
    "gemm_splitK",
    "gemm_splitK_persistent",
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
    # Optimized kernels from QCGemV2.0
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
    "gemm_optimized",
    "_HAS_OPTIMIZED_KERNELS",
    # Split-K features
    "SPLIT_K_THRESHOLD",
    "get_split_k_factor",
    # Quantization utilities from QCGemV2.0
    "WeightQuantizerMXFP",
    "scale_activations_per_token",
    "scale_activations_mxfp8",
    "scale_activations_mxfp4",
    # Helper / model-patching utilities from QCGemV2.0
    "helper",
    "A16W8",
    "A16Wn",
    "A16W8_INT8",
    "A16W8_FP8",
    "A16Wn_HQQ_INT",
    "A16W8_HQQ_INT",
    "A16W4_HQQ_INT",
    "A16W2_HQQ_INT",
    "A16W1_HQQ_INT",
    "A16Wn_MXFP",
    "A16W8_MXFP",
    "A16W4_MXFP",
    "A8W8_dynamic",
    "A8W8_int8_dynamic",
    "A8W8_fp8_dynamic",
    "A8W8_INT8_dynamic",
    "A8W8_FP8_dynamic",
    "A8Wn_HQQ_INT_dynamic",
    "A8W4_HQQ_INT_dynamic",
    "A8W2_HQQ_INT_dynamic",
    "A8W8_MXFP_dynamic",
    "A8Wn_MXFP_dynamic",
    "A8W8_MXFP_dynamic",
    "A8W4_MXFP_dynamic",
    "A4W4_MXFP_dynamic",
    "A4W4_NVFP_dynamic",
    "A16W158_INT",
    "A8W158_INT_dynamic",
    "cleanup_linear",
    "patch_model",
    "warmup",
]
