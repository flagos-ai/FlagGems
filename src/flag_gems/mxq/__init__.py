# SPDX-License-Identifier: Apache-2.0
# __init__.py
# QC-MoE W8A16 模块

from .fused_moe_mxq import (
    fused_moe,
    FusedMoELinear,
    create_moe_op,
)

from .ultis import (
    QuantConfig,
    QuantMode,
    quantize_weights_moe,
    fp16_moe_reference,
    fp16_moe_w1_only_reference,
    calc_moe_tflops,
    calc_moe_gbps,
    verify_moe_accuracy,
    verify_moe_w1_accuracy,
    QWEN3_SHAPES,
    QWEN3_DEFAULT_CONFIG,
    dequantize_w8a16,
)

__all__ = [
    "fused_moe",
    "FusedMoELinear",
    "create_moe_op",
    "QuantConfig",
    "QuantMode",
    "quantize_weights_moe",
    "fp16_moe_reference",
    "fp16_moe_w1_only_reference",
    "calc_moe_tflops",
    "calc_moe_gbps",
    "verify_moe_accuracy",
    "verify_moe_w1_accuracy",
    "QWEN3_SHAPES",
    "QWEN3_DEFAULT_CONFIG",
    "dequantize_w8a16",
]
