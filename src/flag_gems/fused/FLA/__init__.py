# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from flag_gems.fused.FLA.chunk import chunk_gated_delta_rule_fwd
from flag_gems.fused.FLA.fused_recurrent import fused_recurrent_gated_delta_rule_fwd
from flag_gems.fused.FLA.fused_recurrent_fp8 import (
    dequantize_gdn_state_fp8,
    fused_recurrent_gated_delta_rule_fp8_w8a16_decode,
    quantize_gdn_state_fp8,
)

__all__ = [
    "chunk_gated_delta_rule_fwd",
    "dequantize_gdn_state_fp8",
    "fused_recurrent_gated_delta_rule_fwd",
    "fused_recurrent_gated_delta_rule_fp8_w8a16_decode",
    "quantize_gdn_state_fp8",
]
