# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .cross_entropy_loss import cross_entropy_loss
from .FLA import fused_recurrent_gated_delta_rule_fwd
from .flash_mla import flash_mla
from .fused_add_rms_norm import fused_add_rms_norm
from .fused_moe import (
    dispatch_fused_moe_kernel,
    fused_experts_impl,
    inplace_fused_experts,
    invoke_fused_moe_triton_kernel,
    outplace_fused_experts,
)
from .gelu_and_mul import gelu_and_mul
from .matmuladd import matmuladd
from .outer import outer
from .rwkv_mm_sparsity import rwkv_mm_sparsity
from .silu_and_mul import silu_and_mul, silu_and_mul_out
from .skip_layernorm import skip_layer_norm
from .top_k_per_row_decode import top_k_per_row_decode
from .weight_norm import weight_norm

__all__ = [
    "cross_entropy_loss",
    "dispatch_fused_moe_kernel",
    "flash_mla",
    "fused_add_rms_norm",
    "fused_experts_impl",
    "fused_recurrent_gated_delta_rule_fwd",
    "gelu_and_mul",
    "inplace_fused_experts",
    "invoke_fused_moe_triton_kernel",
    "matmuladd",
    "outer",
    "outplace_fused_experts",
    "rwkv_mm_sparsity",
    "silu_and_mul",
    "silu_and_mul_out",
    "skip_layer_norm",
    "top_k_per_row_decode",
    "weight_norm",
]
