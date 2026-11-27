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

from flag_gems.fused.act_quant import act_quant_triton
from flag_gems.fused.add_rms_norm import add_rms_norm
from flag_gems.fused.beam_search_score import beam_search_score, beam_search_score_
from flag_gems.fused.bf16_paged_mqa_logits import bf16_paged_mqa_logits
from flag_gems.fused.bincount import bincount
from flag_gems.fused.concat_and_cache_mla import concat_and_cache_mla
from flag_gems.fused.cross_entropy_loss import cross_entropy_loss
from flag_gems.fused.fused_add_rms_norm import fused_add_rms_norm
from flag_gems.fused.fused_deepseek_v4_qnorm_rope_kv_rope_insert import (
    fused_deepseek_v4_qnorm_rope_kv_rope_insert,
)
from flag_gems.fused.geglu import dgeglu, geglu
from flag_gems.fused.gelu_and_mul import gelu_and_mul
from flag_gems.fused.instance_norm import instance_norm
from flag_gems.fused.matmul_bias_activation import matmul_bias_activation
from flag_gems.fused.matmuladd import matmuladd
from flag_gems.fused.moe_load_balance_loss import moe_load_balance_loss
from flag_gems.fused.outer import outer
from flag_gems.fused.post_layer_norm_residual import post_layer_norm_residual
from flag_gems.fused.reglu import dreglu, reglu
from flag_gems.fused.reshape_and_cache_flash import reshape_and_cache_flash
from flag_gems.fused.rwkv_ka_fusion import rwkv_ka_fusion
from flag_gems.fused.rwkv_mm_sparsity import rwkv_mm_sparsity
from flag_gems.fused.stage_deepseek_v4_mega_moe_inputs import (
    stage_deepseek_v4_mega_moe_inputs,
)
from flag_gems.fused.swiglu import dswiglu, swiglu
from flag_gems.fused.weight_norm import weight_norm

__all__ = [
    "act_quant_triton",
    "add_rms_norm",
    "beam_search_score",
    "beam_search_score_",
    "bf16_paged_mqa_logits",
    "bincount",
    "concat_and_cache_mla",
    "cross_entropy_loss",
    "dgeglu",
    "dreglu",
    "dswiglu",
    "fused_add_rms_norm",
    "fused_deepseek_v4_qnorm_rope_kv_rope_insert",
    "geglu",
    "gelu_and_mul",
    "instance_norm",
    "matmul_bias_activation",
    "matmuladd",
    "moe_load_balance_loss",
    "outer",
    "post_layer_norm_residual",
    "reglu",
    "reshape_and_cache_flash",
    "rwkv_ka_fusion",
    "rwkv_mm_sparsity",
    "stage_deepseek_v4_mega_moe_inputs",
    "swiglu",
    "weight_norm",
]
