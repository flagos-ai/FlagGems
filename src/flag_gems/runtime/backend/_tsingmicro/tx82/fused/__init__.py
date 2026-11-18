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
from .decode_attention import (
    decode_attention_fwd,
    decode_attention_fwd_grouped,
    decode_attention_fwd_normal,
    decode_attention_fwd_sglang,
)
from .extend_attention import extend_attention_fwd
from .fla import chunk_gated_delta_rule, chunk_gated_delta_rule_fwd
from .flash_mla import flash_mla
from .flashmla_sparse import flash_mla_sparse_fwd
from .fused_add_rms_norm import fused_add_rms_norm
from .fused_moe import (
    dispatch_fused_moe_kernel,
    fused_experts_impl,
    inplace_fused_experts,
    invoke_fused_moe_triton_kernel,
    outplace_fused_experts,
)
from .grouped_topk import grouped_topk
from .mhc_post import mhc_post
from .mhc_pre import mhc_pre
from .moe_align_block_size import moe_align_block_size, moe_align_block_size_triton
from .reshape_and_cache_flash import reshape_and_cache_flash
from .rotary_embedding import apply_rotary_pos_emb
from .silu_and_mul import silu_and_mul, silu_and_mul_out
from .silu_and_mul_with_clamp import (
    silu_and_mul_with_clamp,
    silu_and_mul_with_clamp_out,
)
from .top_k_per_row_decode import top_k_per_row_decode

__all__ = [
    "extend_attention_fwd",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_fwd",
    "flash_mla_sparse_fwd",
    "cross_entropy_loss",
    "decode_attention_fwd",
    "decode_attention_fwd_grouped",
    "decode_attention_fwd_normal",
    "decode_attention_fwd_sglang",
    "flash_mla",
    "fused_add_rms_norm",
    "grouped_topk",
    "mhc_post",
    "mhc_pre",
    "moe_align_block_size",
    "moe_align_block_size_triton",
    "reshape_and_cache_flash",
    "apply_rotary_pos_emb",
    "silu_and_mul",
    "silu_and_mul_out",
    "silu_and_mul_with_clamp",
    "silu_and_mul_with_clamp_out",
    "top_k_per_row_decode",
    "dispatch_fused_moe_kernel",
    "fused_experts_impl",
    "inplace_fused_experts",
    "invoke_fused_moe_triton_kernel",
    "outplace_fused_experts",
]
