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

from .add_rms_norm import add_rms_norm
from .bf16_paged_mqa_logits import bf16_paged_mqa_logits
from .beam_search_score import beam_search_score_ as beam_search_score_
from .bincount import bincount
from .concat_and_cache_mla import concat_and_cache_mla
from .cross_entropy_loss import cross_entropy_loss
from .deepseek_v4_attention_compute_global_topk_indices_and_lens import (
    compute_global_topk_indices_and_lens,
)
from .flash_mla import flash_mla
from .flashmla_sparse import flash_mla_sparse_fwd
from .fused_add_rms_norm import fused_add_rms_norm
from .fused_deepseek_v4_qnorm_rope_kv_rope_insert import (
    fused_deepseek_v4_qnorm_rope_kv_rope_insert,
)
from .hc_head_fused_kernel import hc_head_fused_kernel
from .hc_split_sinkhorn import hc_split_sinkhorn
from .geglu import dgeglu, geglu
from .gelu_and_mul import gelu_and_mul
from .instance_norm import instance_norm
from .moe_align_block_size import moe_align_block_size, moe_align_block_size_triton
from .outer import outer
from .pack_seq import pack_seq_triton
from .reglu import dreglu, reglu
from .reshape_and_cache import reshape_and_cache
from .reshape_and_cache_flash import reshape_and_cache_flash
from .rotary_embedding import apply_rotary_pos_emb
from .rwkv_ka_fusion import rwkv_ka_fusion
from .rwkv_mm_sparsity import rwkv_mm_sparsity
from .silu_and_mul import silu_and_mul, silu_and_mul_out
from .silu_and_mul_with_clamp import (
    silu_and_mul_with_clamp,
    silu_and_mul_with_clamp_out,
)
from .skip_layernorm import skip_layer_norm
from .sparse_attention import sparse_attn_triton
from .sparse_mla import triton_sparse_mla_fwd_interface
from .top_k_per_row_prefill import top_k_per_row_prefill
from .topk_softplus_sqrt import topk_softplus_sqrt
from .topk_softmax import topk_softmax
from .unpack_seq import unpack_seq_triton
from .weight_norm import weight_norm

__all__ = [
    "add_rms_norm",
    "bf16_paged_mqa_logits",
    "compute_global_topk_indices_and_lens",
    "flash_mla_sparse_fwd",
    "fused_deepseek_v4_qnorm_rope_kv_rope_insert",
    "hc_head_fused_kernel",
    "hc_split_sinkhorn",
    "pack_seq_triton",
    "top_k_per_row_prefill",
    "triton_sparse_mla_fwd_interface",
    "unpack_seq_triton",
    "apply_rotary_pos_emb",
    "skip_layer_norm",
    "fused_add_rms_norm",
    "silu_and_mul",
    "silu_and_mul_out",
    "silu_and_mul_with_clamp",
    "silu_and_mul_with_clamp_out",
    "geglu",
    "dgeglu",
    "gelu_and_mul",
    "cross_entropy_loss",
    "outer",
    "instance_norm",
    "weight_norm",
    "concat_and_cache_mla",
    "reshape_and_cache",
    "moe_align_block_size",
    "moe_align_block_size_triton",
    "reshape_and_cache_flash",
    "flash_mla",
    "topk_softplus_sqrt",
    "topk_softmax",
    "rwkv_ka_fusion",
    "rwkv_mm_sparsity",
    "dreglu",
    "reglu",
    "sparse_attn_triton",
    "bincount",
]
