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

# Deprecated operators migrated from FlagGems-vllm or TransformerEngine
# These operators are retained here for compatibility but may be removed in future versions

from flag_gems.deprecated.apply_repetition_penalties import apply_repetition_penalties
from flag_gems.deprecated.cp_gather_indexer_k_quant_cache import (
    cp_gather_indexer_k_quant_cache,
)
from flag_gems.deprecated.cutlass_scaled_mm import cutlass_scaled_mm
from flag_gems.deprecated.deepseek_v4_attention_combine_topk_swa_indices import (
    combine_topk_swa_indices,
)
from flag_gems.deprecated.deepseek_v4_attention_compute_global_topk_indices_and_lens import (
    compute_global_topk_indices_and_lens,
)
from flag_gems.deprecated.deepseek_v4_attention_dequantize_and_gather_k_cache import (
    dequantize_and_gather_k_cache,
)
from flag_gems.deprecated.deepseek_v4_attention_fused_q_kv_rmsnorm import (
    fused_q_kv_rmsnorm,
)
from flag_gems.deprecated.DSA.bin_topk import bucket_sort_topk
from flag_gems.deprecated.DSA.sparse_mla import triton_sparse_mla_fwd_interface
from flag_gems.deprecated.flash_mla import flash_mla
from flag_gems.deprecated.flash_mla_with_kvcache import flash_mla_with_kvcache
from flag_gems.deprecated.flashmla_sparse import flash_mla_sparse_fwd
from flag_gems.deprecated.fp8_fp4_mega_moe import (
    fp8_fp4_mega_moe,
    fp8_fp4_mega_moe_torch_ref,
)
from flag_gems.deprecated.fp8_fp4_mqa_logits import fp8_fp4_mqa_logits
from flag_gems.deprecated.fp8_fp4_paged_mqa_logits import fp8_fp4_paged_mqa_logits
from flag_gems.deprecated.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert import (
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
)
from flag_gems.deprecated.fused_indexer_q_rope_quant import fused_indexer_q_rope_quant
from flag_gems.deprecated.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant
from flag_gems.deprecated.fused_moe import (
    dispatch_fused_moe_kernel,
    fused_experts_impl,
    inplace_fused_experts,
    invoke_fused_moe_triton_kernel,
    outplace_fused_experts,
)
from flag_gems.deprecated.grouped_topk import grouped_topk
from flag_gems.deprecated.indexer_k_quant_and_cache import indexer_k_quant_and_cache
from flag_gems.deprecated.mhc import (
    hc_head_fused_kernel,
    hc_head_fused_kernel_ref,
    hc_split_sinkhorn,
    mhc_bwd,
    mhc_bwd_ref,
    mhc_post,
    mhc_pre,
    sinkhorn_forward,
)
from flag_gems.deprecated.moe_align_block_size import (
    moe_align_block_size,
    moe_align_block_size_triton,
)
from flag_gems.deprecated.moe_sum import moe_sum
from flag_gems.deprecated.mrope import mrope
from flag_gems.deprecated.pack_seq import pack_seq_triton
from flag_gems.deprecated.reshape_and_cache import reshape_and_cache
from flag_gems.deprecated.rotary_embedding import apply_rotary_pos_emb
from flag_gems.deprecated.silu_and_mul import silu_and_mul, silu_and_mul_out
from flag_gems.deprecated.silu_and_mul_with_clamp import (
    silu_and_mul_with_clamp,
    silu_and_mul_with_clamp_out,
)
from flag_gems.deprecated.skip_layernorm import skip_layer_norm
from flag_gems.deprecated.sparse_attention import sparse_attn_triton
from flag_gems.deprecated.top_k_per_row_decode import top_k_per_row_decode
from flag_gems.deprecated.top_k_per_row_prefill import top_k_per_row_prefill
from flag_gems.deprecated.topk_softmax import topk_softmax
from flag_gems.deprecated.topk_softplus_sqrt import topk_softplus_sqrt
from flag_gems.deprecated.unpack_seq import unpack_seq_triton

__all__ = [
    "apply_repetition_penalties",
    "apply_rotary_pos_emb",
    "bucket_sort_topk",
    "combine_topk_swa_indices",
    "compute_global_topk_indices_and_lens",
    "cp_gather_indexer_k_quant_cache",
    "cutlass_scaled_mm",
    "dequantize_and_gather_k_cache",
    "dispatch_fused_moe_kernel",
    "flash_mla",
    "flash_mla_sparse_fwd",
    "flash_mla_with_kvcache",
    "fp8_fp4_mega_moe",
    "fp8_fp4_mega_moe_torch_ref",
    "fp8_fp4_mqa_logits",
    "fp8_fp4_paged_mqa_logits",
    "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
    "fused_experts_impl",
    "fused_indexer_q_rope_quant",
    "fused_inv_rope_fp8_quant",
    "fused_q_kv_rmsnorm",
    "grouped_topk",
    "hc_head_fused_kernel",
    "hc_head_fused_kernel_ref",
    "hc_split_sinkhorn",
    "indexer_k_quant_and_cache",
    "inplace_fused_experts",
    "invoke_fused_moe_triton_kernel",
    "mhc_bwd",
    "mhc_bwd_ref",
    "mhc_post",
    "mhc_pre",
    "moe_align_block_size",
    "moe_align_block_size_triton",
    "moe_sum",
    "mrope",
    "outplace_fused_experts",
    "pack_seq_triton",
    "reshape_and_cache",
    "silu_and_mul",
    "silu_and_mul_out",
    "silu_and_mul_with_clamp",
    "silu_and_mul_with_clamp_out",
    "sinkhorn_forward",
    "skip_layer_norm",
    "sparse_attn_triton",
    "top_k_per_row_decode",
    "top_k_per_row_prefill",
    "topk_softmax",
    "topk_softplus_sqrt",
    "triton_sparse_mla_fwd_interface",
    "unpack_seq_triton",
]


# The FLA-based operators pull in a large Triton kernel tree (about 70% of this
# package's import time), so they are resolved on first access instead of at
# import. They are deliberately left out of ``__all__`` so that
# ``from flag_gems.deprecated import *`` does not trigger the import.
#
# name -> (submodule, attribute). ``chunk_gated_delta_rule`` is both a submodule
# and a function name, so the attribute is always looked up on the imported
# submodule rather than on this package, whose attribute may be the submodule.
_LAZY_EXPORTS = {
    "chunk_gated_delta_rule": ("chunk_gated_delta_rule", "chunk_gated_delta_rule"),
    "chunk_gated_delta_rule_fwd": ("FLA", "chunk_gated_delta_rule_fwd"),
    "fused_recurrent_gated_delta_rule_fwd": (
        "FLA",
        "fused_recurrent_gated_delta_rule_fwd",
    ),
}


def _resolve_lazy(name):
    import importlib

    submodule, attr = _LAZY_EXPORTS[name]
    value = getattr(importlib.import_module(f"{__name__}.{submodule}"), attr)
    globals()[name] = value
    return value


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        return _resolve_lazy(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
