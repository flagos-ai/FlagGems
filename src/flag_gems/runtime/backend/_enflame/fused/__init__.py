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

from .. import arch_version

__all__ = []

if arch_version == 300:
    from .gcu300.concat_and_cache_mla import concat_and_cache_mla  # noqa: F401
    from .gcu300.cross_entropy_loss import cross_entropy_loss
    from .gcu300.FLA import (  # noqa: F401
        chunk_gated_delta_rule_fwd,
        fused_recurrent_gated_delta_rule_fwd,
    )
    from .gcu300.flash_mla import flash_mla
    from .gcu300.fused_add_rms_norm import fused_add_rms_norm
    from .gcu300.fused_deepseek_v4_qnorm_rope_kv_rope_insert import (  # noqa: F401
        fused_deepseek_v4_qnorm_rope_kv_rope_insert,
    )
    from .gcu300.fused_moe import (  # noqa: F401
        dispatch_fused_moe_kernel,
        fused_experts_impl,
        inplace_fused_experts,
        invoke_fused_moe_triton_kernel,
        outplace_fused_experts,
    )
    from .gcu300.gelu_and_mul import gelu_and_mul
    from .gcu300.moe_align_block_size import (  # noqa: F401
        moe_align_block_size,
        moe_align_block_size_triton,
    )
    from .gcu300.moe_sum import moe_sum  # noqa: F401
    from .gcu300.mrope import mrope  # noqa: F401
    from .gcu300.rotary_embedding import apply_rotary_pos_emb  # noqa: F401
    from .gcu300.silu_and_mul import silu_and_mul
    from .gcu300.skip_layernorm import skip_layer_norm

    __all__ = [
        "apply_rotary_pos_emb",
        "silu_and_mul",
        "gelu_and_mul",
        "cross_entropy_loss",
        "flash_mla",
        "skip_layer_norm",
        "fused_add_rms_norm",
        "concat_and_cache_mla",
        "moe_align_block_size",
        "moe_align_block_size_triton",
        "moe_sum",
        "mrope",
        "fused_deepseek_v4_qnorm_rope_kv_rope_insert",
        "dispatch_fused_moe_kernel",
        "fused_experts_impl",
        "inplace_fused_experts",
        "invoke_fused_moe_triton_kernel",
        "outplace_fused_experts",
        "chunk_gated_delta_rule_fwd",
        "fused_recurrent_gated_delta_rule_fwd",
    ]
elif arch_version == 400 or arch_version == 410:
    from .gcu400.act_quant import act_quant_triton
    from .gcu400.beam_search_score import beam_search_score, beam_search_score_
    from .gcu400.bin_topk import bucket_sort_topk
    from .gcu400.bincount import bincount
    from .gcu400.chunk_gated_delta_rule import chunk_gated_delta_rule
    from .gcu400.concat_and_cache_mla import concat_and_cache_mla
    from .gcu400.cross_entropy_loss import cross_entropy_loss
    from .gcu400.cutlass_scaled_mm import (
        cutlass_scaled_mm,
        cutlass_scaled_mm_fp8,
        cutlass_scaled_mm_sm75,
        cutlass_scaled_mm_sm80,
        cutlass_scaled_mm_sm89,
        cutlass_scaled_mm_sm90,
        cutlass_scaled_mm_sm100,
        cutlass_scaled_mm_sm120,
        dispatch_scaled_mm,
        get_block_wise_smm_configs,
        get_sm_version_num,
        grouped_launch,
    )
    from .gcu400.FLA import fused_recurrent_gated_delta_rule_fwd
    from .gcu400.flash_mla import flash_mla
    from .gcu400.flash_mla_sparse_fwd import flash_mla_sparse_fwd
    from .gcu400.fp8_fp4_mega_moe import fp8_fp4_mega_moe, fp8_fp4_mega_moe_torch_ref
    from .gcu400.fp8_fp4_mqa_logits import fp8_fp4_mqa_logits
    from .gcu400.fp8_fp4_paged_mqa_logits import fp8_fp4_paged_mqa_logits
    from .gcu400.fused_add_rms_norm import fused_add_rms_norm
    from .gcu400.fused_deepseek_v4_qnorm_rope_kv_rope_insert import (  # noqa: F401
        fused_deepseek_v4_qnorm_rope_kv_rope_insert,
    )
    from .gcu400.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert import (
        fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
    )
    from .gcu400.fused_indexer_q_rope_quant import fused_indexer_q_rope_quant
    from .gcu400.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant
    from .gcu400.fused_moe import (  # noqa: F401
        dispatch_fused_moe_kernel,
        fused_experts_impl,
        inplace_fused_experts,
        invoke_fused_moe_triton_kernel,
        outplace_fused_experts,
    )
    from .gcu400.gelu_and_mul import gelu_and_mul
    from .gcu400.indexer_k_quant_and_cache import indexer_k_quant_and_cache
    from .gcu400.matmuladd import matmuladd
    from .gcu400.moe_align_block_size import (  # noqa: F401
        moe_align_block_size,
        moe_align_block_size_triton,
    )
    from .gcu400.moe_sum import moe_sum  # noqa: F401
    from .gcu400.mrope import mrope  # noqa: F401
    from .gcu400.outer import outer
    from .gcu400.reglu import dreglu, reglu
    from .gcu400.reshape_and_cache_flash import reshape_and_cache_flash
    from .gcu400.rotary_embedding import apply_rotary_pos_emb  # noqa: F401
    from .gcu400.silu_and_mul import silu_and_mul, silu_and_mul_out
    from .gcu400.silu_and_mul_with_clamp import (
        silu_and_mul_with_clamp,
        silu_and_mul_with_clamp_out,
    )
    from .gcu400.skip_layernorm import skip_layer_norm
    from .gcu400.sparse_attention import sparse_attn_triton
    from .gcu400.sparse_mla import triton_sparse_mla_fwd_interface
    from .gcu400.stage_deepseek_v4_mega_moe_inputs import (
        stage_deepseek_v4_mega_moe_inputs,
    )
    from .gcu400.top_k_per_row_prefill import (
        non_tle_top_k_per_row_prefill,
        tle_top_k_per_row_prefill,
        top_k_per_row_prefill,
    )
    from .gcu400.topk_softmax import topk_softmax

    __all__ = [
        "dispatch_fused_moe_kernel",
        "fused_experts_impl",
        "inplace_fused_experts",
        "invoke_fused_moe_triton_kernel",
        "outplace_fused_experts",
        "moe_align_block_size",
        "moe_align_block_size_triton",
        "beam_search_score",
        "beam_search_score_",
        "apply_rotary_pos_emb",
        "gelu_and_mul",
        "silu_and_mul",
        "silu_and_mul_out",
        "silu_and_mul_with_clamp",
        "silu_and_mul_with_clamp_out",
        "cross_entropy_loss",
        "concat_and_cache_mla",
        "flash_mla",
        "moe_sum",
        "mrope",
        "outer",
        "fused_add_rms_norm",
        "fused_deepseek_v4_qnorm_rope_kv_rope_insert",
        "fused_recurrent_gated_delta_rule_fwd",
        "skip_layer_norm",
        "rwkv_ka_fusion",
        "bincount",
        "sparse_attn_triton",
        "triton_sparse_mla_fwd_interface",
        "topk_softmax",
        "bucket_sort_topk",
        "flash_mla_sparse_fwd",
        "act_quant_triton",
        "get_sm_version_num",
        "get_block_wise_smm_configs",
        "grouped_launch",
        "dispatch_scaled_mm",
        "cutlass_scaled_mm_sm90",
        "cutlass_scaled_mm_sm120",
        "cutlass_scaled_mm_sm100",
        "cutlass_scaled_mm_sm89",
        "cutlass_scaled_mm_sm80",
        "cutlass_scaled_mm_sm75",
        "cutlass_scaled_mm",
        "cutlass_scaled_mm_fp8",
        "fp8_fp4_mega_moe",
        "fp8_fp4_mega_moe_torch_ref",
        "fp8_fp4_mqa_logits",
        "fp8_fp4_paged_mqa_logits",
        "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        "fused_indexer_q_rope_quant",
        "fused_inv_rope_fp8_quant",
        "indexer_k_quant_and_cache",
        "matmuladd",
        "reglu",
        "dreglu",
        "stage_deepseek_v4_mega_moe_inputs",
        "tle_top_k_per_row_prefill",
        "non_tle_top_k_per_row_prefill",
        "top_k_per_row_prefill",
        "chunk_gated_delta_rule",
        "reshape_and_cache_flash",
    ]

    # DSA bin_topk is imported directly in tests
    # (``from flag_gems.fused.DSA.bin_topk import bucket_sort_topk``), so the
    # SpecOpRegistrar namespace swap cannot reach it. Patch the attribute of the
    # already-imported module to route those calls to the GCU400 implementation,
    # the same way the kunlunxin backend does.
    def _install_gcu400_bin_topk():
        import sys

        gmod = sys.modules.get("flag_gems.fused.DSA.bin_topk")
        if gmod is not None:
            gmod.bucket_sort_topk = bucket_sort_topk

    _install_gcu400_bin_topk()
