# Fused/Deprecated 文件夹重组总结

## 更改概述

根据新的策略，我们重新组织了 `fused` 和 `deprecated` 文件夹：
- **保留在 deprecated**: 与 FlagGems-vllm 重复的算子或 TransformerEngine (TE) 算子
- **移回 fused**: 其他所有 fused 算子

## 移动的文件（24 个）

从 `src/flag_gems/deprecated/` 移动到 `src/flag_gems/fused/`:

1. act_quant.py
2. add_rms_norm.py
3. beam_search_score.py
4. bf16_paged_mqa_logits.py
5. bincount.py
6. concat_and_cache_mla.py
7. cross_entropy_loss.py
8. fused_add_rms_norm.py
9. fused_deepseek_v4_qnorm_rope_kv_rope_insert.py
10. geglu.py
11. gelu_and_mul.py
12. instance_norm.py
13. matmul_bias_activation.py
14. matmuladd.py
15. moe_load_balance_loss.py
16. outer.py
17. post_layer_norm_residual.py
18. reglu.py
19. reshape_and_cache_flash.py
20. rwkv_ka_fusion.py
21. rwkv_mm_sparsity.py
22. stage_deepseek_v4_mega_moe_inputs.py
23. swiglu.py
24. weight_norm.py

## 移动的算子（28 个函数）

包括上述文件中定义的 28 个函数/算子：

- act_quant_triton
- add_rms_norm
- beam_search_score
- beam_search_score_
- bf16_paged_mqa_logits
- bincount
- concat_and_cache_mla
- cross_entropy_loss
- dgeglu
- dreglu
- dswiglu
- fused_add_rms_norm
- fused_deepseek_v4_qnorm_rope_kv_rope_insert
- geglu
- gelu_and_mul
- instance_norm
- matmul_bias_activation
- matmuladd
- moe_load_balance_loss
- outer
- post_layer_norm_residual
- reglu
- reshape_and_cache_flash
- rwkv_ka_fusion
- rwkv_mm_sparsity
- stage_deepseek_v4_mega_moe_inputs
- swiglu
- weight_norm

## 保留在 deprecated 的算子（40 个 vLLM 算子）

这些算子与 FlagGems-vllm 重复，保留在 `deprecated` 目录：

1. apply_repetition_penalties
2. combine_topk_swa_indices
3. compute_global_topk_indices_and_lens
4. cp_gather_indexer_k_quant_cache
5. cutlass_scaled_mm
6. dequantize_and_gather_k_cache
7. dispatch_fused_moe_kernel
8. flash_mla
9. flash_mla_sparse_fwd
10. fp8_fp4_mega_moe
11. fp8_fp4_mqa_logits
12. fp8_fp4_paged_mqa_logits
13. fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
14. fused_experts_impl
15. fused_indexer_q_rope_quant
16. fused_inv_rope_fp8_quant
17. fused_marlin_moe_w4a16_int4
18. fused_marlin_moe_w4a16_mxfp4
19. fused_marlin_moe_w8a16_int8
20. fused_q_kv_rmsnorm
21. grouped_topk
22. hc_head_fused_kernel
23. indexer_k_quant_and_cache
24. inplace_fused_experts
25. mhc_bwd
26. mhc_post
27. mhc_pre
28. moe_align_block_size_triton
29. moe_sum
30. outplace_fused_experts
31. pack_seq_triton
32. reshape_and_cache
33. silu_and_mul
34. silu_and_mul_out
35. silu_and_mul_with_clamp
36. silu_and_mul_with_clamp_out
37. top_k_per_row_decode
38. topk_softmax
39. topk_softplus_sqrt
40. unpack_seq_triton

加上一些非 vLLM 但仍保留的算子（如 chunk_gated_delta_rule, mrope 等）。

## 更新的文件

1. **src/flag_gems/fused/__init__.py**
   - 新增了 24 个文件的导入
   - 导出 28 个函数

2. **src/flag_gems/deprecated/__init__.py**
   - 移除了已移到 fused 的算子导入
   - 保留 vLLM/TE 相关算子

3. **src/flag_gems/__init__.py**
   - 添加了 `from flag_gems.fused import *`

4. **conf/operators.yaml**
   - 添加了 28 个从 deprecated_operators.yaml 移过来的算子元数据

5. **conf/deprecated_operators.yaml**
   - 保留了 50 个 vLLM/TE 相关算子的元数据

## 未移动的特殊算子

以下算子位于子目录中，未在此次移动中处理（需要特殊处理）：

### DSA 子目录
- bucket_sort_topk (DSA/bin_topk)
- sparse_mla_fwd_interface (DSA/sparse_mla)

### FLA 子目录
- chunk_gated_delta_rule_fwd (FLA/__init__)
- fused_recurrent_gated_delta_rule_fwd (FLA/__init__)

### mhc 子目录
- hc_head_fused_kernel (mhc/__init__)
- mhc_bwd (mhc/__init__)
- mhc_post (mhc/__init__)
- mhc_pre (mhc/__init__)

注：DSA 和 FLA 子目录同时存在于 `fused/` 和 `deprecated/` 中。

## 验证结果

✅ 所有文件移动成功
✅ `flag_gems.fused` 模块导入成功（28 个函数）
✅ `flag_gems.deprecated` 模块导入成功
✅ 主模块 `flag_gems` 可以访问所有移动的函数
✅ CI 检查脚本 `check_init_exports.py` 通过

## 后续工作

1. ✅ 文件已移动
2. ✅ `__init__.py` 已更新
3. ✅ 配置文件已更新
4. ✅ 导入测试通过
5. 🔄 需要运行完整的测试套件验证功能
6. 🔄 需要运行 pre-commit 检查
7. 🔄 考虑是否需要处理 DSA/FLA/mhc 子目录

## 文档更新

AGENTS.md 第 14 行已经更新，反映了新的目录结构：
- `deprecated/` (fused kernels) → 改为 vLLM/TE 算子
- `fused/` → 保留通用 fused 算子
