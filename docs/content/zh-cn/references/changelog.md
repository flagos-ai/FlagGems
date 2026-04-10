---
title: 项目变更历史
weight: 90
---

<!--
# Change History
-->
# 变更历史

## v4.3

**发布日期**：待定

- 新增数学算子：
  `acos`,
  `ceil`,
  `ceil_`,
  `ceil_out`,
  `equal`,
  `logical_and_`,
  `logical_or_`
- 新增 BLAS 算子：
  `bmm.out`
- 新增 Distribution 算子：
  `normal_`
- 新增神经网络算子：
  `one_hot`,
  `triu_`,
  `upsample_linear1d`,
  `upsample_nearest1d`,
- 新增张量算子：
  `unfold_backward`,
  `zero_`
- 移除 Reduction 算子：
  `moe_sum`

## v4.2

**发布日期**：2026-01-04

- 此版本包含 216 个算子，参见[支持的算子列表](/FlagGems/zh-cn/reference/operators/)
- 新增神经网络算子：
  `avg_pool2d`,
  `avg_pool2d_backward`,
  `dgeglu` (*alpha*),
  `dreglu` (*alpha*),
  `geglu` (*alpha*),
  `reglu` (*alpha*),
- 新增 BLAS 算子：
  `baddbmm`,
- 新增卷积算子：
  `conv1d`,
  `conv2d`,
  `conv3d`
- 新增张量算子：
  `copy_`,
  `masked_scatter`,
  `masked_scatter_`,
  `per_token_group_quant_fp8` (*alpha*),
  `scatter_add_`,
  `to_copy`
- 新增 Reduction 算子：
  `moe_sum`,
  `scaled_softmax_backward`,
  `scaled_softmax_forward`
- 新增数学算子：
  `exp_out`,
  `tan`,
  `tan_`,
  `true_divide_out`
- 将张量算子 `continuous` 提升为 Beta 状态
- 将 Reduction 算子 `index` 提升为稳定状态
- 之前的 `upsample` 算子被分为
  `upsample_nearest2d` 和 `upsample_bicubic2d_aa` 两个算子

## v4.1

**发布日期**：2025-11-01

- 新增两个融合形式的 RWKV 算子：
  `rwkv_ka_fusion`、
  `rwkv_mm_sparsity`，
  被 [BlinkDL/Albatross:faster_251101](https://github.com/BlinkDL/Albatross/tree/main/faster_251101)
  RWKV 项目采纳

## v4.0

**发布日期**：2025-10-31

- 总计支持 202 个算子
- 新增线性代数算子：
  `addcdiv`,
  `addcmul`,
  `addmv`,
  `addmv_out`,
  `addr`
- 新增 BLAS 算子：
  `addmm_out`
- 新增数学算子：
  `atan`,
  `atan_`,
  `bitwise_left_shift`,
  `bitwise_right_shift`,
  `clamp_min`,
  `clamp_min_`,
  `exp2`,
  `exp2_`,
  `sqrt_`
- 新增神经网络算子：
  `celu`,
  `celu_`,
  `elu_`,
  `elu_backward`,
  `get_scheduler_metadata`,
  `glu_backward`,
  `moe_align_block_size`,
  `softplus`
- 新增 Reduction 算子：
  `index` (*beta*),
  `std`,
  `trace`
- 新增张量算子：
  `index_add_`,
  `logspace`,
  `max_pool2d_backward`,
  `max_pool2d_with_indices`,
  `topk_softmax`
- Triton JIT C++ 运行时现在包含以下预编译内核：
  `add`,
  `addmm`,
  `argmax`,
  `bmm`,
  `bmm_out`,
  `cat`,
  `contiguous`,
  `embedding`,
  `exponential_`,
  `fill`,
  `flash_attn_varlen_func`,
  `fused_add_rms_norm`,
  `max`,
  `mm`,
  `nonzero`,
  `reshape_and_cache_flash`,
  `rms_norm`,
  `rms_norm_backward`,
  `rotary_embedding`,
  `softmax`,
  `sum`,
  `topk`,
  `zeros`

## v3.0

**发布日期**：2025-07-14

- 总计支持 184 个算子，包含在大模型推理中常用的定制算子
- 新增硬件平台支持：昇腾（Ascend）、AIPU 等
- 新增对 vLLM 框架的兼容，通过对 DeepSeek 模型的推理验证
- 新增 BLAS 算子：
  `dot`,
  `mm_out`
- 新增张量算子：
  `index_put_`,
  `scatter_`,
  `sort_stable`
- 新增线性代数算子：
  `lerp`,
  `sum_dim_out`,
  `sum_out`
- Added math operators:
  `angle`,
  `nan_to_num`,
  `polar`,
  `tanh_backward`
- 新增 Reduction 算子：
  `cummax`,
  `cumsum_out`,
  `eye`,
  `index` (*alpha*),
  `layer_norm_backward`,
  `softmax_backward`
- 新增神经网络算子：
  `batch_norm`,
  `batch_norm_backward`,
  `dropout_backward`,
  `embedding_backward`,
  `flash_attention_forward`,
  `gelu_backward`,
  `glu`,
  `group_norm_backward`,
  `log_softmax` (*stable*),
  `log_softmax_backward`,
  `sigmoid_backward`,
  `silu_backward`,
  `threshold`,
  `threshold_backward`,
  `weight_norm_interface_backward`
- 移除神经网络算子：
  `cross_entropy_loss`,
  `instance_norm`
- 移除线性代数算子：
  `outer` (*fused*),

## v2.2

**发布日期**：2025-04-17

- 新增 BLAS 算子：
  `vdot`
- 新增张量算子：
  `cat`,
  `constant_pad_nd`,
  `contiguous` (*alpha*),
  `diag`,
  `diag_embed`,
  `fill`,
  `fill_`,
  `hstack`,
  `index_add`,
  `index_put`,
  `isin`,
  `kron`,
  `linspace`,
  `masked_fill`,
  `masked_fill_`,
  `quantile`,
  `repeat_interleave_self_tensor`,
  `repeat_interleave_tensor`,
  `scatter`,
  `select_scatter`,
  `slice_scatter`,
  `sort`,
  `stack`,
  `vstack`,
  `where_out` (*stable*),
- 新增线性代数算子：
  `diagonal_backward`
- 新增卷积算子：
  `_conv_depthwise2d`,
- 新增神经网络算子：
  `_upsample_bicubic2d_aa`,
  `elu`,
  `gelu_`,
  `instance_norm` (*fused*),
  `mse_loss`,
  `nll_loss_backward`,
  `nll_loss_forward`,
  `nll_loss2d_backward`,
  `nll_loss2d_forward`,
  `relu_`,
  `scaled_dot_product_attention`,
  `scaled_dot_product_attention_backward`,
  `scaled_dot_product_attention_forward`,
  `sigmoid_`,
  `silu_`,
  `upsample_nearest2d`,
  `weight_norm_interface`,
- 新增数学算子：
  `abs_`,
  `add_`,
  `bitwise_and_`,
  `bitwise_not_`,
  `bitwise_or_`,
  `clamp_`,
  `cos_`,
  `div_mode_`,
  `exp_`,
  `floor_divide_`,
  `log`,
  `log_sigmoid`,
  `logical_and`,
  `logical_not`,
  `logical_or`,
  `logical_xor`,
  `mul_`,
  `pow_`,
  `reciprocal_`,
  `remainder`,
  `remainder_`,
  `rsqrt_`,
  `sin_`,
  `sub_`,
  `tanh_`
- 新增 Reduction 算子：
  `argmin`,
  `count_nonzero`,
  `cummin`,
  `gather`,
  `gather_backward`
- 新增科学算子：
  `erf_`
- 新增 Distribution 算子：
  `randperm`

## v2.1

**发布日期**：2024-09-05

- 新增张量算子：
  `_unique2`,
  `arange`,
  `flip`,
  `full`,
  `full_like`,
  `index_select`,
  `masked_fill` (*alpha*),
  `masked_select`,
  `ones`,
  `ones_like`,
  `pad`,
  `repeat`,
  `tile`,
  `unique`,
  `where`,
  `zeros`,
  `zeros_like`
- 新增神经网络算子：
  `embedding`
- 新增数学算子：
  `allclose`,
  `floor_divide`,
  `isclose`,
  `isfinite`,
  `maximum`,
  `minimum`,
  `true_divide`,
  `trunc_divide`
- 新增 Distribution 算子：
  `exponential_`,
  `multinomial`,
  `nonzero`,
  `normal`,
  `rand`,
  `rand_like`,
  `randn`,
  `randn_like`,
  `topk`,
  `uniform_`
- 新增科学算子：
  `erf`,
  `resolve_conj`,
  `resolve_neg`

## v2.0

**发布日期**：2024-05-31

- 新增 BLAS 算子：
  `mv`,
  `outer`
- 新增数学算子：
  `bitwise_and`,
  `bitwise_not`,
  `bitwise_or`,
  `clamp`,
  `cos`,
  `eq`,
  `ge`,
  `gt`,
  `isinf`,
  `isnan`,
  `le`,
  `lt`,
  `ne`,
  `neg`,
  `or`,
  `sigmoid`
  `sin`,
  `tanh`,
- 新增 Reduction 算子：
  `all`,
  `amax`,
  `any`,
  `argmax`,
  `cross_entropy_loss`,
  `group_norm`,
  `log_softmax` (*alpha*),
  `max`,
  `min`,
  `prod`,
  `rms_norm`,
  `rms_norm_backward`,
  `rms_norm_forward`,
  `sum`,
  `var_mean`,
  `vector_norm`,
- 新增神经网络算子：
  `apply_rotary_position_embedding`,
  `fused_add_rms_norm`,
  `gelu_and_mul`,
  `outer` (*alpha*),
  `silu_and_mul`,
  `silu_and_mul_out`,
  `skip_layer_norm`

## v1.0

**发布日期**：2024-05-10

- 新增 BLAS 算子：
  `addmm`,
  `bmm`,
  `mm`
- 新增数学算子：
  `abs`,
  `add`,
  `div`,
  `dropout`,
  `exp`,
  `gelu`,
  `mul`,
  `pow`,
  `reciprocal`,
  `relu`,
  `rsqrt`,
  `silu`,
  `sub`,
  `triu`
- 新增 Reduction 算子：
  `cumsum`,
  `layer_norm`,
  `mean`,
  `softmax`
