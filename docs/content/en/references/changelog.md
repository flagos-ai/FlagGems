---
title: Changelog
weight: 90
---

# Change History

## v5.0

**Release date**: TBD

- Added math operators:
  `absolute` (_generated_),
  `acos`,
  `arcsinh` (_generated_),
  `arcsinh_` (_generated_),
  `arcsinh.out` (_generated_),
  `arctanh_` (_generated_),
  `asinh_` (_generated_),
  `ceil`,
  `ceil_`,
  `ceil.out`,
  `diagmma_` (_generated_),
  `equal`,
  `floor_` (_generated_),
  `fmin` (_generated_),
  `fmin.out` (_generated_)
  `hardswish_` (_generated_),
  `hypot` (_generated_),
  `i0` (_generated_),
  `i0_` (_generated_),
  `i0.out` (_generated_),
  `log1p_` (_generated_),
  `logaddexp` (_generated_),
  `logaddexp.out` (_generated_),
  `logical_and_`,
  `logical_or_`
  `logit` (_generated_),
  `logit_` (_generated_),
  `logit.out` (_generated_),
  `sgn_` (_generated_),
  `sinh_` (_generated_),
  `special_i1` (_generated_),
  `special_i1.out` (_generated_)
- Added BLAS operator:
  `bmm.out`,
  `cutlass_scaled_mm_sm_90`,
  `tril` (_generated_)
- Added MoE operators:
  `dispatch_fused_moe_kernel`,
  `grouped_topk`,
  `inplace_fused_experts`,
  `outplace_fused_experts`
- Added distribution operator:
  `normal_`,
- Added neural network operators:
  `_upsample_nearest_exact1d`,
  `apply_repetition_penalties` (_generated_),
  `chunk_gated_delta_rule_fwd`,
  `dswiglu`,
  `embedding_dense_backward`,
  `fused_recurrent_gated_delta_rule_fwd`,
  `hardsigmoid` (_generated_),
  `hardsigmoid.out` (_generated_),
  `nll_loss_nd_backward`,
  `nll_loss_nd_forward`,
  `one_hot`,
  `pixel_unshuffle` (_generated_),
  `pixel_unshuffle.out` (_generated_),
  `prelu` (_generated_),
  `reflection_pad1d` (_generated_),
  `reflection_pad1d.out` (_generated_),
  `reflection_pad2d` (_generated_),
  `reflection_pad2d.out` (_generated_),
  `relu6` (_generated_),
  `swiglu`,
  `triu_`,
  `unfold_backward`,
  `upsample_bicubic2d`,
  `upsample_linear1d`,
  `upsample_nearest1d`,
  `upsample_nearest3d`
- Added tensor operators:
  `_functional_sym_constrain_range_for_size` (_generated_),
  `alias_copy` (_generated_),
  `alias_copy.out` (_generated_),
  `fill.Scalar_out`,
  `fill.Tensor_out`,
  `lift_fresh_copy` (_generated_),
  `replication_pad1d` (_generated_),
  `replication_pad1d.out` (_generated_),
  `replication_pad3d`,
  `rrelu_with_noise_backward` (_generated_),
  `selu` (_generated_),
  `selu_` (_generated_),
  `slice_backward` (_generated_),
  `softshrink` (_generated_),
  `softshrink.out` (_generated_),
  `t_copy` (_generated_),
  `t_copy.out` (_generated_),
  `unfold_backward`,
  `zero` (_generated_),
  `zero_`,
  `zero.out` (_generated_)
- Removed reduction operator `moe_sum`.
- Added reduction operators:
  `bincount`
- Added DSA operators:
  `spare_mla_fwd`

## v4.2

**Release date**: 2026-01-04

- This release contains 216 operators, see [Operator List](/FlagGems/reference/operators/)
- Added neural network operators:
  `avg_pool2d`,
  `avg_pool2d_backward`,
  `dgeglu` (*alpha*),
  `dreglu` (*alpha*),
  `geglu` (*alpha*),
  `reglu` (*alpha*),
- Added BLAS operators:
  `baddbmm`,
- Added convolution operators:
  `conv1d`,
  `conv2d`,
  `conv3d`,
- Added tensor operator:
  `copy_`,
  `masked_scatter`,
  `masked_scatter_`,
  `per_token_group_quant_fp8` (*alpha*),
  `scatter_add_`,
  `to_copy`.
- Added reduction operator:
  `moe_sum`,
  `scaled_softmax_backward`,
  `scaled_softmax_forward`,
- Added math operator:
  `exp_out`,
  `tan`,
  `tan_`,
  `true_divide_out`,
- Promoted tensor operator `continuous` to beta stage.
- Promoted reduction operator `index` to stable stage.
- The previous `upsample` operator is now split into
  `upsample_nearest2d` and `upsample_bicubic2d_aa`

## v4.1

**Release date**: 2025-11-01

- Added two fused RWKV operators:
  `rwkv_ka_fusion`,
  `rwkv_mm_sparsity`
- Adopted by the RWKV project in [BlinkDL/Albatross:faster_251101](https://github.com/BlinkDL/Albatross/tree/main/faster_251101)

## v4.0

**Release date**: 2025-10-31

- Supports 202 operators in total.
- Added linear algebra operators:
  `addcdiv`,
  `addcmul`,
  `addmv`,
  `addmv_out`,
  `addr`
- Added BLAS operators:
  `addmm_out`
- Added math operators:
  `atan`,
  `atan_`,
  `bitwise_left_shift`,
  `bitwise_right_shift`,
  `clamp_min`,
  `clamp_min_`,
  `exp2`,
  `exp2_`,
  `sqrt_`
- Added neural network operators:
  `celu`,
  `celu_`,
  `elu_`,
  `elu_backward`,
  `get_scheduler_metadata`,
  `glu_backward`,
  `moe_align_block_size`,
  `softplus`
- Added reduction operators:
  `index` (*beta*),
  `std`,
  `trace`
- Added tensor operators:
  `index_add_`,
  `logspace`,
  `max_pool2d_backward`,
  `max_pool2d_with_indices`,
  `topk_softmax`

- Triton JIT C++ runtime now ships precompiled kernels for:
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
  `reshape_and_cache`,
  `reshape_and_cache_flash`,
  `rms_norm`,
  `rms_norm_backward`,
  `rotary_embedding`,
  `softmax`,
  `sum`,
  `topk`,
  `zeros`

## v3.0

**Release date**: 2025-07-14

- Support 184 operators in total, including custom operators used in large model inference
- New hardware platforms supported:
  Ascend, AIPU, etc.
- Added compatibility with the vLLM framework, with the inference verification of DeepSeek model passed
- Added BLAS operator:
  `dot`,
  `mm_out`
- Added tensor operators:
  `index_put_`,
  `scatter_`,
  `sort_stable`
- Added linear algebra operators:
  `lerp`,
  `sum_dim_out`,
  `sum_out`
- Added math operators:
  `angle`,
  `nan_to_num`,
  `polar`,
  `tanh_backward`
- Added reduction operators:
  `cummax`,
  `cumsum_out`,
  `eye`,
  `index` (*alpha*),
  `layer_norm_backward`,
  `softmax_backward`
- Added neural network operators:
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
- Removed neural network operator:
  `cross_entropy_loss`,
  `instance_norm`
- Removed linear algebra operators:
  `outer` (*fused*),

## v2.2

**Release date**: 2025-04-17

- Added BLAS operators:
  `vdot`
- Added tensor operators:
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
- Added linear algebra operators:
  `diagonal_backward`
- Added convolution operators:
  `_conv_depthwise2d`,
- Added neural network operators:
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
- Added math operators:
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
  `tanh_`,
- Added reduction operators:
  `argmin`,
  `count_nonzero`,
  `cummin`,
  `gather`,
  `gather_backward`,
- Added science operators:
  `erf_`,
- Added distribution operators:
  `randperm`,

## v2.1

**Release date**: 2024-09-05

- Added Tensor operators:
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
- support neural network operator:
  `embedding`
- support basic math operators:
  `allclose`,
  `floor_divide`,
  `isclose`,
  `isfinite`,
  `maximum`,
  `minimum`,
  `true_divide`,
  `trunc_divide`
- support distribution operators:
  `exponential_`,
  `multinomial`,
  `nonzero`,
  `normal`,
  `topk`,
  `rand`,
  `rand_like`,
  `randn`,
  `randn_like`,
  `topk`,
  `uniform_`
- Added science operators:
  `erf`,
  `resolve_conj`,
  `resolve_neg`

## v2.0

**Release date**: 2024-05-31

- Added BLAS operators:
  `mv`,
  `outer`
- Added math operators:
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
- Added reduction operators:
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
- Added neural network operators:
  `apply_rotary_position_embedding`,
  `fused_add_rms_norm`,
  `gelu_and_mul`,
  `outer` (*alpha*),
  `silu_and_mul`,
  `silu_and_mul_out`,
  `skip_layer_norm`

## v1.0

**Release date**: 2024-05-10

- Added BLAS operators:
  `addmm`,
  `bmm`,
  `mm`
- Added math operators:
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
- Added reduction operators:
  `cumsum`,
  `layer_norm`,
  `mean`,
  `softmax`
