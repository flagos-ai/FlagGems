## Change History

<!--TODO(Qiming): Amend the list below with dates-->

### v4.2 (upcoming)

- release targeting 216 operators, aligned with the updated [Operator List](./docs/operators.md)
- additions: `tan`, `tan_`, `baddbmm`, `avg_pool2d`, `clamp_min`, `clamp_min_`, `std`, `trace`, `max_pool2d`, `bitwise_left_shift`, `bitwise_right_shift`
- the previous `upsample` operator will be split into `upsample_nearest2d` and `upsample_bicubic2d_aa`

### v4.1

- dedicated RWKV-focused release with 204 supported operators
- includes fused kernels `rwkv_mm_sparsity` and `rwkv_ka_fusion` optimized for RWKV inference acceleration scenarios
- adopted by the RWKV project in [BlinkDL/Albatross:faster_251101](https://github.com/BlinkDL/Albatross/tree/main/faster_251101)

### v4.0

- support 202 operators in total
- newly added operators: `addcdiv`, `addcmul`, `addmv`, `addr`, `atan`, `atan_`, `celu`, `celu_`, `elu_`, `exp2`, `exp2_`, `get_scheduler_metadata`, `index_add_`, `logspace`, `moe_align_block_size`, `softplus`, `sqrt_`, `topk_softmax`
- Triton JIT C++ runtime now ships precompiled kernels for: `add`, `addmm`, `argmax`, `bmm`, `cat`, `contiguous`, `embedding`, `exponential_`, `fill`, `flash_attn_varlen_func`, `fused_add_rms_norm`, `max`, `mm`, `nonzero`, `reshape_and_cache_flash`, `rms_norm`, `rotary_embedding`, `softmax`, `sum`, `topk`, `zeros`

### v3.0

- support 184 operators in total, including custom operators used in large model inference
- support more hardware platforms, add Ascend, AIPU, etc.
- compatible with the vLLM framework, with the inference verification of DeepSeek model passed

### v2.1

- support Tensor operators: where, arange, repeat, masked_fill, tile, unique, index_select, masked_select, ones, ones_like, zeros, zeros_like, full, full_like, flip, pad
- support neural network operator: embedding
- support basic math operators: allclose, isclose, isfinite, floor_divide, trunc_divide, maximum, minimum
- support distribution operators: normal, uniform\_, exponential\_, multinomial, nonzero, topk, rand, randn, rand_like, randn_like
- support science operators: erf, resolve_conj, resolve_neg

### v2.0

- support BLAS operators: mv, outer
- support pointwise operators: bitwise_and, bitwise_not, bitwise_or, cos, clamp, eq, ge, gt, isinf, isnan, le, lt, ne, neg, or, sin, tanh, sigmoid
- support reduction operators: all, any, amax, argmax, max, min, prod, sum, var_mean, vector_norm, cross_entropy_loss, group_norm, log_softmax, rms_norm
- support fused operators: fused_add_rms_norm, skip_layer_norm, gelu_and_mul, silu_and_mul, apply_rotary_position_embedding

### v1.0

- support BLAS operators: addmm, bmm, mm
- support pointwise operators: abs, add, div, dropout, exp, gelu, mul, pow, reciprocal, relu, rsqrt, silu, sub, triu
- support reduction operators: cumsum, layernorm, mean, softmax
