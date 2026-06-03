# vLLM Patch Registry

## 概述

FlagGems 为 vLLM 提供了统一的 patch 注册系统，用于替换 vLLM 的算子实现。

## 架构

- `patch_util.py`: 核心注册表和工具函数
  - `_LIB_OPS`: 定义需要 patch 的库和算子列表
  - `_OP_SIGNATURES`: 算子的 PyTorch schema 签名
  - `_LIB_PATCHES`: 统一的 patch 配置（运行时生成）
  
- `patch_vllm_all.py`: 实现函数和应用逻辑
  - `custom_*`: 各个算子的 FlagGems 实现
  - `apply_gems_patches_to_vllm()`: 应用所有 patches

## 添加新的 Patch

### 步骤 1: 实现 custom 函数

在 `patch_vllm_all.py` 中添加:

```python
def custom_my_new_op(arg1: torch.Tensor, arg2: int) -> torch.Tensor:
    return flag_gems.my_new_op(arg1, arg2)
```

### 步骤 2: 添加 signature

在 `patch_util.py` 的 `_OP_SIGNATURES` 中添加:

```python
"_C": {
    # ... existing ops
    "my_new_op": "(Tensor arg1, int arg2) -> Tensor",
}
```

### 步骤 3: 更新 _LIB_OPS

在 `patch_util.py` 中:

```python
_LIB_OPS = {
    "_C": [
        # ... existing ops
        "my_new_op",
    ],
}
```

### 步骤 4: 注册到 patch 列表

在 `patch_vllm_all.py` 的 `_register_all_lib_patches()` 函数中添加:

```python
def _register_all_lib_patches():
    patches_to_register = [
        # ... existing patches
        ("_C", "my_new_op", "custom_my_new_op"),
    ]
    return patches_to_register
```

### 步骤 5: 测试

验证没有语法错误:

```bash
python3 -m py_compile src/flag_gems/patches/patch_util.py
python3 -m py_compile src/flag_gems/patches/patch_vllm_all.py
```

## 支持的库

- `_C`: vLLM 核心 C++ 扩展
- `_moe_C`: MoE (Mixture of Experts) 算子
- `_vllm_fa3_C`: FlashAttention v3 集成
- `_C_cache_ops`: KV cache 相关算子

## 注意事项

1. 所有 custom 函数必须匹配原始 vLLM 算子的签名
2. Signature 必须遵循 PyTorch custom op schema 语法
3. 使用 `register_lib_patch()` 可以在运行时动态添加 patches

## 当前已注册的 Patches

### _C (核心库)
- `rms_norm` - RMS normalization
- `silu_and_mul` - SiLU activation with multiplication
- `silu_and_mul_with_clamp` - SiLU activation with clamp
- `hc_head_fused_kernel` - HC head fused kernel
- `cutlass_scaled_mm` - Scaled matrix multiplication
- `per_token_group_fp8_quant` - Per-token group FP8 quantization
- `apply_repetition_penalties_` - Repetition penalties application
- `top_k_per_row_prefill` - Top-K per row prefill

### _moe_C (MoE 库)
- `topk_softmax` - Top-K softmax
- `moe_align_block_size` - MoE block size alignment
- `grouped_topk` - Grouped top-K selection
- `moe_sum` - MoE sum operation

### _vllm_fa3_C (FlashAttention v3)
- `get_scheduler_metadata` - Get scheduler metadata

### _C_cache_ops (KV Cache)
- `concat_and_cache_mla` - Concatenate and cache for MLA
