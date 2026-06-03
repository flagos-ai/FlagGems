# vLLM Patch Registry

## 概述

FlagGems 为 vLLM 提供了统一的 patch 注册系统，用于替换 vLLM 的算子实现。

## 架构

所有配置集中在 `patch_util.py` 的 `_VLLM_OPS_REGISTRY` 变量中：

```python
_VLLM_OPS_REGISTRY = {
    "lib_name": {
        "op_name": {
            "signature": "PyTorch schema",
            "impl": "custom_function_name",
        },
    },
}
```

- `signature`: PyTorch custom op schema 签名
- `impl`: `patch_vllm_all.py` 中的实现函数名称

## 添加新的 Patch

### 步骤 1: 实现 custom 函数

在 `patch_vllm_all.py` 中添加:

```python
def custom_my_new_op(arg1: torch.Tensor, arg2: int) -> torch.Tensor:
    return flag_gems.my_new_op(arg1, arg2)
```

### 步骤 2: 注册到 _VLLM_OPS_REGISTRY

在 `patch_util.py` 的 `_VLLM_OPS_REGISTRY` 中添加:

```python
_VLLM_OPS_REGISTRY = {
    "_C": {
        # ... existing ops
        "my_new_op": {
            "signature": "(Tensor arg1, int arg2) -> Tensor",
            "impl": "custom_my_new_op",
        },
    },
}
```

### 步骤 3: 测试

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

## 辅助函数

```python
from flag_gems.patches.patch_util import (
    get_vllm_ops_registry,  # 获取完整注册表
    get_lib_ops,            # 获取指定库的 op 列表
    get_op_signature,       # 获取 op 的 signature
    get_op_impl_name,       # 获取 op 的实现函数名
)
```

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
