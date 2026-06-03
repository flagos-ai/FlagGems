# Merge vLLM Patch Configurations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `patch_vllm_all.py` 中的 `lib_patches` 与 `patch_util.py` 中的 `_LIB_OPS` 和 `_OP_SIGNATURES` 合并为统一的配置。

**Architecture:** 在 `patch_util.py` 中创建统一的 patch 配置注册表，包含 lib_name、op_name、implementation 和 signature。修改 `patch_vllm_all.py` 使用这个统一配置，移除重复代码。

**Tech Stack:** Python 3, PyTorch, vLLM

---

## File Structure

**Files to modify:**
- `src/flag_gems/patches/patch_util.py` - 添加统一的 patch 配置注册表
- `src/flag_gems/patches/patch_vllm_all.py` - 使用新的统一配置替代 `lib_patches`

## Task 1: 扩展 patch_util.py 的配置结构

**Files:**
- Modify: `src/flag_gems/patches/patch_util.py:42-92`

- [ ] **Step 1: 添加 _LIB_PATCHES 配置字典的测试**

创建测试文件验证配置结构：

```python
# tests/test_patch_util_config.py
def test_lib_patches_structure():
    """测试 _LIB_PATCHES 配置结构正确"""
    from flag_gems.patches.patch_util import _LIB_PATCHES
    
    # 验证配置存在
    assert "_C" in _LIB_PATCHES
    assert "_moe_C" in _LIB_PATCHES
    
    # 验证每个配置项包含必需字段
    for lib_name, ops in _LIB_PATCHES.items():
        for op_name, config in ops.items():
            assert "signature" in config
            assert "impl" in config
            assert callable(config["impl"])
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /data/wangzhen/FlagGems
pytest tests/test_patch_util_config.py::test_lib_patches_structure -v
```

预期: FAIL - `_LIB_PATCHES` 未定义

- [ ] **Step 3: 在 patch_util.py 中添加 _LIB_PATCHES 配置**

在 `_OP_SIGNATURES` 定义后添加：

```python
# src/flag_gems/patches/patch_util.py (after line 92)

_LIB_PATCHES = {}

def register_lib_patch(lib_name: str, op_name: str, impl_func: callable, signature: str = None):
    """注册 vLLM library patch
    
    Args:
        lib_name: Library name (_C, _moe_C, etc.)
        op_name: Operation name
        impl_func: Implementation function
        signature: Optional signature override (uses _OP_SIGNATURES if None)
    """
    if lib_name not in _LIB_PATCHES:
        _LIB_PATCHES[lib_name] = {}
    
    # 获取 signature
    if signature is None:
        signature = _OP_SIGNATURES.get(lib_name, {}).get(op_name)
    
    _LIB_PATCHES[lib_name][op_name] = {
        "impl": impl_func,
        "signature": signature
    }

def get_lib_patches():
    """返回所有已注册的 library patches"""
    return _LIB_PATCHES
```

- [ ] **Step 4: 运行测试验证实现**

```bash
pytest tests/test_patch_util_config.py::test_lib_patches_structure -v
```

预期: PASS

- [ ] **Step 5: 提交更改**

```bash
git add src/flag_gems/patches/patch_util.py tests/test_patch_util_config.py
git commit -m "feat: add unified lib patch registry in patch_util"
```

## Task 2: 在 patch_vllm_all.py 中注册所有 patches

**Files:**
- Modify: `src/flag_gems/patches/patch_vllm_all.py:632-649`
- Modify: `src/flag_gems/patches/patch_util.py` (import statement)

- [ ] **Step 1: 添加集成测试**

```python
# tests/test_vllm_patch_integration.py
def test_all_patches_registered():
    """验证所有 vLLM patches 都已注册"""
    from flag_gems.patches.patch_util import get_lib_patches
    
    expected_patches = {
        "_C": ["rms_norm", "silu_and_mul", "cutlass_scaled_mm", 
               "per_token_group_fp8_quant", "apply_repetition_penalties_",
               "top_k_per_row_prefill"],
        "_moe_C": ["moe_align_block_size", "topk_softmax", "moe_sum", "grouped_topk"],
        "_vllm_fa3_C": ["get_scheduler_metadata"],
        "_C_cache_ops": ["concat_and_cache_mla"],
    }
    
    patches = get_lib_patches()
    for lib_name, op_names in expected_patches.items():
        assert lib_name in patches
        for op_name in op_names:
            assert op_name in patches[lib_name]
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_vllm_patch_integration.py::test_all_patches_registered -v
```

预期: FAIL - patches 未注册

- [ ] **Step 3: 更新 patch_util.py 导出**

```python
# src/flag_gems/patches/patch_util.py (update import section)
__all__ = [
    "init_vllm_libraries",
    "patch_module_method",
    "patch_vllm_lib",
    "register_lib_patch",
    "get_lib_patches",
]
```

- [ ] **Step 4: 在 patch_vllm_all.py 开头注册所有 patches**

在文件开头（imports 之后）添加：

```python
# src/flag_gems/patches/patch_vllm_all.py (after imports, before custom functions)
from flag_gems.patches.patch_util import register_lib_patch

# 注册所有 library patches - 在模块加载时执行
def _register_all_lib_patches():
    """注册所有 vLLM library patches 到统一注册表"""
    # 注意: custom_* 函数在下面定义，这里只是前向引用
    patches_to_register = [
        ("_C", "rms_norm", "custom_rms_norm_out"),
        ("_C", "silu_and_mul", "custom_silu_and_mul"),
        ("_C", "silu_and_mul_with_clamp", "custom_silu_and_mul_with_clamp"),
        ("_C", "hc_head_fused_kernel", "custom_hc_head_fused_kernel"),
        ("_C", "cutlass_scaled_mm", "custom_cutlass_scaled_mm"),
        ("_C", "per_token_group_fp8_quant", "custom_per_token_group_fp8_quant"),
        ("_C", "apply_repetition_penalties_", "custom_apply_repetition_penalties"),
        ("_C", "top_k_per_row_prefill", "custom_top_k_per_row_prefill"),
        ("_moe_C", "moe_align_block_size", "custom_moe_align_block_size"),
        ("_moe_C", "topk_softmax", "custom_topk_softmax"),
        ("_moe_C", "moe_sum", "custom_moe_sum"),
        ("_moe_C", "grouped_topk", "custom_moe_grouped_topk"),
        ("_vllm_fa3_C", "get_scheduler_metadata", "custom_get_scheduler_metadata"),
        ("_C_cache_ops", "concat_and_cache_mla", "custom_concat_and_cache_mla"),
    ]
    return patches_to_register

_PATCHES_TO_REGISTER = _register_all_lib_patches()
```

- [ ] **Step 5: 运行测试验证**

```bash
pytest tests/test_vllm_patch_integration.py::test_all_patches_registered -v
```

预期: 仍然 FAIL - 需要在 apply_gems_patches_to_vllm 中实际注册

- [ ] **Step 6: 提交更改**

```bash
git add src/flag_gems/patches/patch_util.py src/flag_gems/patches/patch_vllm_all.py tests/test_vllm_patch_integration.py
git commit -m "feat: define lib patches registry structure"
```

## Task 3: 重构 apply_gems_patches_to_vllm 使用统一配置

**Files:**
- Modify: `src/flag_gems/patches/patch_vllm_all.py:632-649`

- [ ] **Step 1: 编写测试验证 patch 应用**

```python
# tests/test_vllm_patch_integration.py (add new test)
def test_patches_applied_correctly(monkeypatch):
    """验证 patches 正确应用到 vLLM"""
    from flag_gems.patches.patch_vllm_all import apply_gems_patches_to_vllm
    from flag_gems.patches.patch_util import get_lib_patches
    
    # Mock patch_vllm_lib 来验证调用
    calls = []
    def mock_patch_vllm_lib(lib_name, fn_name, fn, key, verbose=True):
        calls.append((lib_name, fn_name, fn.__name__))
    
    monkeypatch.setattr("flag_gems.patches.patch_vllm_all.patch_vllm_lib", 
                       mock_patch_vllm_lib)
    
    # 应用 patches
    apply_gems_patches_to_vllm(verbose=False)
    
    # 验证所有注册的 patches 都被应用
    patches = get_lib_patches()
    for lib_name, ops in patches.items():
        for op_name in ops:
            assert any(c[0] == lib_name and c[1] == op_name for c in calls), \
                f"Patch {lib_name}::{op_name} was not applied"
```

- [ ] **Step 2: 运行测试确认当前行为**

```bash
pytest tests/test_vllm_patch_integration.py::test_patches_applied_correctly -v
```

预期: FAIL - patches 未从注册表应用

- [ ] **Step 3: 重构 apply_gems_patches_to_vllm**

替换 lines 632-649:

```python
# src/flag_gems/patches/patch_vllm_all.py
def apply_gems_patches_to_vllm(verbose=True):
    import vllm  # noqa: F401
    import vllm._custom_ops as ops  # noqa: F401

    try:
        from vllm.attention.ops import vit_attn_wrappers as vitw
    except (ModuleNotFoundError, ImportError):
        vitw = None
    from vllm.attention.ops.paged_attn import PagedAttention
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.backends.mla.flashattn_mla import FlashAttnMLAImpl
    from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl

    dispatch_key = flag_gems.runtime.device.dispatch_key
    init_vllm_libraries()
    
    # 首先注册所有 patches 到统一注册表
    _register_patches_to_registry()

    # Patch module methods
    module_patches = [
        (RMSNorm, "forward_cuda", custom_gems_rms_forward_cuda),
        (RotaryEmbedding, "forward_cuda", custom_gems_rope_forward_cuda),
        (PagedAttention, "write_to_paged_cache", custom_gems_write_to_paged_cache),
        (SiluAndMul, "forward_cuda", custom_gems_silu_and_mul),
        (TritonMLAImpl, "_forward_decode", custom_gems_flash_mla_forward),
        (FlashAttentionImpl, "forward", custom_gems_flash_attention_impl_forward),
        (FlashAttnMLAImpl, "_forward_decode", custom_gems_flashattn_mla_forward_decode),
    ]
    for cls, method_name, new_method in module_patches:
        patch_module_method(cls, method_name, new_method, verbose)

    # Patch library ops from unified registry
    patches = get_lib_patches()
    for lib_name, ops_dict in patches.items():
        for op_name, config in ops_dict.items():
            patch_vllm_lib(lib_name, op_name, config["impl"], dispatch_key, verbose)

    if vitw is not None:
        patch_vllm_vit_to_attn(vitw)


def _register_patches_to_registry():
    """将所有 custom 函数注册到统一的 patch 注册表"""
    for lib_name, op_name, func_name in _PATCHES_TO_REGISTER:
        impl_func = globals()[func_name]
        register_lib_patch(lib_name, op_name, impl_func)
```

- [ ] **Step 4: 运行测试验证重构**

```bash
pytest tests/test_vllm_patch_integration.py::test_patches_applied_correctly -v
```

预期: PASS

- [ ] **Step 5: 提交更改**

```bash
git add src/flag_gems/patches/patch_vllm_all.py tests/test_vllm_patch_integration.py
git commit -m "refactor: use unified patch registry in apply_gems_patches_to_vllm"
```

## Task 4: 添加缺失的 op signatures

**Files:**
- Modify: `src/flag_gems/patches/patch_util.py:54-92`

- [ ] **Step 1: 添加测试验证所有 ops 有 signature**

```python
# tests/test_patch_util_config.py (add test)
def test_all_registered_ops_have_signatures():
    """验证所有注册的 ops 都有对应的 signature"""
    from flag_gems.patches.patch_util import get_lib_patches, _OP_SIGNATURES
    
    patches = get_lib_patches()
    missing = []
    
    for lib_name, ops_dict in patches.items():
        for op_name, config in ops_dict.items():
            if config["signature"] is None:
                # 检查是否在 _OP_SIGNATURES 中定义
                if lib_name not in _OP_SIGNATURES or op_name not in _OP_SIGNATURES[lib_name]:
                    missing.append(f"{lib_name}::{op_name}")
    
    assert len(missing) == 0, f"Missing signatures for: {missing}"
```

- [ ] **Step 2: 运行测试识别缺失的 signatures**

```bash
pytest tests/test_patch_util_config.py::test_all_registered_ops_have_signatures -v
```

预期: FAIL - 列出缺失 signatures 的 ops

- [ ] **Step 3: 添加缺失的 signatures 到 _OP_SIGNATURES**

```python
# src/flag_gems/patches/patch_util.py (update _OP_SIGNATURES dict)
_OP_SIGNATURES = {
    "_moe_C": {
        "topk_softmax": "(Tensor(a!) topk_weights, Tensor(b!) topk_indices, "
        "Tensor(c!) token_expert_indices, Tensor gating_output) -> ()",
        "moe_align_block_size": "(Tensor topk_ids, int num_experts, "
        "int block_size, Tensor(a!) sorted_token_ids, Tensor(b!) experts_ids, "
        "Tensor(c!) num_tokens_post_pad) -> ()",
        "grouped_topk": "(Tensor gating_output, int n_group, int topk_group, "
        "int topk, bool renormalize, float routed_scaling_factor, Tensor? bias, "
        "int scoring_func=0) -> (Tensor, Tensor, Tensor)",
        "moe_sum": "(Tensor input, Tensor(a!) output) -> ()",
    },
    "_C": {
        "rms_norm": "(Tensor(a!) result, Tensor input, Tensor weight, float epsilon) -> ()",
        "silu_and_mul": "(Tensor(a!) out, Tensor input) -> ()",
        "silu_and_mul_with_clamp": "(Tensor(a!) out, Tensor input, float limit) -> ()",
        "hc_head_fused_kernel": "(Tensor hs_flat, Tensor fn, Tensor hc_scale, "
        "Tensor hc_base, Tensor(a!) out, int hidden_size, float rms_eps, "
        "float hc_eps, int hc_mult) -> Tensor",
        "cutlass_scaled_mm": "(Tensor(a!) out, Tensor input, Tensor weight, "
        "Tensor scale_a, Tensor scale_b, Tensor? bias=None) -> ()",
        "per_token_group_fp8_quant": "(Tensor input, Tensor(a!) output_q, "
        "Tensor(b!) output_s, int group_size, float eps, float fp8_min, "
        "float fp8_max, bool scale_ue8m0=False) -> ()",
        "apply_repetition_penalties_": "(Tensor(a!) logits, Tensor prompt_mask, "
        "Tensor output_mask, Tensor repetition_penalties) -> Tensor",
        "top_k_per_row_prefill": "(Tensor logits, Tensor row_starts, Tensor row_ends, "
        "Tensor(a!) indices, int num_rows, int stride0, int stride1, int top_k) -> ()",
    },
    "_vllm_fa3_C": {
        "get_scheduler_metadata": "(int batch_size, int max_seqlen_q, int max_seqlen_k, "
        "int num_heads, int num_heads_k, int headdim, int headdim_v, "
        "ScalarType qkv_dtype, Tensor seqused_k, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, "
        "Tensor? cu_seqlens_k_new=None, Tensor? seqused_q=None, "
        "Tensor? leftpad_k=None, int? page_size=None, "
        "int max_seqlen_k_new=0, bool is_causal=False, "
        "int window_size_left=-1, int window_size_right=-1, "
        "bool has_softcap=False, int num_splits=0, "
        "bool? pack_gqa=None, int sm_margin=0) -> Tensor",
    },
    "_C_cache_ops": {
        "concat_and_cache_mla": "(Tensor kv_c, Tensor k_pe, Tensor(a!) kv_cache, "
        "Tensor slot_mapping, str kv_cache_dtype, Tensor scale) -> ()",
    },
}
```

- [ ] **Step 4: 运行测试验证所有 signatures 已添加**

```bash
pytest tests/test_patch_util_config.py::test_all_registered_ops_have_signatures -v
```

预期: PASS

- [ ] **Step 5: 提交更改**

```bash
git add src/flag_gems/patches/patch_util.py tests/test_patch_util_config.py
git commit -m "feat: add missing op signatures to _OP_SIGNATURES"
```

## Task 5: 清理旧代码并验证完整性

**Files:**
- Modify: `src/flag_gems/patches/patch_vllm_all.py:632-649`
- Modify: `src/flag_gems/patches/patch_util.py:42-52`

- [ ] **Step 1: 添加端到端测试**

```python
# tests/test_e2e_vllm_patches.py
import pytest

def test_patch_registry_matches_lib_ops():
    """验证 patch registry 覆盖所有 _LIB_OPS 中定义的 ops"""
    from flag_gems.patches.patch_util import _LIB_OPS, get_lib_patches
    from flag_gems.patches.patch_vllm_all import _register_patches_to_registry
    
    # 触发注册
    _register_patches_to_registry()
    patches = get_lib_patches()
    
    for lib_name, op_list in _LIB_OPS.items():
        assert lib_name in patches, f"Library {lib_name} not in patches"
        for op_name in op_list:
            assert op_name in patches[lib_name], \
                f"Op {lib_name}::{op_name} not registered"

def test_no_duplicate_patch_definitions():
    """验证没有重复的 patch 定义"""
    from flag_gems.patches.patch_vllm_all import _PATCHES_TO_REGISTER
    
    seen = set()
    duplicates = []
    
    for lib_name, op_name, _ in _PATCHES_TO_REGISTER:
        key = f"{lib_name}::{op_name}"
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    
    assert len(duplicates) == 0, f"Duplicate patches: {duplicates}"
```

- [ ] **Step 2: 运行端到端测试**

```bash
pytest tests/test_e2e_vllm_patches.py -v
```

预期: PASS

- [ ] **Step 3: 更新 _LIB_OPS 以匹配实际 patches**

根据 lib_patches 更新 _LIB_OPS:

```python
# src/flag_gems/patches/patch_util.py
_LIB_OPS = {
    "_C": [
        "rms_norm",
        "silu_and_mul",
        "silu_and_mul_with_clamp",
        "hc_head_fused_kernel",
        "cutlass_scaled_mm",
        "per_token_group_fp8_quant",
        "apply_repetition_penalties_",
        "top_k_per_row_prefill",
    ],
    "_moe_C": [
        "topk_softmax",
        "moe_align_block_size",
        "grouped_topk",
        "moe_sum",
    ],
    "_vllm_fa3_C": [
        "get_scheduler_metadata",
    ],
    "_C_cache_ops": [
        "concat_and_cache_mla",
    ],
}
```

- [ ] **Step 4: 删除旧的 lib_patches 列表注释**

在 `apply_gems_patches_to_vllm` 中删除注释掉的旧代码（如果有）。

- [ ] **Step 5: 运行所有测试验证完整性**

```bash
pytest tests/test_patch_util_config.py tests/test_vllm_patch_integration.py tests/test_e2e_vllm_patches.py -v
```

预期: 所有测试 PASS

- [ ] **Step 6: 提交最终更改**

```bash
git add src/flag_gems/patches/patch_util.py src/flag_gems/patches/patch_vllm_all.py tests/test_e2e_vllm_patches.py
git commit -m "refactor: complete merge of lib_patches into unified registry"
```

## Task 6: 添加文档和使用示例

**Files:**
- Create: `docs/vllm_patch_registry.md`

- [ ] **Step 1: 创建文档文件**

```markdown
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

在 `patch_vllm_all.py` 的 `_PATCHES_TO_REGISTER` 中添加:

```python
_PATCHES_TO_REGISTER = _register_all_lib_patches()

# 在 _register_all_lib_patches() 函数中添加:
patches_to_register = [
    # ... existing patches
    ("_C", "my_new_op", "custom_my_new_op"),
]
```

### 步骤 5: 测试

```bash
pytest tests/test_e2e_vllm_patches.py -v
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
```

- [ ] **Step 2: 提交文档**

```bash
git add docs/vllm_patch_registry.md
git commit -m "docs: add vLLM patch registry documentation"
```

---

## 验收标准

- [ ] 所有测试通过
- [ ] `lib_patches` 列表已从 `patch_vllm_all.py` 移除
- [ ] `_LIB_OPS` 和 `_OP_SIGNATURES` 完全匹配实际 patches
- [ ] 统一注册表在 `patch_util.py` 中正常工作
- [ ] 文档完整且清晰
- [ ] 没有功能回归（vLLM patches 仍正常工作）

## 测试命令

```bash
# 运行所有相关测试
pytest tests/test_patch_util_config.py tests/test_vllm_patch_integration.py tests/test_e2e_vllm_patches.py -v

# 验证没有 import 错误
python -c "from flag_gems.patches.patch_vllm_all import apply_gems_patches_to_vllm"

# 检查代码质量
ruff check src/flag_gems/patches/
```
