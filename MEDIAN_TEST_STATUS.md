# median 测试状态报告

## 问题说明

`median` 操作在测试时出现递归错误。问题在于：

1. `torch.median` 调用我们的 `median_dim` 实现
2. `median_dim` 调用 `torch.ops.aten.median.dim.redispatch` 
3. `redispatch` 仍然会调用我们的实现，导致递归

## 尝试的修复方法

1. ✅ 修复了 `.default` 属性错误（`median.dim` 没有 `.default` 属性）
2. ❌ 使用 `CompositeExplicitAutograd` keyset - 仍然递归
3. ❌ 使用 `_DisableTorchDispatch` - 仍然递归
4. ❌ 使用 CPU/CUDA keyset - 仍然递归

## 当前实现

```python
def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    logger.debug("GEMS MEDIAN DIM")
    # Use _DisableTorchDispatch to avoid recursion when calling redispatch
    with torch._C._DisableTorchDispatch():
        return torch.ops.aten.median.dim.redispatch(
            _FALLBACK_KEYSET, self, dim, keepdim
        )
```

## 错误信息

```
RecursionError: maximum recursion depth exceeded in __instancecheck__
!!! Recursion detected (same locals & position)
```

## 建议的解决方案

由于 `median` 可能只是一个 pass-through 实现（没有实际的 Triton 优化），可以考虑：

1. **完全禁用 median 实现**：如果不需要优化，可以不在 FlagGems 中注册
2. **使用不同的 dispatch 机制**：可能需要使用 `_DispatchKeyGuard` 或其他机制
3. **直接调用底层实现**：绕过 PyTorch 的 dispatch 系统

## 测试状态

- **准确性测试**: ❌ 36个测试全部失败（递归错误）
- **性能测试**: ⚠️ 未测试（准确性测试未通过）

---

**生成时间**: 2026-02-13
**状态**: 需要进一步调查递归问题
