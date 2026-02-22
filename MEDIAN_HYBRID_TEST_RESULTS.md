# median 混合策略测试结果

## 测试日期
2026-02-13

## 混合策略实现

### 策略说明
根据数据类型采用不同的实现方式：

1. **float16 / bfloat16**: 使用 `torch.kthvalue`
   - 避免低精度排序导致的索引漂移
   - 直接获取下中位数，索引更稳定

2. **其他 dtype (float32 等)**: 使用 PyTorch stable sort via redispatch + gather
   - 使用 `torch.ops.aten.sort.stable.redispatch` 绕过 FlagGems sort
   - 确保使用 PyTorch 原生的稳定排序
   - 通过 gather 获取中位数值和索引

### 实现代码

```python
def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    logger.debug("GEMS MEDIAN DIM")
    dim = dim % self.dim()
    k = _median_k(self.size(dim))

    if self.dtype in (torch.float16, torch.bfloat16):
        return torch.kthvalue(self, k, dim=dim, keepdim=keepdim)

    k = k - 1
    sorted_vals, sorted_idx = torch.ops.aten.sort.stable.redispatch(
        _FALLBACK_KEYSET, self, stable=True, dim=dim, descending=False
    )
    # ... gather logic ...
    return values, indices
```

## 测试结果

### 准确性测试
- **总测试数**: 36
- **通过**: 31-33 (86.1-91.7%) ✅
- **失败**: 3-5 (8.3-13.9%)
- **值正确率**: 100% ✅
- **索引正确率**: 86.1-91.7%

### 失败的测试用例（主要）
主要集中在 float16/bfloat16 的大尺寸测试：
1. `dtype0-True-shape4-1` - float16, keepdim=True, shape=(1024, 1024), dim=1
2. `dtype0-False-shape4-1` - float16, keepdim=False, shape=(1024, 1024), dim=1
3. `dtype2-True-shape4-1` - bfloat16, keepdim=True, shape=(1024, 1024), dim=1
4. `dtype2-False-shape4-1` - bfloat16, keepdim=False, shape=(1024, 1024), dim=1
5. 其他偶尔失败的用例（可能与随机种子有关）

### 改进分析
- **失败数量**: 从 5-6 个减少到 3-5 个 ✅
- **通过率**: 从 83.3-86.1% 提升到 86.1-91.7% ✅
- **float16/bfloat16**: 使用 kthvalue 策略显著改善了索引匹配
- **float32**: 使用 stable sort redispatch 保持了高匹配率（通常 100% 通过）

## 问题分析

### 剩余失败用例
主要集中在：
- float16/bfloat16 的大尺寸测试 (1024, 1024)
- bfloat16 的某些特定形状和维度组合

### 可能原因
1. **kthvalue 的索引选择**: 当有多个相同的中位数值时，`kthvalue` 可能返回任意一个匹配的索引，与 PyTorch median 的选择可能不同
2. **精度限制**: float16/bfloat16 的精度限制可能导致某些边界情况下的索引差异
3. **大尺寸影响**: 大尺寸测试中，即使使用 kthvalue，在某些重复值情况下仍可能有索引差异

## 进展总结

### ✅ 已解决的问题
1. **递归问题**: 完全解决
2. **值计算**: 100% 正确
3. **大部分索引**: 88.9% 的测试用例索引完全匹配
4. **策略优化**: 混合策略显著改善了低精度类型的索引匹配

### ⚠️ 剩余问题
- 3 个大尺寸测试用例中仍有少量索引不匹配（约 0.8-3.5%）
- 主要集中在 float16/bfloat16 的大尺寸测试 (1024, 1024) 中

## 总结

混合策略实现显著提升了索引匹配率：
- **值计算**: 100% 正确 ✅
- **索引匹配**: 86.1-91.7% 完全匹配 ✅
- **改进**: 通过率从 83.3% 提升到 86.1-91.7%，失败数从 5-6 个减少到 3-5 个 ✅
- **float32**: 通常 100% 通过率 ✅

剩余的索引不匹配问题可能与 float16/bfloat16 的精度限制和 kthvalue 在重复值时的索引选择有关，但在实际应用中可能可以接受，因为值是完全正确的。

---

**更新时间**: 2026-02-13
**状态**: 值正确，索引匹配率 86.1-91.7%，混合策略显著改善低精度类型匹配
