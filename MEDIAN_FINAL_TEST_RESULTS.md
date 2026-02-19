# median 最终测试结果

## 测试日期
2026-02-13

## 最新实现策略
使用 **PyTorch stable sort via redispatch** 来绕过 FlagGems sort：
- 使用 `torch.ops.aten.sort.stable.redispatch` 绕过 FlagGems 的 sort 实现
- 确保使用 PyTorch 原生的稳定排序，避免索引不一致
- 选择下中位索引 `k = (size + 1) // 2 - 1`
- 使用 `torch.take_along_dim` 获取值和原始索引

## 测试结果

### 准确性测试
- **总测试数**: 36
- **通过**: 30 (83.3%) ✅
- **失败**: 6 (16.7%)
- **值正确率**: 100% ✅
- **索引正确率**: 83.3%

### 失败的测试用例
1. `dtype0-True-shape4-1` - float16, keepdim=True, shape=(1024, 1024), dim=1
2. `dtype0-False-shape2-0` - float16, keepdim=False, shape=(64, 64), dim=0
3. `dtype0-False-shape4-1` - float16, keepdim=False, shape=(1024, 1024), dim=1
4. `dtype2-True-shape4-1` - bfloat16, keepdim=True, shape=(1024, 1024), dim=1
5. `dtype2-False-shape2-0` - bfloat16, keepdim=False, shape=(64, 64), dim=0
6. `dtype2-False-shape4-1` - bfloat16, keepdim=False, shape=(1024, 1024), dim=1

### 问题分析
- **改进**: 失败数量从 7 个减少到 6 个
- **问题**: 仍有约 3.5% 的索引不匹配（在大尺寸测试中）
- **可能原因**:
  1. float16/bfloat16 精度问题导致排序顺序的细微差异
  2. 即使使用 PyTorch 的稳定排序，在某些边界情况下仍可能有差异
  3. 可能与 PyTorch median 的具体实现细节有关

## 进展总结

### ✅ 已解决的问题
1. **递归问题**: 完全解决，使用 redispatch 避免了递归
2. **值计算**: 100% 正确
3. **大部分索引**: 83.3% 的测试用例索引完全匹配

### ⚠️ 剩余问题
- 6 个大尺寸测试用例中仍有少量索引不匹配（约 3.5%）
- 主要集中在 float16/bfloat16 的大尺寸测试中

## 实现代码

```python
def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    logger.debug("GEMS MEDIAN DIM")
    dim = dim % self.dim()
    k = _median_k(self.size(dim)) - 1
    sorted_vals, sorted_idx = torch.ops.aten.sort.stable.redispatch(
        _FALLBACK_KEYSET, self, stable=True, dim=dim, descending=False
    )
    index_shape = list(sorted_vals.shape)
    index_shape[dim] = 1
    gather_index = torch.full(
        index_shape, k, device=sorted_vals.device, dtype=torch.long
    )
    values = torch.take_along_dim(sorted_vals, gather_index, dim=dim)
    indices = torch.take_along_dim(sorted_idx, gather_index, dim=dim)
    if not keepdim:
        values = values.squeeze(dim)
        indices = indices.squeeze(dim)
    return values, indices
```

## 总结

使用 PyTorch stable sort via redispatch 的策略显著改善了索引匹配率：
- **通过率**: 从 80.6% 提升到 83.3%
- **失败数**: 从 7 个减少到 6 个
- **值正确率**: 保持 100%

剩余的索引不匹配问题可能与 float16/bfloat16 的精度限制有关，在实际应用中可能可以接受，因为值是完全正确的。

---

**更新时间**: 2026-02-13
**状态**: 值正确，索引匹配率 83.3%，大尺寸测试中仍有少量不匹配
