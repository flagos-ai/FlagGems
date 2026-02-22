# median 当前测试状态

## 测试结果总结

### ✅ 进展
- **递归问题已解决**: 新实现使用 `torch.kthvalue` 避免了递归
- **30个测试通过** (83.3%)
- **值（values）完全正确**: 所有测试中，中位数值都与参考实现一致

### ❌ 剩余问题
- **6个测试失败** (16.7%)
- **失败原因**: 索引（indices）不匹配
- **问题本质**: 当有多个相同的中位数值时，`kthvalue` 返回的索引可能与 PyTorch 的 `median` 返回的索引不同

## 问题分析

### 当前实现
```python
def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    logger.debug("GEMS MEDIAN DIM")
    dim = dim % self.dim()
    k = _median_k(self.size(dim))  # (size + 1) // 2
    return torch.kthvalue(self, k, dim=dim, keepdim=keepdim)
```

### 问题原因
当数组中有重复的中位数值时：
- `torch.kthvalue` 可能返回任意一个匹配的索引
- PyTorch 的 `median` 可能有特定的索引选择规则（例如选择第一个或最后一个）

### 示例
```python
# 输入: [1.0, 2.0, 2.0, 3.0]
# 中位数: 2.0 (第2个或第3个元素)
# kthvalue 可能返回索引 1 或 2
# PyTorch median 可能返回索引 1 (第一个匹配)
```

## 失败的测试用例

1. `dtype0-True-shape4-1` - float16, keepdim=True, shape=(1024, 1024), dim=1
2. `dtype0-False-shape4-1` - float16, keepdim=False, shape=(1024, 1024), dim=1
3. `dtype2-True-shape2-0` - bfloat16, keepdim=True, shape=(64, 64), dim=0
4. `dtype2-True-shape3-1` - bfloat16, keepdim=True, shape=(64, 64), dim=1
5. `dtype2-True-shape4-1` - bfloat16, keepdim=True, shape=(1024, 1024), dim=1
6. `dtype2-False-shape4-1` - bfloat16, keepdim=False, shape=(1024, 1024), dim=1

## 可能的解决方案

### 方案1: 检查并修复索引选择逻辑
当有重复的中位数值时，需要确保选择与 PyTorch 一致的索引。可能需要：
- 找到所有等于中位数的索引
- 选择第一个或最后一个（取决于 PyTorch 的行为）

### 方案2: 直接使用 PyTorch 的 median 实现
如果只是索引问题，可以考虑直接调用 PyTorch 的实现，但这会失去优化的意义。

### 方案3: 放宽索引检查
如果值是正确的，可以考虑在测试中放宽对索引的检查要求（但这可能不是最佳方案）。

## 测试统计

- **总测试数**: 36
- **通过**: 30 (83.3%)
- **失败**: 6 (16.7%)
- **值正确率**: 100%
- **索引正确率**: 83.3%

## 下一步

1. 调查 PyTorch `median` 的索引选择规则
2. 修复索引选择逻辑以匹配 PyTorch 的行为
3. 重新运行测试验证修复

---

**更新时间**: 2026-02-13
**状态**: 值正确，索引需要修复
