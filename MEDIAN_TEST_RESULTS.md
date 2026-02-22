# median 测试结果总结

## 测试日期
2026-02-13

## 最新实现
使用稳定排序 + 取下中位的方法：
- 使用 `torch.sort(..., stable=True)` 进行稳定排序
- 选择下中位索引 `k = (size + 1) // 2 - 1`
- 使用 `torch.take_along_dim` 获取值和原始索引

## 测试结果

### 准确性测试
- **总测试数**: 36
- **通过**: 29 (80.6%)
- **失败**: 7 (19.4%)
- **值正确率**: 100% ✅
- **索引正确率**: 80.6%

### 失败的测试用例
1. `dtype0-True-shape3-1` - float16, keepdim=True, shape=(64, 64), dim=1
2. `dtype0-True-shape4-1` - float16, keepdim=True, shape=(1024, 1024), dim=1
3. `dtype0-False-shape4-1` - float16, keepdim=False, shape=(1024, 1024), dim=1
4. `dtype2-True-shape3-1` - bfloat16, keepdim=True, shape=(64, 64), dim=1
5. `dtype2-True-shape4-1` - bfloat16, keepdim=True, shape=(1024, 1024), dim=1
6. `dtype2-False-shape3-1` - bfloat16, keepdim=False, shape=(64, 64), dim=1
7. `dtype2-False-shape4-1` - bfloat16, keepdim=False, shape=(1024, 1024), dim=1

### 问题分析
- **简单测试用例**: ✅ 索引匹配（如 `[1.0, 2.0, 2.0, 3.0]`）
- **大尺寸测试**: ❌ 仍有索引不匹配（约 0.6-3.3% 的索引不匹配）
- **可能原因**:
  1. float16/bfloat16 精度问题导致排序顺序不同
  2. 稳定排序在某些边界情况下可能仍不能完全匹配 PyTorch 的行为
  3. 需要进一步调查 PyTorch median 的具体实现逻辑

## 进展
- ✅ 递归问题已解决
- ✅ 值计算 100% 正确
- ⚠️ 索引匹配率 80.6%，大尺寸测试仍有问题

## 下一步
1. 调查 float16/bfloat16 精度对排序的影响
2. 检查 PyTorch median 在重复值时的具体索引选择规则
3. 可能需要使用不同的方法或放宽索引检查要求

---

**更新时间**: 2026-02-13
**状态**: 值正确，索引在大尺寸测试中仍有少量不匹配
