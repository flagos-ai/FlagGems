# Experimental Ops 测试文件分类统计报告

**总文件数: 196**

## 分类明细

### ✅ 类别1: 标准模式 - ref在前 (180个文件)
**模式**: `ref_x = x.clone()` → 修改为 `ref_x = to_reference(x)`

这是最主要的类别，占92%。可以用正则批量替换：
```python
r'(\s+)(ref_\w+)\s*=\s*(\w+)\.clone\(\)' → r'\1\2 = to_reference(\3)'
```

---

### ✅ 类别2: 标准模式 - ref在后 (2个文件)
**模式**: `x_ref = x.clone()` → 修改为 `x_ref = to_reference(x)`

文件列表:
- expand_copy_test.py
- expm1__test.py

正则替换:
```python
r'(\s+)(\w+_ref)\s*=\s*(\w+)\.clone\(\)' → r'\1\2 = to_reference(\3)'
```

---

### ✅ 类别3: 其他变量模式 (6个文件)
**模式**: `act_input = ref_input.clone()` 或其他非ref开头的变量

文件列表:
- _functional_sym_constrain_range_for_size_test.py
- celu__test.py  
- fill__test.py
- log2__test.py
- select_backward_test.py
- slice_backward_test.py

**特殊情况**: 这些文件中，ref变量本身就是在GPU上创建的，不需要修改clone的地方，只需要在创建ref变量时使用to_reference。

---

### ⚠️ 类别4: requires_grad特殊处理 (1个文件)
**模式**: `ref_input = input_tensor.clone().requires_grad_(True)`

文件列表:
- detach_test.py

**修改方式**:
```python
# 原代码
ref_input = input_tensor.clone().requires_grad_(requires_grad)

# 修改为
ref_input = to_reference(input_tensor).requires_grad_(requires_grad)
```

---

### ⚠️ 类别5: 列表推导 (1个文件)
**模式**: `ref_tensors = [t.clone() for t in tensors]`

文件列表:
- stack_test.py

**修改方式**:
```python
# 原代码
ref_tensors = [t.clone() for t in tensors]

# 修改为  
ref_tensors = [to_reference(t) for t in tensors]
```

---

### ⚠️ 类别6: 内联clone (1个文件)
**模式**: 在函数调用中直接使用 `.clone()` 作为参数

文件列表:
- slice_scatter_test.py

示例:
```python
# 原代码
ref_out = torch.ops.aten.slice_scatter(x.clone(), src.clone(), dim, s, e, step)

# 修改为
ref_out = torch.ops.aten.slice_scatter(to_reference(x), to_reference(src), dim, s, e, step)
```

**注意**: reciprocal_test.py 和 square_test.py 虽然也有内联clone，但同时有标准模式，所以归入类别1。

---

### ✅ 类别7: 无需修改 (3个文件)
这些文件不使用 `.clone()`，无需修改

文件列表:
- _log_softmax_backward_data_test.py
- rmsnorm_test.py
- scalar_tensor_test.py

---

### ❌ 类别8: 特殊情况 - 仅有act_input (2个文件)
**模式**: 只有 `act_input = ref_input.clone()`，没有 `ref_xxx = xxx.clone()`

文件列表:
- abs__test.py
- transpose__test.py

**分析**: 这些文件中，ref_input 本身是在GPU上创建的原始变量，不需要修改。

---

## 修改策略

### 方案A: 批量自动修改 (推荐)

1. **类别1 (180个)**: 用正则批量替换 ✅
2. **类别2 (2个)**: 用正则批量替换 ✅  
3. **类别3-6 (9个)**: 手动修改或用专门脚本 ⚠️
4. **类别7-8 (5个)**: 无需修改 ✅

### 方案B: 全手动修改
工作量太大，不推荐

### 方案C: 混合方案 (最佳)

1. **先处理类别1和2 (182个文件)**: 用脚本批量添加to_reference函数和import，批量替换clone
2. **再手动处理类别3-6 (9个文件)**: 根据具体情况修改
3. **类别7-8 (5个文件)**: 跳过

---

## 统计汇总

| 类别 | 文件数 | 修改方式 | 难度 |
|------|--------|----------|------|
| 1. 标准模式-ref在前 | 180 | 自动批量 | 简单 |
| 2. 标准模式-ref在后 | 2 | 自动批量 | 简单 |
| 3. 其他变量模式 | 6 | 手动/脚本 | 中等 |
| 4. requires_grad | 1 | 手动 | 简单 |
| 5. 列表推导 | 1 | 手动 | 简单 |
| 6. 内联clone | 1 | 手动 | 中等 |
| 7. 无需修改 | 3 | 无 | - |
| 8. 特殊情况 | 2 | 无 | - |
| **总计** | **196** | - | - |

**可自动处理: 182个 (93%)**  
**需手动处理: 9个 (5%)**  
**无需修改: 5个 (2%)**
