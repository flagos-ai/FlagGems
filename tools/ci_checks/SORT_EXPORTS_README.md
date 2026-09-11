# Auto-Sort Exports Feature

自动排序 `__all__` 导出列表和 `operators.yaml` 的三层防护机制。

## 快速开始

### 开发者本地使用

1. **安装 pre-commit hooks**（首次）：
   ```bash
   pip install pre-commit
   pre-commit install
   ```

2. **正常开发和提交**：
   ```bash
   # 修改 __init__.py 或 operators.yaml
   git add src/flag_gems/ops/__init__.py conf/operators.yaml
   git commit -m "feat: add new operator"
   # pre-commit hook 会自动排序
   ```

3. **如果 pre-commit 修改了文件**：
   ```bash
   # 文件已被自动排序，需要重新 add 并 commit
   git add -u
   git commit -m "feat: add new operator"
   ```

### 手动排序

```bash
# 检查哪些文件需要排序
python tools/ci_checks/sort_exports.py --check

# 自动修复所有文件
python tools/ci_checks/sort_exports.py --fix

# 查看修复效果（不实际修改）
python tools/ci_checks/sort_exports.py --fix --dry-run

# 只修复特定文件
python tools/ci_checks/sort_exports.py --fix --files src/flag_gems/__init__.py
```

### PR merge 后排序乱了？

如果 rebase/merge master 后排序被打乱：

**方案 1**：本地手动修复
```bash
git pull origin main
python tools/ci_checks/sort_exports.py --fix
git add -A
git commit --amend --no-edit
git push --force-with-lease
```

**方案 2**：在 PR 评论里输入 `/fix-sort`
- Bot 会自动排序并 push 到你的 PR 分支

---

## 说明

### 三层防护

```
Layer 1: pre-commit hook (本地自动修复)
  ↓ 如果开发者没装 pre-commit
Layer 2: CI 检查 (报错 + 提示修复命令)
  ↓ 如果 merge 后乱了
Layer 3: /fix-sort 评论触发 (自动修复 + push)
```

### 涉及的文件

| 文件 | 作用 |
|------|------|
| `tools/ci_checks/sort_exports.py` | 核心排序脚本 |
| `.pre-commit-config.yaml` | pre-commit hook 配置 |
| `.github/workflows/fix-sort.yaml` | `/fix-sort` 评论触发的 workflow |
| `tools/ci_checks/check_init_exports.py` | CI 检查脚本（已增加修复提示） |
| `tools/ci_checks/check_operators_yaml.py` | CI 检查脚本（已启用排序检查） |

### 排序规则

所有文件统一使用 **`casefold()` 排序**（大小写不敏感）：

```python
sorted(items, key=str.casefold)
```

示例：
```
# 正确的排序
["__ilshift__", "__irshift__", "abs", "Abs", "ABS", "add"]

# 错误（区分大小写）
["ABS", "Abs", "__ilshift__", "__irshift__", "abs", "add"]
```

---

## 排序的文件

1. **`src/flag_gems/__init__.py`** 的 `__all__` 列表（6 项）
2. **`src/flag_gems/ops/__init__.py`** 的 `__all__` 列表（1029 项）
3. **`conf/operators.yaml`** 的 `ops` 列表（1096 项）

---

## CI 行为

### 1. PR 提交后

CI 跑 `check_init_exports.py` 和 `check_operators_yaml.py`：

- ✅ **排序正确** → 检查通过
- ❌ **排序错误** → 检查失败，输出：
  ```
  ❌ src/flag_gems/ops/__init__.py: __all__ is not sorted by casefold
     Position 0: got '_nested_sum_backward', expected '__ilshift__'

  💡 To fix sorting issues, run:
     python tools/ci_checks/sort_exports.py --fix
     git add src/flag_gems/ops/__init__.py
     git commit -m 'fix: sort __all__ exports'
  ```

### 2. PR 评论 `/fix-sort`

触发 `.github/workflows/fix-sort.yaml`：

1. Checkout PR 分支
2. 运行 `sort_exports.py --fix`
3. Commit 并 push 到 PR
4. 评论结果：
   - "✅ Sorting fixed! Changes have been pushed to this PR."
   - 或 "✅ All files are already sorted correctly."

---

## FAQ

### Q: 为什么不在 CI 里自动修复？

A: CI 自动修复会有以下问题：
- PR 的 `pull_request` 事件没有写权限
- 自动 commit 会触发新的 CI 运行，可能循环
- 开发者本地代码和远程不一致

所以采用"本地 hook 自动修 + CI 检查报错 + 评论触发修复"的组合方案。

### Q: 如果我不想装 pre-commit 怎么办？

A: 可以手动跑 `sort_exports.py --fix`，或者等 CI 检查失败后根据提示修复。

### Q: `/fix-sort` 会修改我的其他代码吗？

A: 不会。`sort_exports.py` 只修改 `__all__` 列表和 `operators.yaml` 的顺序，不碰其他代码。

### Q: 排序会影响功能吗？

A: 不会。Python 的 `__all__` 和 YAML 列表的顺序不影响功能，只是为了代码可读性和 diff 友好。

---

## 开发者指南

### 添加新算子

```python
# 1. 在 src/flag_gems/ops/__init__.py 添加导入
from flag_gems.ops.new_op import new_op

# 2. 在 __all__ 列表里随便插入一个位置（不用管排序）
__all__ = [
    "abs",
    "new_op",  # 直接加，不用按顺序
    "add",
    # ...
]

# 3. 提交时 pre-commit hook 会自动排序
git add src/flag_gems/ops/__init__.py
git commit -m "feat: add new_op"
```

### 修改 operators.yaml

```yaml
# 直接在任意位置添加
ops:
  - id: abs
    ...
  - id: new_op  # 直接加，不用管顺序
    description: "..."
    for: [new_op]
    labels: [aten]
    kind: [pointwise]
    stages: [...]
  - id: add
    ...
```

pre-commit hook 会自动按 `id` 的 casefold 顺序排序。

---

## 维护者指南

### 更新排序规则

如果需要修改排序规则（比如改成按字母顺序或按类别分组），修改 `tools/ci_checks/sort_exports.py` 的 `sorted(..., key=...)` 部分。

### 禁用某个文件的排序

在 `.pre-commit-config.yaml` 的 `files` 正则里移除对应文件即可。

### 测试排序脚本

```bash
# 运行单元测试
python -m pytest tools/ci_checks/test_sort_exports.py

# 手动测试
python tools/ci_checks/sort_exports.py --check
python tools/ci_checks/sort_exports.py --fix --dry-run
```

---

## 技术细节

### 排序算法

1. **Python `__all__`**：
   - 用正则表达式匹配 `__all__ = [ ... ]`
   - 提取所有字符串项
   - 按 `casefold()` 排序
   - 保留原始格式（缩进、引号风格）

2. **YAML `operators.yaml`**：
   - 用 `pyyaml` 解析
   - 按 `op['id'].casefold()` 排序
   - 保留头部 copyright 注释
   - 用 `yaml.dump` 重新序列化

### 为什么用 casefold 而不是 lower？

`casefold()` 是 Unicode 标准的 case-folding，比 `lower()` 更通用：

```python
>>> "ß".lower() == "ss"
False
>>> "ß".casefold() == "ss"
True
```

对于英文字符，`casefold()` 和 `lower()` 效果相同，但更规范。

---

## 许可证

Apache License 2.0
