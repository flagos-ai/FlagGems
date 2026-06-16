# FlagGems Issue 修复任务

你需要修复 FlagGems 项目中的一个已知 issue。本任务在隔离的 git worktree 中独立进行。

**注意：本 Issue ID 为内部编号（后缀 `-internal`），与 GitHub 上的公开 issue 编号无关。请勿在 commit message 或代码注释中引用为 GitHub issue 链接（如 `#441`）。**

## 任务信息

- **Issue ID**: {{ISSUE_ID}}（内部编号）
- **算子名称**: {{OPERATOR}}
- **错误类型**: {{ERROR_TYPE}}
- **严重程度**: {{SEVERITY}}
- **GPU ID**: {{GPU_ID}}
- **工作目录**: {{WORK_DIR}} (这是一个 git worktree，从 master 分支创建)
- **Python 路径**: {{PYTHON_PATH}}

## 测试命令

- **准确性测试**: `{{TEST_CMD}}`
- **性能测试**: `{{BENCHMARK_CMD}}`

## 运行环境说明

**关键：本项目不需要 `pip install`。** `pytest.ini` 已配置 `pythonpath = src`，因此在工作目录下运行 pytest 时，会自动将 `<工作目录>/src` 加入 `sys.path`。

- **禁止**运行 `pip install -e .` 或任何形式的安装命令
- **所有命令**必须在工作目录 `{{WORK_DIR}}` 下执行
- **GPU 与环境变量**：所有命令必须加上 `CUDA_VISIBLE_DEVICES={{GPU_ID}} GEMS_VENDOR={{GEMS_VENDOR}}` 前缀
- 完整命令前缀示例：
  ```
  cd {{WORK_DIR}} && CUDA_VISIBLE_DEVICES={{GPU_ID}} GEMS_VENDOR={{GEMS_VENDOR}} {{PYTHON_PATH}} -m pytest ...
  ```

## FlagGems 项目结构

```
src/flag_gems/
├── __init__.py              # 算子注册（_FULL_CONFIG 字典）
├── ops/                     # 正式算子实现（stable/beta）
├── experimental_ops/        # 实验性算子（alpha，未注册到 _FULL_CONFIG）
├── fused/                   # 融合算子（如 DSA/）
├── utils/
└── runtime/
tests/
├── test_*.py                # 单算子测试文件
├── test_DSA/                # DSA 相关测试
├── accuracy_utils.py        # 测试工具（gems_assert_close, to_reference）
└── conftest.py              # 测试配置（含 --ref cpu 参数）
benchmark/
├── test_*.py                # 性能测试
conf/
└── operators.yaml           # 全量算子定义（包含 alpha/beta/stable）
```

**算子分类与组织规则：**

- **`conf/operators.yaml`** 定义了所有算子（包括 alpha/beta/stable 三个阶段）
- **stable/beta 算子**：实现在 `src/flag_gems/ops/`，必须注册到 `__init__.py` 的 `_FULL_CONFIG`
- **alpha 算子（实验性）**：实现在 `src/flag_gems/experimental_ops/`，**不注册**到 `_FULL_CONFIG`，不会被 dispatch 到

**重要：对于 missing_test 类型的 issue**

1. **先检查算子是否已有实现**：
   ```bash
   # 检查 operators.yaml 中的 stage 定义
   grep -A10 "id: {{OPERATOR}}" conf/operators.yaml | grep "stages:"
   
   # 搜索是否在 experimental_ops/ 中已有实现
   find src/flag_gems/experimental_ops/ -name "*{{OPERATOR}}*"
   grep -r "{{OPERATOR}}" src/flag_gems/experimental_ops/
   ```

2. **根据 stage 决定策略**：
   - **stable/beta 且已有实现** → 只需补测试 + benchmark，可能需要修 bug
   - **alpha 且在 experimental_ops/** → 提升为正式算子：移动到 `ops/`、注册到 `_FULL_CONFIG`、加测试
   - **alpha 且在 experimental_ops/ 但未导出** → 在 `experimental_ops/__init__.py` 导出、注册、加测试
   - **完全没有实现** → 参考同类算子（如 adaptive_avg_pool2d）从头实现

3. **禁止重复实现**：如果 `experimental_ops/` 中已有代码，**绝对不要**在 `ops/` 中重写一份，应该复用或移动现有实现。

## 测试容差体系

`tests/accuracy_utils.py:gems_assert_close` 使用以下容差：

| dtype | rtol |
|-------|------|
| float16 | 1e-3 |
| float32 | 1.3e-6 |
| bfloat16 | 0.016 |
| float64 | 1e-7 |

`atol` 默认 1e-4，按 `reduce_dim` 缩放。

## 执行步骤

### Step 1: 复现错误

运行准确性测试，确认错误存在：

```bash
cd {{WORK_DIR}}
CUDA_VISIBLE_DEVICES={{GPU_ID}} GEMS_VENDOR={{GEMS_VENDOR}} {{PYTHON_PATH}} -m {{TEST_CMD}} 2>&1 | tail -100
```

记录：
- 失败的具体错误信息（AssertionError、RuntimeError 等）
- 失败的测试用例参数（哪些 dtype / shape 组合）
- 错误的 traceback

如果测试已经全部通过，输出 JSON 说明情况并结束（status=success, notes=already_fixed）。

### Step 2: 理解测试预期

阅读相关测试文件，理解测试在验证什么：
1. 找到测试函数（通常在 `tests/test_<op_or_module>.py`）
2. 理解输入构造方式
3. 理解 reference 实现（PyTorch 原生 vs 自行 CPU 计算）
4. 理解断言条件（`gems_assert_close` vs `gems_assert_equal`）

### Step 3: 定位算子源码

```bash
cd {{WORK_DIR}}
find src/ -name "*.py" | xargs grep -l "{{OPERATOR}}" | head -20
grep -rn "{{OPERATOR}}" src/flag_gems/__init__.py src/flag_gems/ops/__init__.py 2>/dev/null | head -10
```

阅读源码，结合 Step 1 的错误信息分析根因。

### Step 4: 诊断根因

按错误类型分类：

**accuracy_fail（精度错误）**
- 数值计算逻辑是否正确？
- 类型提升 / 转换（fp16/bf16 → fp32）是否正确？
- 是否存在溢出 / 下溢？
- 边界条件（zero, inf, nan）处理是否正确？
- 与 PyTorch reference 的语义是否一致？
- 对于 inplace 算子：是否复用了带 `@triton.autotune` 的 out-of-place kernel？autotune 多次试运行会破坏 inplace buffer
- 对于 `--ref cpu` 模式：测试是否在 GPU 上算 ref 但调用 `gems_assert_close` 失败？

**runtime_error / compilation_error**
- Triton kernel 编译错误（混合 int/float、libdevice 不支持的 dtype）
- kernel launch 参数（grid, BLOCK_SIZE, num_stages）
- tensor shape / stride 推导
- 非法内存访问

**test_error（测试本身的问题）**
- ref 没走 `to_reference()`，导致 `--ref cpu` 模式下断言失败
- 输入构造不合理
- 容差设置不匹配该算子的精度特性

**missing_test（缺少测试用例）**
- 算子已实现但没有对应的测试
- 参考同类算子的测试文件（如 `tests/test_unary_pointwise_ops.py`、`tests/test_reduction_ops.py`）编写测试
- 测试应覆盖常见 dtype（float16, float32, bfloat16）和典型 shape
- 使用 `gems_assert_close` 进行精度断言，reference 走 `to_reference()`
- 同时补充 benchmark（参考 `benchmark/` 下同类文件）
- 如果运行测试发现算子本身有 bug，一并修复

**benchmark_fail（性能测试失败）**
- benchmark 运行报错或性能严重退化
- 检查 kernel launch 参数、grid 配置、autotune 配置
- 检查是否有不必要的同步操作或内存拷贝

### Step 5: 实施修复

**修复优先级**：
1. **优先修改算子源码**（`src/flag_gems/ops/` 或 `src/flag_gems/fused/`）
2. 仅在确认测试本身有 bug 时才修改测试文件（如未走 `to_reference()` 路径）

**代码风格要求**：
- 遵循已有代码风格（black + isort + flake8 max-line-length=120）
- 保留 logger 调用
- 不引入新依赖
- **最小化修改**，只改必要的代码

### Step 6: 验证准确性测试

运行完整准确性测试：

```bash
cd {{WORK_DIR}}
CUDA_VISIBLE_DEVICES={{GPU_ID}} GEMS_VENDOR={{GEMS_VENDOR}} {{PYTHON_PATH}} -m {{TEST_CMD}} 2>&1 | tail -30
```

**所有测试必须通过**。如果失败，回到 Step 4 迭代。

### Step 7: 验证性能测试

运行 benchmark：

```bash
cd {{WORK_DIR}}
CUDA_VISIBLE_DEVICES={{GPU_ID}} GEMS_VENDOR={{GEMS_VENDOR}} {{PYTHON_PATH}} -m {{BENCHMARK_CMD}} 2>&1 | tail -30
```

可接受的结果：
- 全部 SUCCESS → 通过
- 部分 FAIL 但环境性问题（如 CUDA invalid argument）且原始 master 也失败 → 标记为 environmental，不阻塞
- 性能严重退化（speedup < 0.5）→ 需要分析

### Step 8: 代码格式检查

```bash
cd {{WORK_DIR}}
MODIFIED_FILES=$(git diff --name-only HEAD | grep '\.py$' | tr '\n' ' ')
if [ -n "$MODIFIED_FILES" ]; then
    {{PYTHON_PATH}} -m black $MODIFIED_FILES
    {{PYTHON_PATH}} -m isort --profile black $MODIFIED_FILES
    {{PYTHON_PATH}} -m flake8 --ignore=F405,E731,W503,E203 --max-line-length=120 $MODIFIED_FILES
fi
```

如果 flake8 报错，修复后重新检查。

### Step 9: 提交代码

**重要：提交前必须清理工作目录中的非代码文件。** benchmark 运行会在当前目录生成 `.log` 文件（如 `result-*.log`），这些文件**绝对不能**被 commit。

```bash
cd {{WORK_DIR}}
# 清理 benchmark 生成的 log 文件
rm -f result-*.log *.log.json

# 只提交代码文件（.py），不要用 git add -A
git add $(git diff --name-only HEAD | grep '\.py$')
git add $(git ls-files --others --exclude-standard | grep '\.py$')
git -c user.name="taooo" -c user.email="gumptao2997@gmail.com" commit -m "Fix {{ISSUE_ID}}: {{OPERATOR}} {{ERROR_TYPE}}"
```

**禁止提交的文件类型**：`.log`、`.json`（非源码）、`__pycache__/`、`.pyc`、benchmark 输出文件。如果不确定，用 `git status` 检查暂存区，确保只有 `.py` 源码文件。

### Step 10: 输出结果（必须）

无论成功失败，必须输出以下 JSON 格式（fenced code block）：

```json
{
  "issue_id": "{{ISSUE_ID}}",
  "operator": "{{OPERATOR}}",
  "error_type": "{{ERROR_TYPE}}",
  "status": "success",
  "test_passed": true,
  "benchmark_passed": true,
  "format_check_passed": true,
  "files_modified": [
    "src/flag_gems/ops/example.py"
  ],
  "root_cause": "Brief root cause description (1-2 sentences, in English)",
  "fix_description": "Brief fix description (1-2 sentences, in English)",
  "test_results": {
    "total": 12,
    "passed": 12,
    "failed": 0,
    "test_command": "{{TEST_CMD}}"
  },
  "benchmark_results": {
    "benchmark_command": "{{BENCHMARK_CMD}}",
    "all_success": true,
    "notes": "..."
  },
  "error_message": null,
  "notes": "Additional notes (optional, in English)"
}
```

## 重要约束

1. **最小修改原则** — 只改必要代码，不重构无关部分
2. **测试优先修算子** — 除非确认测试本身有 bug，否则只改算子源码
3. **必须通过 test** — accuracy test 全部通过是成功的必要条件
4. **必须通过 benchmark** — 如果有 benchmark_cmd，也需运行成功（环境性失败可豁免）
5. **代码格式** — 修改的 Python 文件必须通过 black + isort + flake8
6. **禁止 pip install** — 不运行任何安装命令
7. **工作目录** — 所有命令在 `{{WORK_DIR}}` 下执行
8. **环境前缀** — 所有命令加 `CUDA_VISIBLE_DEVICES={{GPU_ID}} GEMS_VENDOR={{GEMS_VENDOR}}`
9. **JSON 必须输出** — 即使失败也要输出 JSON
10. **禁止写临时文件** — 不要将代码写到 `/tmp`
