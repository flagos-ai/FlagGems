# FlagGems Vendor 特化算子生成任务

你需要为 FlagGems 项目实现一个 vendor 特化版本的 Triton 算子，替换通用层实现以修复精度或性能问题。

## 任务信息

- **算子名称**: {{OPERATOR}}
- **目标 Vendor**: {{VENDOR}}
- **设备 ID**: {{GPU_ID}}
- **工作目录**: {{WORK_DIR}} (这是一个 git worktree)
- **Python 路径**: {{PYTHON_PATH}}
- **设备命令前缀**: {{DEVICE_PREFIX}}
- **Vendor 算子目录**: {{VENDOR_OPS_DIR}}
- **问题类型**: {{ERROR_TYPE}}（为空则需自行诊断）
- **问题描述**: {{ERROR_DESC}}
- **自定义测试命令**: {{TEST_CMD}}（为空则使用默认命令）

## 运行环境说明

**重要**：本项目**不需要** `pip install`。`pytest.ini` 已配置 `pythonpath = src`，因此在工作目录（worktree 根目录）下运行 pytest 时，会自动将 `<工作目录>/src` 加入 `sys.path`，从而正确导入当前 worktree 的 `flag_gems` 代码。

- **禁止**运行 `pip install -e .` 或任何形式的 `pip install flag-gems`
- **所有命令**必须在工作目录 `{{WORK_DIR}}` 下执行
- **设备指定**：所有涉及设备的命令必须加上 `{{DEVICE_PREFIX}}` 前缀
- **环境变量**：`GEMS_VENDOR={{VENDOR}}` 已在环境中设置，FlagGems 会自动加载对应 vendor backend
- 运行测试时使用：`{{DEVICE_PREFIX}} {{PYTHON_PATH}} -m pytest ...`

## FlagGems Vendor 特化架构

```
src/flag_gems/
├── ops/                          # 通用层实现（所有 vendor 共享）
│   └── {{OPERATOR}}.py          # 你要覆盖的通用实现
└── runtime/backend/
    └── _{{VENDOR}}/
        └── ops/                  # Vendor 特化层（覆盖通用层）
            ├── __init__.py       # 注册点：import 并 __all__ 导出
            └── {{OPERATOR}}.py   # 你要创建的特化实现
```

**覆盖机制**：vendor 的 `ops/__init__.py` 中导出的函数会自动覆盖通用层的同名函数（通过 `SpecOpRegistrar`）。你只需要：
1. 在 `{{VENDOR_OPS_DIR}}/` 下创建实现文件
2. 在 `{{VENDOR_OPS_DIR}}/__init__.py` 中添加 import 和 `__all__` 条目

## 执行步骤

请严格按照以下步骤执行：

### Step 1: 阅读参考代码

**阅读该 vendor 已有的特化实现**（了解代码风格和常用模式）：

```bash
ls {{WORK_DIR}}/{{VENDOR_OPS_DIR}}/
```

选择 2-3 个已有的 `.py` 文件阅读，了解该 vendor 特化算子的：
- 代码组织方式
- 常用的硬件特有 API 或优化模式
- 函数签名风格

**阅读通用层当前实现**：

```bash
cat {{WORK_DIR}}/src/flag_gems/ops/{{OPERATOR}}.py
```

理解当前通用层实现的逻辑、函数签名、参数处理方式。你的特化实现必须保持**相同的函数签名**。

### Step 2: 诊断问题

**如果问题类型 (ERROR_TYPE) 已知**，跳到 Step 3。

**如果问题类型为空**，先运行通用实现看看出了什么问题：

```bash
cd {{WORK_DIR}}
{{DEVICE_PREFIX}} {{PYTHON_PATH}} -m pytest tests/ -m {{OPERATOR}} -vs --log-cli-level=DEBUG 2>&1
```

观察输出，判断是：
- **精度问题**（assertion error, tolerance exceeded）
- **性能问题**（测试通过但延迟过高）
- **编译/运行时错误**（crash, compilation failure, unsupported operation）

记录诊断结果，后续步骤需要针对性处理。

### Step 3: 实现 vendor 特化

在 `{{VENDOR_OPS_DIR}}/{{OPERATOR}}.py` 创建特化实现。

**要求：**
- 函数签名**必须**与通用层一致（函数名、参数列表），因为是直接覆盖
- 必须有 `import logging` 和 `logger = logging.getLogger(__name__)`
- 参考同目录下已有实现的代码风格
- 针对诊断出的问题做针对性修复：
  - 精度问题：检查类型提升、数值稳定性、特殊值处理
  - 性能问题：优化 kernel launch 参数、block size、内存访问模式
  - 编译错误：使用该 vendor 支持的 Triton 操作替代不支持的操作

### Step 4: 注册算子

在 `{{VENDOR_OPS_DIR}}/__init__.py` 中添加 import 和 `__all__` 条目。

**查看当前注册格式**：
```bash
head -30 {{WORK_DIR}}/{{VENDOR_OPS_DIR}}/__init__.py
```

按照已有格式，在适当位置（按字母顺序）添加：
```python
from .{{OPERATOR}} import function_name
```

并确保 `__all__` 列表中包含导出的函数名。如果该文件没有显式的 `__all__` 列表（用隐式导出），则只需添加 import 行。

### Step 5: 验证正确性

**使用已有测试验证**（不要写新测试）：

```bash
cd {{WORK_DIR}}
{{DEVICE_PREFIX}} {{PYTHON_PATH}} -m pytest tests/ -m {{OPERATOR}} -vs --log-cli-level=DEBUG 2>&1
```

**验证要点**：
1. 所有测试必须通过
2. DEBUG 日志中应出现 `GEMS` 相关输出，确认你的算子被调用
3. 如果使用了自定义测试命令：`{{DEVICE_PREFIX}} {{PYTHON_PATH}} -m {{TEST_CMD}}`

**如果测试失败：**
1. 分析失败原因
2. 修复 `{{VENDOR_OPS_DIR}}/{{OPERATOR}}.py`
3. 重新运行测试，直到通过

### Step 6: 性能验证

**运行 benchmark 对比特化实现与通用实现的性能**：

```bash
cd {{WORK_DIR}}
{{DEVICE_PREFIX}} {{PYTHON_PATH}} -m pytest benchmark/ -m {{OPERATOR}} -vs 2>&1
```

**特化实现至少不能比通用实现差**。如果性能反而下降了，需要重新优化。

如果问题类型是性能问题（performance），确认 benchmark 数据显示有改善。

### Step 7: 运行代码格式检查

**当测试通过且性能验证完成后**，对所有修改的文件运行 FlagGems 的 pre-commit hooks：

```bash
cd {{WORK_DIR}}

# 暂存所有修改
git add -A

# 收集所有修改的 Python 文件
MODIFIED_FILES=$(git diff --cached --name-only --diff-filter=ACMR | grep '\.py$' | tr '\n' ' ')

if [ -n "$MODIFIED_FILES" ]; then
    echo "Running pre-commit on: $MODIFIED_FILES"
    {{PYTHON_PATH}} -m pre_commit run --files $MODIFIED_FILES

    # Pre-commit 可能自动修复了一些问题，重新 add
    git add $MODIFIED_FILES
fi
```

**重要说明**：
- pre-commit 包含 **black**（格式化）、**isort**（import 排序）、**flake8**（linter）
- 大部分问题会被自动修复（black/isort）
- 如果 flake8 报告无法自动修复的问题（如未使用的变量、逻辑错误等），**必须手动修复代码**，然后重新运行 pre-commit 直到通过
- flake8 配置：`--ignore=F405,E731,W503,E203 --max-line-length=120`

**如果 pre-commit 检查失败**：
1. 查看错误信息，确定是格式问题还是代码逻辑问题
2. 修复代码后重新 `git add` 并运行 pre-commit
3. 重复直到所有检查通过

### Step 8: 更新算子目录

**当 pre-commit 检查通过后**，在 `conf/operators.yaml` 中注册新算子的元数据。

**重要**：从 FlagGems v4.2 开始，所有新算子都必须在算子目录中注册，用于跟踪其成熟度和文档。

**操作步骤**：

1. 打开 `conf/operators.yaml`
2. 找到合适的字母顺序位置（按 operator 名称排序）
3. 为每个变体添加一个条目（如 `sign`, `sign_`, `sign.out`）

**参考同类算子的格式**（如 `abs`, `ceil`, `floor` 等）：

```yaml
- id: {{OPERATOR}}
  description: |
    [算子功能的简短描述，1-2 句话]
  for:
    - {{OPERATOR}}
  labels:
    - aten
    - pointwise  # 或其他类别：reduction, norm, blas 等
  kind:
    - Math  # 或 NeuralNetwork, Tensor 等
  stages:
    - stable: '1.0'  # 或 beta: 'X.X'

- id: {{OPERATOR}}_
  description: In-place version of {{OPERATOR}}().
  for:
    - {{OPERATOR}}_
  labels:
    - aten
    - pointwise
  kind:
    - Math

- id: {{OPERATOR}}_out
  description: A variant of {{OPERATOR}}() that assigns the output to the out tensor.
  for:
    - {{OPERATOR}}.out
  labels:
    - aten
    - pointwise
  kind:
    - Math
```

**字段说明**：
- `id`: 唯一标识符（通常是算子名）
- `description`: 功能描述（参考 PyTorch 官方文档）
- `for`: ATen 算子名列表（注意 `.out` 后缀格式）
- `labels`: 分类标签（`aten`, `pointwise`, `reduction` 等）
- `kind`: 算子类型（`Math`, `NeuralNetwork`, `Tensor` 等）
- `stages`: 成熟度阶段（`beta` 或 `stable`，带版本号）

**验证 YAML 语法**：

```bash
cd {{WORK_DIR}}
{{PYTHON_PATH}} -c "import yaml; yaml.safe_load(open('conf/operators.yaml'))"
```

如果有语法错误，修复后重新验证。

**提交 operators.yaml 修改**：

```bash
git add conf/operators.yaml
```

### Step 9: 提交代码

**当所有 pre-commit 检查通过且 operators.yaml 已更新后**，提交代码：

```bash
cd {{WORK_DIR}}
git commit --author="taooo <gumptao2997@gmail.com>" -m "[{{VENDOR}}] Add {{OPERATOR}} vendor specialization"
```

### Step 10: 输出结果

**【必须】** 输出以下 JSON 格式的最终结果。用 ````json` 和 ```` ` 代码块包裹：

```json
{
  "operator": "{{OPERATOR}}",
  "vendor": "{{VENDOR}}",
  "status": "success 或 failed",
  "accuracy_passed": true/false,
  "root_cause": "诊断出的问题根因描述",
  "files_created": [
    "{{VENDOR_OPS_DIR}}/{{OPERATOR}}.py"
  ],
  "files_modified": [
    "{{VENDOR_OPS_DIR}}/__init__.py"
  ],
  "test_results": {
    "total": 12,
    "passed": 12,
    "failed": 0,
    "test_command": "pytest tests/ -m {{OPERATOR}} -vs"
  },
  "benchmark_results": {
    "benchmark_command": "pytest benchmark/ -m {{OPERATOR}} -vs",
    "data": [
      {
        "dtype": "torch.float16",
        "shape": "[1024, 1024]",
        "torch_latency_ms": 0.056,
        "gems_latency_ms": 0.045,
        "speedup": 1.24
      }
    ]
  },
  "error_message": "null 或错误描述",
  "notes": "实现说明或优化策略"
}
```

## 重要约束

1. **正确性优先**：必须通过已有的 accuracy 测试
2. **不写新测试**：验证使用已有测试，不修改 tests/ 下的文件
3. **不改通用层**：只改 `{{VENDOR_OPS_DIR}}/` 下的文件和对应的 `__init__.py`
4. **函数签名一致**：特化实现的函数名和参数必须与通用层完全一致
5. **性能不退化**：特化实现至少不能比通用实现差
6. **代码风格**：严格遵循该 vendor 目录下已有代码的风格
7. **JSON 结果必须输出**：即使失败也要输出 JSON，标明 status 为 failed
8. **禁止 pip install**：不要运行任何安装命令
9. **工作目录**：所有命令必须在 `{{WORK_DIR}}` 下执行
10. **vendor 字段**：JSON 结果中必须包含 `"vendor": "{{VENDOR}}"` 字段
