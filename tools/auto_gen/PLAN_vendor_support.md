# auto_gen vendor 特化支持 — 实施计划

## 背景

FlagGems 通用层算子在非 nvidia vendor 硬件（ascend、cambricon、iluvatar 等）上可能出现精度不对或性能不达标的问题。需要扩展 `tools/auto_gen/` 支持批量生成 vendor 特化算子，复用现有 orchestrator 架构。

---

## 改动清单

### 1. config.yaml 新增字段

```yaml
# Vendor 特化配置（null 或不填 = 通用层生成模式）
vendor: ascend

# 设备环境变量会根据 vendor 自动确定，无需手动配置
# 手动指定设备 ID（vendor 机器上无 nvidia-smi，必须手动）
device:
  gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
```

orchestrator 读取 `vendor` 字段：
- `null` → 通用层模式（行为不变）
- 非 null → vendor 特化模式

### 2. 输入格式改为 YAML

新建 `vendor_ops.yaml`（vendor 模式下使用）：

```yaml
ops:
  - operator: softmax
  - operator: layernorm
    type: accuracy_fail
    error_desc: "fp16 精度超出 rtol=1e-3"
  - operator: matmul
    type: performance
    error_desc: "比 pytorch eager 慢 2x"
    test_cmd: "pytest -m matmul tests/ -vs"
```

字段说明：
- `operator`（必填）：算子基础名
- `type`（可选）：`accuracy_fail` | `performance` | `compilation_error`，不填则 CC 自诊断
- `error_desc`（可选）：问题描述，帮助 CC 定位方向
- `test_cmd`（可选）：自定义测试命令，不填用默认 `pytest -m <op> tests/ -vs`

orchestrator 兼容：
- vendor 模式 → 读 YAML
- 通用模式 → 仍读 `ops_list.txt`（保持向后兼容）

### 3. orchestrator.py 改动

#### 3.1 新增 vendor_env_map（从 `tools/run_tests.py` 复制）

```python
VENDOR_ENV_MAP = {
    "ascend": ["ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"],
    "hygon": ["HIP_VISIBLE_DEVICES"],
    "metax": ["MACA_VISIBLE_DEVICES"],
    "mthreads": ["MUSA_VISIBLE_DEVICES"],
    "tsingmicro": ["TXDA_VISIBLE_DEVICES"],
    "iluvatar": ["ILUVATAR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"],
    "thead": ["CUDA_VISIBLE_DEVICES"],
    "cambricon": ["MLU_VISIBLE_DEVICES"],
    "kunlunxin": ["CUDA_VISIBLE_DEVICES"],
    "sunrise": ["TANG_VISIBLE_DEVICES"],
}
# 默认 fallback: ["CUDA_VISIBLE_DEVICES"]
```

#### 3.2 新增 YAML 加载函数

```python
def load_vendor_ops(path: str) -> list[dict]:
    """Load vendor ops from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    ops = data.get("ops", [])
    validated = []
    for op in ops:
        if "operator" not in op:
            logger.warning(f"Op entry missing 'operator', skipping: {op}")
            continue
        op.setdefault("type", "")
        op.setdefault("error_desc", "")
        op.setdefault("test_cmd", "")
        validated.append(op)
    return validated
```

#### 3.3 修改 create_worktree

```python
def create_worktree(flaggems_dir: str, operator: str, vendor: str = None) -> tuple[str, str]:
    if vendor:
        branch_name = f"auto-gen/{vendor}/{operator}"
        worktree_path = os.path.join(flaggems_dir, ".worktrees", f"gen-{vendor}-{operator}")
    else:
        branch_name = f"auto-gen/{operator}"
        worktree_path = os.path.join(flaggems_dir, ".worktrees", f"gen-{operator}")
    # ... 其余逻辑不变
```

#### 3.4 修改 launch_cc

```python
def launch_cc(operator, worktree_path, gpu_id, config, template_path, log_dir, vendor_op=None):
    vendor = config.get("vendor")

    # 构建设备前缀
    if vendor:
        env_vars = VENDOR_ENV_MAP.get(vendor, ["CUDA_VISIBLE_DEVICES"])
        device_prefix = " ".join(f"{v}={gpu_id}" for v in env_vars)
    else:
        device_prefix = f"CUDA_VISIBLE_DEVICES={gpu_id}"

    variables = {
        "OPERATOR": operator,
        "GPU_ID": str(gpu_id),
        "WORK_DIR": worktree_path,
        "PYTHON_PATH": config.get("python_path", "python"),
        "DEVICE_PREFIX": device_prefix,
        "VENDOR": vendor or "",
        "VENDOR_OPS_DIR": f"src/flag_gems/runtime/backend/_{vendor}/ops" if vendor else "",
        "ERROR_TYPE": (vendor_op or {}).get("type", ""),
        "ERROR_DESC": (vendor_op or {}).get("error_desc", ""),
        "TEST_CMD": (vendor_op or {}).get("test_cmd", ""),
    }
    prompt = render_template(template_path, variables)

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env["IS_SANDBOX"] = "1"
    if vendor:
        env["GEMS_VENDOR"] = vendor
    # ... 其余逻辑不变
```

#### 3.5 修改 run() 主函数

```python
# 根据 vendor 选择模板和输入
vendor = config.get("vendor")
if vendor:
    template_path = os.path.join(script_dir, config.get("vendor_template", "templates/generate_vendor_op.md"))
    ops_input_path = args.ops_list or os.path.join(script_dir, "vendor_ops.yaml")
    ops = load_vendor_ops(ops_input_path)
    queue = deque((op["operator"], 0, op) for op in ops)
else:
    template_path = os.path.join(script_dir, config.get("template", "templates/generate_op.md"))
    ops_list_path = args.ops_list or os.path.join(script_dir, "ops_list.txt")
    ops = load_ops_list(ops_list_path)
    queue = deque((op, 0, None) for op in ops)
```

#### 3.6 结果校验新增 vendor cross-check

```python
def validate_vendor_result(result: dict, vendor: str) -> bool:
    """Check that files_created paths match the declared vendor."""
    if not vendor:
        return True
    files = result.get("files_created", [])
    vendor_dir = f"_{vendor}"
    return any(vendor_dir in f for f in files)
```

### 4. 新建 templates/generate_vendor_op.md

模板大纲（与通用模板的差异标注）：

```
# FlagGems Vendor 特化算子生成任务

## 任务信息
- 算子名称: {{OPERATOR}}
- 目标 Vendor: {{VENDOR}}
- GPU ID: {{GPU_ID}}
- 工作目录: {{WORK_DIR}}
- Python 路径: {{PYTHON_PATH}}
- 设备前缀: {{DEVICE_PREFIX}}
- Vendor 算子目录: {{VENDOR_OPS_DIR}}
- 问题类型: {{ERROR_TYPE}}（为空则需自诊断）
- 问题描述: {{ERROR_DESC}}

## 执行步骤

### Step 1: 阅读参考代码
- 阅读 {{VENDOR_OPS_DIR}}/ 下 2-3 个已有实现，了解该 vendor 的代码风格
- 阅读 src/flag_gems/ops/{{OPERATOR}}.py 了解通用层当前实现

### Step 2: 诊断问题（如果 ERROR_TYPE 为空）
- 运行: {{DEVICE_PREFIX}} {{PYTHON_PATH}} -m pytest -m {{OPERATOR}} tests/ -vs
- 观察输出，判断是精度问题、性能问题还是编译错误

### Step 3: 实现 vendor 特化
- 写到: {{VENDOR_OPS_DIR}}/{{OPERATOR}}.py
- 函数签名必须与通用层一致（用于覆盖）

### Step 4: 注册
- 在 {{VENDOR_OPS_DIR}}/__init__.py 中添加 import 和 __all__ 条目

### Step 5: 验证正确性
- 运行: {{DEVICE_PREFIX}} {{PYTHON_PATH}} -m pytest -m {{OPERATOR}} tests/ -vs
- 必须全部通过

### Step 6: 验证性能（如果问题类型是 performance 或自诊断发现性能问题）
- 运行 benchmark 对比
- 特化版本至少不能比通用实现差

### Step 7: 提交代码
- git add && git commit

### Step 8: 输出 JSON 结果
- 必须包含 "vendor" 字段
- files_created 路径必须在 vendor 目录下
```

### 5. config.yaml.example 更新

新增的配置项：
```yaml
# Vendor 特化（null = 通用层模式，保持向后兼容）
vendor: null

# Vendor 特化模板路径
vendor_template: templates/generate_vendor_op.md
```

---

## 不改动的部分

- `device_manager.py` — 不改（手动指定 gpu_ids）
- 通用模式 — 行为完全不变
- 测试文件 — vendor 特化不写新测试
- `_FULL_CONFIG` — vendor 特化不动通用注册

## 文件变更汇总

| 文件 | 操作 |
|------|------|
| `orchestrator.py` | 修改（+~80 行） |
| `config.yaml.example` | 修改（+3 行） |
| `templates/generate_vendor_op.md` | 新建（~200 行） |
| `vendor_ops.yaml.example` | 新建（示例输入） |

## 测试验证

1. vendor=null 时行为不变（回归测试）
2. vendor=ascend 时：
   - 分支命名正确（`auto-gen/ascend/<op>`）
   - GEMS_VENDOR 正确设置
   - 设备前缀正确（`ASCEND_RT_VISIBLE_DEVICES=X NPU_VISIBLE_DEVICES=X`）
   - 模板选择正确
   - CC 输出的 JSON 中 vendor 字段和 files_created 一致
