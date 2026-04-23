# EnvDetectAgent Skill — 环境检测专属

## 核心职责

执行优化流程 Step 1，完成环境检测、源代码定位与核心逻辑解析，为后续优化提供基础支撑。

## 全局约束

- 严格遵循 GCU 设备相关检测要求，优先检测 GCU400 相关环境与路径
- **不参与任何优化操作**，仅完成环境与代码的基础解析
- 环境不达标或路径不存在时，立即上报 SchedulerAgent

---

## 1. 环境检测

### 1.1 动态 Python 环境检测

**Step 0.0a: 检测当前 Python 环境**

```bash
python -c "
import torch, triton, flag_gems, os
fg_file = flag_gems.__file__
flaggems_root = os.path.abspath(os.path.join(os.path.dirname(fg_file), '..', '..'))
print('FLAGGEMS_CHECKIN=' + flaggems_root)
print('torch=' + torch.__version__)
print('triton=' + triton.__version__)
print('flag_gems=' + flag_gems.__version__)
print('device=' + str(flag_gems.device))
has_gcu = hasattr(torch, 'gcu')
print('has_gcu=' + str(has_gcu))
if has_gcu:
    print('gcu_device_count=' + str(torch.gcu.device_count()))
"
```

- 成功且 `has_gcu=True` → `PYTHON_CMD=python`，提取 `FLAGGEMS_CHECKIN`，继续后续步骤
- `has_gcu=False` → **按设备不可用流程处理**（见下方 Step 0.0b）
- 失败（import error）→ 区分原因：
  - 若 `ModuleNotFoundError` → 环境问题，尝试 Step 0.0c 自动创建
  - 若设备相关错误 → **按 SchedulerAgent Section 4.1 的设备异常处理规则判断**（白名单信号先重试，非白名单二次确认后再进入 Step 0.0b）

> **设备异常判断规则**：所有设备相关异常的判断标准和处理流程，统一参见 **SchedulerAgent Section 4.1（设备异常处理）**。EnvDetectAgent 不重复定义判断规则。

**Step 0.0b: GCU 设备不可用处理**

当 `has_gcu=False` 或**按 SchedulerAgent Section 4.1 二次确认后**设备确实不可用时，**必须通过 AskQuestion 提示用户**。**Agent 不能自行跳过或忽略设备异常**。

等待用户确认后，**重新执行 Step 0.0a**。若多次重试仍失败，进入 Step 0.0c 尝试自动创建环境。

**Step 0.0c: 自动创建环境（仅当 Step 0.0a 和 0.0b 均失败时）**

当没有现成的 Python 环境包含 flag_gems + GCU 支持时，自动创建新环境。

**1. 定位 FlagGems 项目根目录**:
```bash
for candidate in "." "../FlagGems" "../../FlagGems" "../FlagGems_checkin" "../../FlagGems_checkin"; do
  if [ -f "$candidate/pyproject.toml" ] && grep -q "flag.gems" "$candidate/pyproject.toml" 2>/dev/null; then
    echo "REPO_ROOT=$(cd "$candidate" && pwd)"
    break
  fi
done
```
- 找到 → 记录 `REPO_ROOT`
- 未找到 → 通过 AskQuestion 请用户提供 FlagGems 仓库绝对路径

**2. 检测 GCU Arch 版本**（提前检测，用于后续安装）:
```bash
python -c "
try:
    import importlib.util
    if importlib.util.find_spec('triton.backends.enflame') is not None:
        from triton.backends.enflame.driver import _GCUDriver
    else:
        from triton_gcu.triton.driver import _GCUDriver
    import re
    driver = _GCUDriver()
    arch = driver.get_arch()
    arch_version = int(re.search(r'gcu(\d+)', arch).group(1))
    print('GCU_ARCH=' + arch)
    print('GCU_ARCH_VERSION=' + str(arch_version))
except Exception as e:
    print('GCU_DETECT_ERROR=' + str(e))
"
```

**3. 创建 Conda 环境**:
```bash
ENV_NAME="flaggems_gcu_env"
conda create -n ${ENV_NAME} python=3.10 -y
```

**4. 安装 PyTorch with GCU 支持**: 通过 AskQuestion 请用户提供安装命令（依赖 GCU SDK 版本）:
```
请提供 torch + torch_gcu 的 pip install 命令（匹配您的 GCU SDK 版本）。
示例: pip install torch torch_gcu -f https://your-internal-pypi/...
```

安装后验证:
```bash
conda run -n ${ENV_NAME} python -c "import torch; print('has_gcu:', hasattr(torch, 'gcu')); print('device_count:', torch.gcu.device_count() if hasattr(torch, 'gcu') else 0)"
```

**5. 安装剩余依赖**:
```bash
conda run -n ${ENV_NAME} pip install triton  # 或 triton_gcu
conda run -n ${ENV_NAME} pip install -e ${REPO_ROOT}
conda run -n ${ENV_NAME} pip install pytest numpy scipy pyyaml packaging
# 检查 enflame 专属依赖
if [ -f "${REPO_ROOT}/flag_tree_requirements/requirements_enflame.txt" ]; then
    conda run -n ${ENV_NAME} pip install -r ${REPO_ROOT}/flag_tree_requirements/requirements_enflame.txt
fi
```

**6. 验证安装**:
```bash
conda run -n ${ENV_NAME} python -c "
import torch, triton, flag_gems, os
fg_file = flag_gems.__file__
flaggems_root = os.path.abspath(os.path.join(os.path.dirname(fg_file), '..', '..'))
print('FLAGGEMS_CHECKIN=' + flaggems_root)
print('torch=' + torch.__version__)
print('triton=' + triton.__version__)
print('flag_gems=' + flag_gems.__version__)
print('has_gcu=' + str(hasattr(torch, 'gcu')))
"
```

- 成功 → `PYTHON_CMD=conda run -n ${ENV_NAME} python`，提取 `FLAGGEMS_CHECKIN`
- 失败 → 终止并上报: `ERROR: Auto-environment creation failed.`

**Step 0.0d: 检测 GCU Arch 版本**

```bash
python -c "
try:
    import importlib.util
    if importlib.util.find_spec('triton.backends.enflame') is not None:
        from triton.backends.enflame.driver import _GCUDriver
    else:
        from triton_gcu.triton.driver import _GCUDriver
    import re
    driver = _GCUDriver()
    arch = driver.get_arch()
    arch_version = int(re.search(r'gcu(\d+)', arch).group(1))
    print('GCU_ARCH=' + arch)
    print('GCU_ARCH_VERSION=' + str(arch_version))
except Exception as e:
    print('GCU_DETECT_ERROR=' + str(e))
"
```

若检测失败，尝试回退方式：
```bash
echo "ARCH=$ARCH"
```

若仍失败，通过 AskQuestion 询问用户 GCU arch 版本（300 或 400）。

### 1.2 记录环境变量

完成检测后，确定以下变量供所有后续步骤使用：
- `PYTHON_CMD`: Python 执行命令
- `FLAGGEMS_CHECKIN`: **待修改的工作仓库**根目录（FlagGems_checkin）
- `FLAGGEMS_GOLDEN`: **不可修改的 golden reference** 仓库根目录（FlagGems 或 FlagGems_golden）
- `GCU_ARCH_VERSION`: GCU 架构版本（300 或 400）

---

## 2. 双仓库定位

### 2.1 双仓库架构说明

优化流程使用**两个独立的 FlagGems 代码仓库**，职责严格区分：

| 仓库 | 变量名 | 典型目录名 | 用途 | 可修改性 |
|------|--------|-----------|------|---------|
| **工作仓库** | `FLAGGEMS_CHECKIN` | `FlagGems_checkin` | 所有代码调优修改、优化迭代的目标仓库 | ✅ 可修改 |
| **Golden Reference** | `FLAGGEMS_GOLDEN` | `FlagGems` 或 `FlagGems_golden` | 基础性能对比和功能测试对照的不变量 | ❌ 不可修改 |

**双仓库模式**（推荐，两套独立代码）:
1. **所有代码修改**（kernel 优化、注册、迁移）仅在 `FLAGGEMS_CHECKIN` 中进行
2. **Golden Reference 用作对照基准**：运行 golden 版本的 benchmark 获取 baseline 数据，对比优化效果
3. **Golden Reference 永不修改**：不在 golden 仓库中做任何代码变更
4. 正确性验证和 benchmark 在 `FLAGGEMS_CHECKIN` 上执行（验证优化后的代码）
5. 当需要对比优化前后差异时，可在 `FLAGGEMS_GOLDEN` 上运行相同的 benchmark 命令获取 baseline

**单仓库模式**（只有一套 FlagGems 代码时）:
- `FLAGGEMS_CHECKIN` 和 `FLAGGEMS_GOLDEN` 指向**同一个目录**
- 该仓库既作为 golden reference（通过 backup 保留原始代码作为基线），又作为优化改写的目标
- **初始 baseline 必须在任何修改前先运行并记录**，此数据即为 golden 基线
- Backup 文件（`_optimize_backup/{op_name}_original.py`）充当 golden reference 的角色
- 如需恢复原始代码进行对比，从 backup 文件还原

### 2.2 仓库路径自动检测

**Step 1: 检测工作仓库 (FLAGGEMS_CHECKIN)**

优先从当前工作目录或 workspace 路径中查找包含 `FlagGems_checkin` 的目录：
```bash
# 方法 1: 从 flag_gems 安装路径推导
python -c "
import flag_gems, os
fg_file = flag_gems.__file__
checkin_root = os.path.abspath(os.path.join(os.path.dirname(fg_file), '..', '..'))
print('FLAGGEMS_CHECKIN=' + checkin_root)
"

# 方法 2: 搜索 workspace 中的 FlagGems_checkin 目录
for candidate in "." "./FlagGems_checkin" "../FlagGems_checkin" "../../FlagGems_checkin"; do
  if [ -f "$candidate/pyproject.toml" ] && grep -q "flag.gems" "$candidate/pyproject.toml" 2>/dev/null; then
    echo "FLAGGEMS_CHECKIN=$(cd "$candidate" && pwd)"
    break
  fi
done
```

**Step 2: 检测 Golden Reference (FLAGGEMS_GOLDEN)**

在 `FLAGGEMS_CHECKIN` 的同级目录中查找 golden reference 仓库：
```bash
CHECKIN_PARENT=$(dirname ${FLAGGEMS_CHECKIN})
for candidate in "${CHECKIN_PARENT}/FlagGems" "${CHECKIN_PARENT}/FlagGems_golden"; do
  if [ -d "$candidate" ] && [ -f "$candidate/pyproject.toml" ]; then
    echo "FLAGGEMS_GOLDEN=$(cd "$candidate" && pwd)"
    break
  fi
done
```

- 找到 → 记录 `FLAGGEMS_GOLDEN`，进入**双仓库模式**
- 未找到 → 进入**单仓库模式**：`FLAGGEMS_GOLDEN = FLAGGEMS_CHECKIN`，同一套代码既当 golden 又做优化改写。此时初始 baseline 数据和 backup 文件共同充当 golden reference 角色

### 2.3 项目路径约定

以下路径均相对于 `${FLAGGEMS_CHECKIN}`（工作仓库），所有修改操作在此仓库中进行：

| 类型 | 相对路径 |
|------|---------|
| 内置算子目录 | `src/flag_gems/ops/` |
| 实验算子目录 | `src/flag_gems/experimental_ops/` |
| GCU 后端算子目录 | `src/flag_gems/runtime/backend/_enflame/` |
| GCU300 后端算子 | `src/flag_gems/runtime/backend/_enflame/gcu300/ops/` |
| GCU400 后端算子 | `src/flag_gems/runtime/backend/_enflame/gcu400/ops/` |
| GCU fused 算子 (GCU300) | `src/flag_gems/runtime/backend/_enflame/fused/gcu300/` |
| GCU fused 算子 (GCU400) | `src/flag_gems/runtime/backend/_enflame/fused/gcu400/` |
| 注册文件 (backend ops) | `src/flag_gems/runtime/backend/_enflame/ops/__init__.py` |
| 注册文件 (fused ops) | `src/flag_gems/runtime/backend/_enflame/fused/__init__.py` |
| 注册文件 (experimental) | `src/flag_gems/experimental_ops/__init__.py` |
| 实验单元测试 | `experimental_tests/unit/` |
| 实验性能测试 | `experimental_tests/performance/` |
| 框架测试目录 | `tests/` |
| 框架 benchmark 目录 | `benchmark/` |

`${FLAGGEMS_GOLDEN}` 使用相同的目录结构，但仅用于读取和对比运行。

### 2.4 PATH A 代码定位（内置算子）

在 **`FLAGGEMS_CHECKIN`** 中按优先级从高到低查找算子源代码：

1. **GCU arch-specific override**（最高优先级 — 实际运行的代码）:
   - GCU400: `${FLAGGEMS_CHECKIN}/src/flag_gems/runtime/backend/_enflame/gcu400/ops/{op_name}.py`
   - GCU300: `${FLAGGEMS_CHECKIN}/src/flag_gems/runtime/backend/_enflame/gcu300/ops/{op_name}.py`
   - 检查对应 `__init__.py` 确认算子已注册

2. **GCU backend common override**:
   - `${FLAGGEMS_CHECKIN}/src/flag_gems/runtime/backend/_enflame/ops/{op_name}.py`

3. **Fused operators**:
   - `${FLAGGEMS_CHECKIN}/src/flag_gems/runtime/backend/_enflame/fused/gcu400/{op_name}.py`
   - `${FLAGGEMS_CHECKIN}/src/flag_gems/runtime/backend/_enflame/fused/gcu300/{op_name}.py`

4. **通用 FlagGems 实现**（最低优先级）:
   - `${FLAGGEMS_CHECKIN}/src/flag_gems/ops/{op_name}.py`

**当 arch-specific 实现不存在时**：通知 SchedulerAgent 需要调度 RegistrarAgent 进行代码迁移。

**Golden Reference 对照**：可在 `FLAGGEMS_GOLDEN` 的相同路径中查看原始未修改的实现，用于对比分析。

### 2.3 PATH B 代码定位（外部目录）

解析外部目录，确认以下文件存在：
- `*_triton.py` — Triton 算子文件
- `*_torch.py` — PyTorch 参考实现
- `test_*.py` — 单元测试
- `benchmark_*.py` — 性能基准测试

### 2.4 命名约定 (PATH B)

| 文件类型 | 命名格式 | 示例 |
|---------|---------|------|
| Triton 算子 | `{op_name}_triton_{gpu_name}_CC.py` | `index_put_triton_enflame_CC.py` |
| 单元测试 | `{op_name}_test_{gpu_name}_CC.py` | `index_put_test_enflame_CC.py` |
| Benchmark | `{op_name}_benchmark_{gpu_name}_CC.py` | `index_put_benchmark_enflame_CC.py` |

---

## 3. 测试文件验证

### 3.1 PATH A 测试标记检查

```bash
cd ${FLAGGEMS_CHECKIN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest tests/ -m {op_name} --collect-only 2>&1
```
- 选中 > 0 → `TEST_CMD_FLAG = -m`
- 选中 == 0 → 回退 `-k {op_name}`
- 两者均为 0 → Error: `NO_TEST`

同样检查 benchmark:
```bash
cd ${FLAGGEMS_CHECKIN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest benchmark/ -m {op_name} --collect-only 2>&1
```
- 找到 → `HAS_BENCHMARK = true`
- 未找到 → `HAS_BENCHMARK = false`，警告

---

## 4. 核心逻辑解析

阅读并解析目标算子代码，形成解析报告：

1. **实现模式**: `@pointwise_dynamic` / raw `@triton.jit` / 代码生成器 / hybrid
2. **核心计算逻辑**: 算子的数学公式与计算流程
3. **内核结构**: kernel 函数签名、参数配置、grid/block 设置
4. **GCU 特殊处理**: 是否使用 `@libentry()`、`torch_device_fn.device()`、`MAX_GRID_DIM` 等
5. **数据类型**: 支持的 dtype、是否有 FP32 累积
6. **潜在优化点**: 初步识别可优化的方向

---

## 5. Backup 创建

### PATH A
```bash
WORK_DIR="${FLAGGEMS_CHECKIN}/_optimize_backup/{op_name}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORK_DIR"
cp ${FLAGGEMS_CHECKIN}/src/flag_gems/ops/{op_name}.py "$WORK_DIR/{op_name}_original.py"
```

### PATH B
```bash
WORK_DIR="<operator_path>_flaggems_optimize_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORK_DIR/iterations"
cp <operator_path>/*.py "$WORK_DIR/"
```

---

## 6. 输出规范

向 SchedulerAgent 提交以下数据：

```
环境检测报告:
  PYTHON_CMD: str
  FLAGGEMS_CHECKIN: str          # 工作仓库路径（所有修改在此进行）
  FLAGGEMS_GOLDEN: str | None    # Golden Reference 路径（对照基准，可为 None）
  GCU_ARCH_VERSION: int (300 or 400)
  torch_version: str
  triton_version: str
  flag_gems_version: str

代码路径 (均在 FLAGGEMS_CHECKIN 中):
  source_file: str            # 算子源码路径
  test_file: str              # 正确性测试路径
  benchmark_file: str         # 性能基准测试路径
  test_cmd_flag: str           # -m 或 -k
  bench_cmd_flag: str
  has_gcu400_impl: bool        # 是否有 GCU400 专属实现
  needs_migration: bool        # 是否需要代码迁移
  golden_source_file: str | None  # Golden 中对应的源码路径（对照用）

逻辑解析报告:
  impl_pattern: str            # 实现模式
  core_logic: str              # 核心计算逻辑描述
  kernel_structure: str        # 内核结构描述
  optimization_hints: list     # 初步优化方向

备份路径:
  work_dir: str
  backup_file: str

异常信息:
  error_code: str | None
  error_message: str | None
```

---

## 7. PATH B 预检查

当 SchedulerAgent 指定 PATH B 时，EnvDetectAgent 需额外验证外部目录中的必需文件：

```
必需文件:
- *_triton.py     （Triton 算子实现）
- *_torch.py      （PyTorch 参考实现）
- test_*.py       （正确性测试）
- benchmark_*.py  （性能测试）
```

检测命令:
```bash
ls <operator_path>/*_triton*.py <operator_path>/test_*.py <operator_path>/benchmark_*.py
```

- 全部存在 → 返回文件路径列表
- 缺少任何文件 → 上报异常: `Error + 列出缺失文件名`

---

## 8. GCU 设备特性参考

- GCU300 和 GCU400 架构特性不同，arch 版本动态检测
- GCU **不支持** FP64 (`torch.float64`) — 始终使用 FP32/FP16/BF16
- GCU **有限支持** INT64 (`torch.int64`)
  - GCU300默认不支持int64，如果需要部分支持（尤其是int32转换引入了精度问题时），可以通过环境变量打开:
    export ENABLE_I64_CHECK=0 加 export TORCH_GCU_ENABLE_INT64_AND_UINT64=1
  - GCU400默认支持int64，分两种情况：
    - kernel内创建的变量，默认支持int64，不需要通过环境变量打开
    - 如果kernel参数需要支持int64 Tensor，则需要打开export TORCH_GCU_ENABLE_INT64_AND_UINT64=1
- Triton 后端: `triton.backends.enflame` 或 `triton_gcu`
- GCU 使用 `torch.gcu` 设备接口（非 `torch.cuda`）
- GCU kernel 启动使用 `torch_device_fn.device()` 上下文管理器
- GCU 有独立的 `pointwise_dynamic` 实现
- GCU 有特定的 `MAX_GRID_DIM` 和 `MAX_BLOCK_SIZE` 限制
