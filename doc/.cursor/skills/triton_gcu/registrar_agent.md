# RegistrarAgent Skill — 注册专属

## 核心职责

负责公共代码迁移（从通用实现创建 GCU400 专属实现）、算子注册，确保特化实现不影响通用实现的正确性。

## 全局约束

- 迁移与注册过程**不修改通用代码的功能**
- 严格遵循 GCU400 专属实现的命名与路径规范
- 函数签名必须与通用代码完全一致

---

## 1. 公共代码迁移

### 1.1 迁移触发条件

当 EnvDetectAgent 报告算子在 GCU400 后端无专属实现（`needs_migration = true`），而是使用通用公共代码（如 `flag_gems/ops/` 下的 `pointwise_dynamic` 等代码生成器实现）时，执行迁移。

### 1.2 迁移步骤

#### Step 1: 理解公共代码逻辑

- 阅读 `FlagGems/src/flag_gems/ops/{op_name}.py` 中的通用实现
- 理解核心计算公式（如 `celu(x) = max(0,x) + min(0, alpha*(exp(x/alpha)-1))`）
- 确认函数签名（参数名、默认值）和返回值行为
- 区分原地操作（`_` 后缀）和非原地操作

#### Step 2: 创建 GCU400 专属实现文件

在 `FlagGems/src/flag_gems/runtime/backend/_enflame/gcu400/ops/` 目录下创建 `{op_name}.py`。

**必须包含的元素**:
- `@libentry()` + `@triton.jit(do_not_specialize=[...])` 装饰器
- 1D flat kernel + grid striding 模式
- Python dispatch 函数（签名必须与公共代码完全一致）
- `.to(tl.float32)` 类型提升（保证计算精度）

**典型迁移模板**:
```python
import logging
import triton
import triton.language as tl
import torch
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

@libentry()
@triton.jit(do_not_specialize=["N_total"])
def {op_name}_kernel(x_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    block_start = pid * BLOCK
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        # ... 核心计算 ...
        tl.store(out_ptr + off, result, mask=mask)

def {op_name}(A, ...):
    logger.debug("GEMS {OP_NAME}")
    out = torch.empty_like(A)
    N_total = A.numel()
    BLOCK = 8192
    grid_size = min((N_total + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
    with torch_device_fn.device(A.device):
        {op_name}_kernel[(grid_size,)](A, out, N_total, BLOCK=BLOCK, num_warps=4)
    return out

def {op_name}_(A, ...):
    logger.debug("GEMS {OP_NAME}_")
    N_total = A.numel()
    BLOCK = 8192
    grid_size = min((N_total + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
    with torch_device_fn.device(A.device):
        {op_name}_kernel[(grid_size,)](A, A, N_total, BLOCK=BLOCK, num_warps=4)
    return A
```

### 1.3 典型迁移案例

**celu 迁移**:
- 公共代码 (`flag_gems/ops/celu.py`): 使用 `@pointwise_dynamic` 装饰器，自动生成 kernel
- GCU400 专属 (`gcu400/ops/celu.py`): 手写 1D flat Triton kernel，性能从 0.09x → 1.18x
- 注册: `from .celu import celu, celu_` + `__all__` 中添加 `"celu", "celu_"`

---

## 2. 算子注册

### 2.1 注册两步操作（缺一不可）

#### Step A: 添加 import 语句

在 `gcu400/ops/__init__.py` 顶部添加：
```python
from .{op_name} import {func1}, {func2}
```

#### Step B: 添加到 `__all__` 列表

在 `__all__ = [...]` 列表末尾添加：
```python
"{func1}", "{func2}"
```

**⚠️ 重要**: 两步必须都完成！仅有 import 缺少 `__all__` 虽然功能上可工作（因为 `BackendArchEvent.get_arch_ops()` 使用 `inspect.getmembers` 而非 `__all__`），但为了代码规范性和一致性，必须同时添加。

### 2.2 验证注册生效

```python
import os; os.environ['GEMS_VENDOR'] = 'enflame'
import torch, torch_gcu
from flag_gems.runtime import backend
event = backend.BackendArchEvent()
arch_ops = event.get_arch_ops()
target_ops = [(n, fn.__module__) for n, fn in arch_ops if '{op_name}' in n]
print(target_ops)  # 应显示 module 为 'gcu400.ops.{op_name}'
```

### 2.3 检查 Fused 层覆盖

**重要**: `_enflame/fused/__init__.py` 的 import 优先级高于 `ops` 层。如果 fused 层已有旧实现（如 mul-based `outer`），需要：
1. 检查 `fused/__init__.py` 是否导入了该算子
2. 若有，更新 fused 层 import 指向新的 ops 实现
3. 或移除 fused 层的旧导入

### 2.4 检查 _FULL_CONFIG 注册

某些算子需要在 `flag_gems/__init__.py` 的 `_FULL_CONFIG` 元组中注册。若算子未在此处注册，`use_gems()` 不会通过 `torch.library` 注册该算子，导致 PyTorch 走 decomposition 路径而非 FlagGems 实现。

---

## 3. 框架注册机制说明

- FlagGems 通过 `BackendArchEvent.get_arch_ops()` 加载 GCU400 专属算子
- 该方法使用 `inspect.getmembers(ops_module, inspect.isfunction)` 获取所有函数
- 然后通过 `replace_customized_ops()` 将这些函数替换到全局 ops 命名空间
- 替换后，`flag_gems.use_gems()` 上下文中调用 PyTorch API 时自动调用 GCU400 专属实现

---

## 4. PATH B 集成注册（Phase 4B）

当外部目录算子优化完成后，需将成果集成到 FlagGems 框架中。

### 4.1 Step 4B.1: 放置 Triton 算子

从 `$WORK_DIR/*_triton.py` 复制到 FlagGems:
```bash
cp $WORK_DIR/*_triton.py ${FLAGGEMS_CHECKIN}/src/flag_gems/experimental_ops/{op_name}_triton_enflame_CC.py
```

### 4.2 Step 4B.2: 注册到 experimental_ops/__init__.py

1. 检查是否已注册（避免重复）:
   ```python
   # 检查 __init__.py 中是否已有该算子的 import
   grep -q "{op_name}_triton_enflame_CC" ${FLAGGEMS_CHECKIN}/src/flag_gems/experimental_ops/__init__.py
   ```
2. 若未注册，添加 import 和 `__all__` 条目

### 4.3 Step 4B.3: 转换并放置单元测试

从外部目录的 `test_*.py` 转换为 FlagGems 实验测试格式:
```bash
cp $WORK_DIR/test_*.py ${FLAGGEMS_CHECKIN}/experimental_tests/unit/{op_name}_test_enflame_CC.py
```

转换要点:
- 更新 import 路径，确保引用 `flag_gems.experimental_ops.{op_name}_triton_enflame_CC`
- 确保测试文件可在 FlagGems 框架下独立运行

### 4.4 Step 4B.4: 转换并放置性能测试

```bash
cp $WORK_DIR/benchmark_*.py ${FLAGGEMS_CHECKIN}/experimental_tests/performance/{op_name}_benchmark_enflame_CC.py
```

转换要点:
- 更新 import 路径
- 确保 benchmark 输出格式与 FlagGems 标准一致（含 speedup 计算）

### 4.5 集成后验证（Phase 5B）

RegistrarAgent 放置完文件后，通知 SchedulerAgent 触发 ValidatorAgent 执行 Phase 5B 集成验证:
```bash
cd ${FLAGGEMS_CHECKIN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest -s experimental_tests/unit/{op_name}_test_enflame_CC.py -v
cd ${FLAGGEMS_CHECKIN} && GEMS_VENDOR=enflame ${PYTHON_CMD} -m pytest -s experimental_tests/performance/{op_name}_benchmark_enflame_CC.py -v
```
- 通过 → 继续 Step 6（报告）
- 失败 → Error `FAILED_INTEGRATION`

---

## 5. 命名规范 (PATH B)

| 文件类型 | 命名格式 | 示例 |
|---------|---------|------|
| Triton 算子 | `{op_name}_triton_{gpu_name}_CC.py` | `index_put_triton_enflame_CC.py` |
| 单元测试 | `{op_name}_test_{gpu_name}_CC.py` | `index_put_test_enflame_CC.py` |
| Benchmark | `{op_name}_benchmark_{gpu_name}_CC.py` | `index_put_benchmark_enflame_CC.py` |

---

## 6. 输出规范

向 SchedulerAgent 提交：

```
迁移结果:
  created_file: str              # 新创建的 GCU400 实现文件路径
  source_pattern: str            # 实现模式 (1D flat / 2D tile / pointwise_dynamic)
  functions_exported: [str]      # 导出的函数名列表

注册结果:
  init_file_updated: bool        # __init__.py 是否更新
  import_added: bool             # import 语句已添加
  all_list_updated: bool         # __all__ 已更新
  registration_verified: bool    # 注册验证通过
  target_module: str             # 验证显示的模块路径

异常信息:
  error_code: str | None
  error_message: str | None
```
